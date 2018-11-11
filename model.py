
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import copy


class GraphSageLayer(nn.Module):
	def __init__(self, input_dim, output_dim, representation_size, iterations=3):
		# input_dim: size of vector representation of incoming nodes
		# output_dim: size of node output dimension per node
		# representation_size: size of internal hidden layers
		#
		#
		#			    --find all incoming edges -> in_to_representation of source nodes--
		#			   /																   \
		# input_nodes -----node_to_rep-----------------------------------------------------CONCAT---node_update
		#			   \																   /
		#			    --find all outgoing edges -> out_to_representation of source nodes-
		#
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.representation_size = representation_size
		self.iterations = iterations
		self.other_representation = nn.Linear(input_dim, representation_size)
		self.other_to_addr = nn.Linear(input_dim, representation_size)
		self.out_to_addr = nn.Linear(input_dim, representation_size)
		self.node_to_rep = nn.Linear(input_dim, representation_size)
		self.attention = nn.Linear(representation_size*3, representation_size*3)
		self.node_update = nn.Linear(3*representation_size, output_dim)

	def cuda(self):
		self.other_representation = self.other_representation.cuda()
		self.other_to_addr = self.other_to_addr.cuda()
		self.out_to_addr = self.out_to_addr.cuda()
		self.node_to_rep = self.node_to_rep.cuda()
		self.attention = self.attention.cuda()
		self.node_update = self.node_update.cuda()
		return self

	def forward(self, nodes_adj):
		# graph_nodes_batch: (batch, node, vector)
		# graph_adj_batch: (batch, node^2)
		# graph_adj_batch is a directional adjacency matrix, can accept non-binary inputs

		# Aggregation may replaced by smarter aggregation in the future.
		# For now it is sum for simplicity and efficiency.
		
		node_representation = self.other_representation(nodes_adj[0])
		node_addr = F.normalize(self.other_to_addr(nodes_adj[0]), dim=2)

		o = torch.zeros((*node_addr.shape[:2], self.representation_size))
		if torch.cuda.is_available():
			o = o.cuda()
		addr = o
		hidden = o
		for i in range(self.iterations):
			dp = torch.einsum('bja,bia->bij', (node_addr, addr))
			dp = torch.exp(dp) # softmax based attention
			dp = dp*nodes_adj[1] # of course disconnected nodes should not affect each other...
			dp = F.normalize(dp, dim=2)
			in_src = torch.einsum('bjv,bij->biv', (node_representation, dp))
			out = self.attention(torch.cat((in_src, addr, hidden), dim=2))
			in_aggregated, addr, hidden = out[:,:,:self.representation_size] ,out[:,:,self.representation_size:self.representation_size*2], out[:,:,self.representation_size*2:]
			hidden = F.relu(hidden)

		addr = o
		hidden = o
		for i in range(self.iterations):
			dp = torch.einsum('bja,bia->bij', (node_addr, addr))
			dp = torch.exp(dp) # softmax based attention
			dp = dp*nodes_adj[1] # of course disconnected nodes should not affect each other...
			dp = F.normalize(dp, dim=1)
			out_src = torch.einsum('biv,bij->bjv', (node_representation, dp))
			out = self.attention(torch.cat((out_src, addr, hidden), dim=2))
			out_aggregated, addr, hidden = out[:,:,:self.representation_size] ,out[:,:,self.representation_size:self.representation_size*2], out[:,:,self.representation_size*2:]
			hidden = F.relu(hidden)
		
		in_aggregated = F.relu(in_aggregated)
		node_id_rep = F.relu(self.node_to_rep(nodes_adj[0]))
		out_aggregated = F.relu(out_aggregated)

		update_src = torch.cat((in_aggregated, node_id_rep, out_aggregated), dim=2)
		return F.relu(self.node_update(update_src))


class PyramidGraphSage(nn.Module):
	# This architecture allows for skip connections:
	# (Example of layout with 8 layers)
	# Input
	# | \
	# |  \
	# |   L0
	# |  /| \
	# | / |  \
	# | | |   L1
	# | | |  /| \
	# | | | / |  \
	# | | | | |   L2
	# | | | | |  /| \
	# | | | | | / |  \
	# | | | | | | |   L3
	# | | | | | | |  /
	# | | | | | | | /
	# | | | | | | L4
	# | | | | | | |
	# | | | | \ |/
	# | | | |  L5
	# | | | | /
	# | | \ |/
	# | |  L6
	# | | /
	# \ |/
	#  L7
	#  |
	# Output
	#
	# This allows efficient training with "Lazy layer training":
	# I->L7,
	# I->L0->L7,
	# I->L0->L6->L7,
	# I->L0->L1->L6->L7...
	# Effectively "training one layer at a time" continously

	def __init__(self, num_layers, feature_sizes, representation_sizes=None, batchnorm_dim=None):
		assert num_layers%2 == 0
		assert num_layers == len(feature_sizes)-1
		super().__init__()
		self.num_layers = num_layers
		if representation_sizes is None:
			representation_sizes = feature_sizes[:-1]
		self.layers = []
		self.norm_layers = []
		for i in range(self.num_layers):
			if i < self.num_layers//2:
				self.layers.append(GraphSageLayer(
					feature_sizes[i],
					feature_sizes[i+1],
					representation_sizes[i]))
			elif i == self.num_layers//2:
				self.layers.append(GraphSageLayer(
					feature_sizes[i]+feature_sizes[self.num_layers-i-1],
					feature_sizes[i+1],
					representation_sizes[i]))
			else:
				self.layers.append(GraphSageLayer(
					feature_sizes[i]+feature_sizes[self.num_layers-i]+feature_sizes[self.num_layers-i-1],
					feature_sizes[i+1],
					representation_sizes[i]))
			if batchnorm_dim:
				self.norm_layers.append(nn.BatchNorm1d(batchnorm_dim))
				

	def cuda(self):
		self.layers = list(map(lambda x: x.cuda(), self.layers))
		self.norm_layers = list(map(lambda x: x.cuda(), self.norm_layers))
		return self

	def forward(self, nodes_adj):
		fpass_graph = nodes_adj[0]
		adj = nodes_adj[1]
		stashed_results = []
		for i in range(self.num_layers):
			if i < self.num_layers//2:
				stashed_results.append(fpass_graph)
			elif i == self.num_layers//2:
				# Concatenate skip connection inputs for pyramid "downward slope"
				fpass_graph = torch.cat((fpass_graph, stashed_results[self.num_layers-i-1]), dim=2)
			else:
				# Concatenate skip connection inputs for pyramid "downward slope"
				fpass_graph = torch.cat((fpass_graph, stashed_results[self.num_layers-i], stashed_results[self.num_layers-i-1]), dim=2)
			fpass_graph = self.layers[i]((fpass_graph, adj))
			if self.norm_layers:
				fpass_graph = self.norm_layers[i](fpass_graph)
		return fpass_graph

