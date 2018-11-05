
import networkx

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import copy

class GraphSageLayer(nn.Module):
	def __init__(self, input_dim, output_dim, representation_size):
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
		self.in_to_representation = nn.Linear(input_dim, representation_size)
		self.out_to_representation = nn.Linear(input_dim, representation_size)
		self.node_to_rep = nn.Linear(input_dim, representation_size)
		self.node_update = nn.Linear(3*representation_size, output_dim)

	def forward(self, nodes_adj):
		# graph_nodes_batch: (batch, node, vector)
		# graph_adj_batch: (batch, node^2)
		# graph_adj_batch is a directional adjacency matrix, can accept non-binary inputs
		in_node_representation = F.elu_(self.in_to_representation(nodes_adj[0]))
		out_node_representation = F.elu_(self.out_to_representation(nodes_adj[0]))
		node_id_rep = F.elu_(self.node_to_rep(nodes_adj[0]))

		# Aggregation may replaced by smarter aggregation in the future.
		# For now it is sum for simplicity and efficiency.
		in_aggregated = torch.einsum('bjv,bij->biv', (in_node_representation, nodes_adj[1]))
		out_aggregated = torch.einsum('biv,bij->bjv', (out_node_representation, nodes_adj[1]))
		
		update_src = torch.cat((in_aggregated, node_id_rep, out_aggregated), dim=2)
		return torch.tanh(self.node_update(update_src))


class PyramidGraphSage(nn.Module):
	# This architecture allows for skip connections:
	# Input
	# | \
	# |  L0
	# |  | \
	# |  |  L1
	# |  | /
	# |  L2
	# | /
	# L3 -> Output
	def __init__(self, num_layers, feature_sizes, representation_sizes=None):
		assert num_layers == len(feature_sizes)-1
		super().__init__()
		self.num_layers = num_layers
		if representation_sizes is None:
			representation_sizes = feature_sizes[:-1]
		self.layers = []
		for i in range(self.num_layers):
			if i <= (self.num_layers-1)//2:
				self.layers.append(GraphSageLayer(
					feature_sizes[i],
					feature_sizes[i+1],
					representation_sizes[i]))
			else:
				self.layers.append(GraphSageLayer(
					feature_sizes[i]+feature_sizes[self.num_layers-i-1],
					feature_sizes[i+1],
					representation_sizes[i]))

	def forward(self, nodes_adj):
		fpass_graph = nodes_adj[0]
		adj = nodes_adj[1]
		stashed_results = []
		for i in range(self.num_layers):
			if i <= (self.num_layers-1)//2:
				stashed_results.append(fpass_graph)
				fpass_graph = self.layers[i]((fpass_graph, adj))
			else:
				# Concatenate skip connection inputs for pyramid "downward slope"
				fpass_graph = torch.cat((fpass_graph, stashed_results[self.num_layers-i-1]), dim=2)
				fpass_graph = self.layers[i]((fpass_graph, adj))
		return fpass_graph

