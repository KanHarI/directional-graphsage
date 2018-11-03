
import networkx

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import copy

class GraphSageLayer(nn.Module):
	def __init__(self, input_dim, output_dim, representation_size):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.in_to_representation = nn.Linear(input_dim, representation_size)
		self.out_to_representation = nn.Linear(input_dim, representation_size)
		self.lstm_size = attention_index_size + representation_size
		self.node_to_rep = nn.Linear(input_dim, representation_size)
		self.node_update = nn.Linear(3*representation_size, output_dim)

	def forward(self, graph_nodes_batch, graph_adj_batch):
		# graph_nodes_batch: (batch, node, vector)
		# graph_adj_batch: (batch, node^2)
		# graph_adj_batch is a directional adjacency matrix, can accept non-binary inputs
		in_node_representation = F.elu_(self.in_to_representation(graph_nodes_batch))
		out_node_representation = F.elu_(self.out_to_representation(graph_nodes_batch))
		node_id_rep = F.elu_(self.node_to_rep(graph_nodes_batch))

		# Aggregation may replaced by smarter aggregation in the future.
		# For now it is sum for simplicity and efficiency.
		in_aggregated = torch.einsum('bjv,bij->biv', in_node_representation, graph_adj_batch)
		out_aggregated = torch.einsum('biv,bij->bjv', out_node_representation, graph_adj_batch)
		
		update_src = torch.cat((in_aggregated, node_id_rep, out_aggregated), dim=2)
		return F.tanh(self.node_update(update_src))


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
		assert num_layers == feature_sizes-1
		super().__init__()
		if representation_sizes is None:
			representation_sizes = feature_sizes[:-1]
		self.layers = []
		for i in range(num_layers):
			if i <= (num_layers-1)//2:
				self.layers.append(GraphSageLayer(
					feature_sizes[i],
					feature_sizes[i+1],
					representation_size[i]))
			else:
				self.layers.append(GraphSageLayer(
					feature_sizes[i]+feature_sizes[num_layers-i-1],
					feature_sizes[i+1],
					representation_size[i]))

	def forward(self, graph_nodes_batch, graph_adj_batch):
		fpass_graph = graph_nodes_batch
		stashed_results = []
		for i in range(num_layers):
			if i <= (num_layers-1)//2:
				stashed_results.append(fpass_graph)
				fpass_graph = self.layers[i](fpass_graph)
			else:
				fpass_graph = torch.cat((fpass_graph, stashed_results[num_layers-i-1]))
				fpass_graph = self.layers[i](fpass_graph)

