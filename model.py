
import networkx

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

import copy

class DirectionalGraphSage(nn.Module):
	# node_pretrans: translates the input node vector into a vector that is
	# accepted by the transformer
	# node_pretrans: (INPUT_NODE_SIZE) -> (INTERNAL_SIZE)
	#
	# edge_pretrans: translates an input edge into a vectir that us accepted
	# by the transformers
	# edge_pretrans: (INPUT_EDGE_SIZE) -> (INTERNAL_SIZE)
	#
	# paggregator, saggregator: Aggregates the successors\predecessors into a
	# single tensor in the size accepted by the transformator
	# paggregator, saggregator: 
	# (NUMNER_OF_PREDECESSORS\SUCCESORS, 2*INTERNAL_SIZE) 
	# 		-> (INTERNAL_SIZE)
	# paggregator, saggregator: [src_node1+edge1, src_node2+edge2,...] -> aggregated input
	#
	# node_transformer - accepts a representation of all predecessors and all
	# 					successors of current node, and calculates the
	# 					representation of the next layer
	# node_transformer: (3, INTERNAL_SIZE) -> (OUT_NODE_SIZE)
	# node_transformer: [pred_line, curr_line, successor_line] -> out_node_representation
	#
	# edge_transformer: (3, INTERNAL_SIZE) -> (OUT_EDGE_SIZE)
	# edge_transformer: [src_node, edge, dst_node] -> out_edge_representation
	def __init__(self, node_pretrans, edge_pretrans, paggregator, saggregator, node_transformer, edge_transformer):
		super().__init__()
		self.node_pretrans = node_pretrans
		self.edge_pretrans = edge_pretrans
		self.paggregator = paggregator
		self.saggregator = saggregator
		self.node_transformer = node_transformer
		self.edge_transformer = edge_transformer

	# Transforms the nodes and edges via the pretrans translator
	def pretrans(self, graph):
		newgraph = networkx.DiGraph()
		for node_label in graph.nodes:
			newgraph.add_node(node_label, pretrans=self.node_pretrans(graph.node[node_label]['vec']))
		for edge in graph.edges:
			newgraph.add_edge(*edge, pretrans=self.edge_pretrans(graph[edge[0]][edge[1]]['vec']))
		return newgraph

	# returns a tensor of incoming and outgoing edges of a node
	def get_preds_succs(self, graph, node_label):
		predecessors = graph.predecessors(node_label)
		successors = graph.successors(node_label)
		pred_node_to_vec = lambda x: torch.cat(
				[graph.node[x]['pretrans'],
				graph[x][node_label]['pretrans']])
		succ_node_to_vec = lambda x: torch.cat(
				[graph.node[x]['pretrans'],
				graph[node_label][x]['pretrans']])
		predecessors = map(pred_node_to_vec, predecessors)
		successors = map(succ_node_to_vec, successors)
		return predecessors, successors

	def forward(self, graph):
		newgraph = self.pretrans(graph)
		for node_label in newgraph.nodes:
			predecessors, successors = self.get_preds_succs(newgraph, node_label)
			predecessors = self.paggregator(predecessors)
			successors = self.saggregator(successors)
			newgraph.node[node_label]['vec'] = self.node_transformer(
				torch.stack(
					[predecessors,
					newgraph.node[node_label]['pretrans'],
					successors])
				)
		for edge in newgraph.edges:
			newgraph[edge[0]][edge[1]]['vec'] = self.edge_transformer(
				torch.stack(
					[newgraph.node[edge[0]]['pretrans'],
					newgraph[edge[0]][edge[1]]['pretrans'],
					newgraph.node[edge[1]]['pretrans']])
				)
		return newgraph

class LstmAggregator(nn.Module):
	def __init__(self, internal_size, shuffle=False, average_permutations=False):
		super().__init__()
		self.lstm = nn.LSTM(2*internal_size, 2*internal_size)
		self.reducer = nn.Linear(2*internal_size, internal_size)
		self.internal_size = internal_size
		self.shuffle = shuffle
		self.average_permutations = average_permutations

	def forward(self, inp):
		inp = list(inp)
		if self.average_permutations:
			raise NotImplementedError()
		else:
			if self.shuffle:
				inp = list(inp)
				random.shuffle(inp)
		hidden = (torch.zeros(1,1,self.internal_size*2), torch.zeros(1,1,self.internal_size*2))
		out = torch.zeros(1,1,self.internal_size*2)
		if inp != []:
			out, hidden = self.lstm(torch.stack(inp).view(len(inp), 1, -1), hidden)
		return F.relu(self.reducer(F.relu(out.view(-1))))

class LstmReluGraphSage(DirectionalGraphSage):
	def __init__(self, input_node_size, input_edge_size, output_node_size, output_edge_size, internal_size):
		node_pretrans_l = nn.Linear(input_node_size, internal_size)
		node_pretrans = lambda x: F.relu(node_pretrans_l(x))
		edge_pretrans_l = nn.Linear(input_edge_size, internal_size)
		edge_pretrans = lambda x: F.relu(edge_pretrans_l(x))

		paggregator = LstmAggregator(internal_size, shuffle=True)
		saggregator = LstmAggregator(internal_size, shuffle=True)

		node_transformer_l = nn.Linear(internal_size*3, output_node_size)
		node_transformer = lambda x: F.relu(node_transformer_l(x.view(-1)))
		edge_transformer_l = nn.Linear(internal_size*3, output_edge_size)
		edge_transformer = lambda x: F.relu(edge_transformer_l(x.view(-1)))

		super().__init__(
			node_pretrans,
			edge_pretrans,
			paggregator,
			saggregator,
			node_transformer,
			edge_transformer)
