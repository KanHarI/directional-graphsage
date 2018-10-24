
import networkx

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

class DirectionalGraphSage(nn.model):
	# saggregator - successor aggregator
	# paggregator - predecessor aggregator
	# paggregator, saggregator: map object -> torch vector of the size of a 
	# 							line acceptable by transformator
	# paggregator, saggregator: (NUMNER_OF_PREDECESSORS\SUCCESORS, line_size) -> line_size
	#
	# transformator - accepts a representation of all predecessors and all
	# 					successors of current node, and calculates the
	# 					representation of the next layer
	# transformator: (3, line_size) -> line size
	# transformator: (pred_line, curr_line, successor_line) -> line_size
	def __init__(self, paggregator, saggregator, transformator):
		self.paggregator = paggregator
		self.saggregator = saggregator
		self.transformator = transformator

	# graph: an networkx DiGraph object
	# Every node of the graph is expected to have a property 'vec' which is a
	# 1d pytorch tensor of length equal to transformator's line size
	def forward(self, graph):
		newgraph = copy.deepcopy(graph)
		for node_label in list(graph.nodes):
			predecessors = map(lambda x: x['vec'], graph.predecessors(node_label))
			successors = map(lambda x: x['vec'], graph.successors(node_label))
			predecessors = self.paggregator(predecessors)
			successors = self.saggregator(successors)
			newgraph[node_label]['vec'] = self.transformator(torch.stack([predecessors, graph.nodes[node_label]['vec'], successors]))
		return newgraph
