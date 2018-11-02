
import networkx

import torch
import torch.nn as nn
import torch.nn.functional as F

import model

atoms = ['B', 'Bi', 'Br', 'C', 'Cl', 'Cr', 'Cu', 'F', 'Hg', 'N', 'Na', 'Ni', 'O', 'P', 'Pt', 'Rh', 'S', 'Si', 'Sn', 'Zn']

atom_dim = len(atoms) + 2 # +2 for mass delta and charge delta

class SdfModel(nn.Module):
	def __init__(self):
		self.input_node_size = atom_dim
		self.input_edge_size = 1 # bond type
