
import networkx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import model
import sdf_loader
import itertools

atoms = ['B', 'Bi', 'Br', 'C', 'Cl', 'Cr', 'Cu', 'F', 'Hg', 'N', 'Na', 'Ni', 'O', 'P', 'Pt', 'Rh', 'S', 'Si', 'Sn', 'Zn']
atoms_dict = {atoms[i]:i for i in range(len(atoms))}
atom_dim = len(atoms) + 2 # +2 for mass delta and charge delta

MAX_MOLECULE_SIZE = 100

def mol_to_graph(molecule):
	node_vals = torch.zeros((MAX_MOLECULE_SIZE, atom_dim), dtype=torch.float32)
	for i in range(len(molecule.atoms)):
		# An atom is represented as:
		#        atom_type('C')   mass charge
		#            |              |   /
		#            V              V  V 
		# [0,0,...,0,1,0,0...0,0,0,dd,ccc]
		node_vals[i, atoms_dict[molecule.atoms[i].symb]] = 1
		node_vals[i, atom_dim-2] = molecule.atoms[i].dd # mass delta
		node_vals[i, atom_dim-1] = molecule.atoms[i].ccc # charge delta

	adj_matrix = torch.zeros((MAX_MOLECULE_SIZE,MAX_MOLECULE_SIZE), dtype=torch.float32)
	for bond in molecule.bonds:
		a,b = molecule.atoms[bond.fst-1], molecule.atoms[bond.snd-1]
		# bond.bond_type is 1,2,3... for single, double, triple... bond
		# If both atoms are carbon or neither of them, the edge is bidirectional
		# If one atom is carbon, the connection is directional from it to the 
		# second atom
		if (a.symb == 'C') == (b.symb == 'C'):
			adj_matrix[bond.fst-1, bond.snd-1] = bond.bond_type/2 # halving to preserve the same average
			adj_matrix[bond.snd-1, bond.fst-1] = bond.bond_type/2 # degree between nodes
			continue
		if (a.symb == 'C'):
			adj_matrix[bond.fst-1, bond.snd-1] = bond.bond_type
			continue
		adj_matrix[bond.snd-1, bond.fst-1] = bond.bond_type
	
	return node_vals, adj_matrix


class MoleculeDataset(data.Dataset):
	def __init__(self, file_names):
		tmp = map(sdf_loader.SdfFile, file_names)
		tmp = list(itertools.chain.from_iterable(map(lambda x: x.molecules, tmp)))
		self.molecules = list(map(mol_to_graph, tmp))
		self.vals = list(map(lambda x: torch.tensor(0) if x.value<0 else torch.tensor(1), tmp))
		assert len(self.molecules) == len(self.vals)

	def __len__(self):
		return len(self.molecules)

	def __getitem__(self, idx):
		# A tuple of (node, adjacency matrix, value)
		return self.molecules[idx][0], self.molecules[idx][1], self.vals[idx]


INTERMEDIATE_LAYER_SIZE = 10
# Pyramid with 7 layers
NUM_LAYERS = 7

class SdfModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.network = model.PyramidGraphSage(NUM_LAYERS, [atom_dim] + [INTERMEDIATE_LAYER_SIZE]*NUM_LAYERS)
		self.final_layer_1 = nn.Linear(40, 9)
		self.final_layer_2 = nn.Linear(9, 2)

	def forward(self, nodes_adj):
		nodes = self.network(nodes_adj)

		# Extracting macro features from nodes via taking the 
		# min, max, mean and sum alongst every direction.
		# This is required because of the graph features being on a 
		# per-node basis.
		mx, mn, av, sm = torch.max(nodes, 1), torch.min(nodes, 1), torch.mean(nodes, 1), torch.sum(nodes, 1)
		inp = torch.cat((mx[0], mn[0], av, sm), 1)
		inp = F.elu_(self.final_layer_1(inp))
		return self.final_layer_2(inp)

def train(file_names, epochs):
	sdf_model = SdfModel()

	trainloader = torch.utils.data.DataLoader(
		MoleculeDataset(file_names),
		batch_size=128,
		shuffle=True,
		num_workers=4)

	optimizer = optim.Adam(sdf_model.parameters())
	criterion = nn.CrossEntropyLoss()

	running_loss = 0.0
	for epoch in range(epochs):
		for i, data in enumerate(trainloader):
			# get the inputs
			nodes, adjs, labels = data
	
			# zero the parameter gradients
			optimizer.zero_grad()
	
			# forward + backward + optimize
			outputs = sdf_model((nodes, adjs))

			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
	
			# print statistics
			running_loss += loss.item()
			if i == 5: # Print the 5th mini-batch of every epoch
				print('[%d, %5d] loss: %f' %
					  (epoch + 1, i + 1, running_loss))
				running_loss = 0.0
