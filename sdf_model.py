
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import model
import sdf_loader
import itertools

import pickle

import random


atoms = [
	'Ac', 'Ag', 'Al', 'As', 'Au',
	'B', 'Bi', 'Br',
	'C', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cu',
	'Dy',
	'Er', 'Eu',
	'F', 'Fe',
	'Ga', 'Gd', 'Ge',
	'Hf', 'Hg',
	'I', 'In', 'Ir',
	'K',
	'La',
	'Mg', 'Mn', 'Mo',
	'N', 'Nb', 'Nd', 'Na', 'Ni',
	'O', 'Os',
	'P', 'Pb', 'Pd', 'Pt',
	'Re', 'Rh', 'Ru',
	'S', 'Sb', 'Se', 'Si', 'Sm', 'Sn',
	'Ta', 'Te', 'Th', 'Ti', 'Tl',
	'U',
	'V',
	'W',
	'Y',
	'Zn', 'Zr']
atoms_dict = {atoms[i]:i for i in range(len(atoms))}
atom_dim = len(atoms) + 2 # +2 for mass delta and charge delta

MAX_MOLECULE_SIZE = 128

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
		tmp = map(lambda x: pickle.load(open(x, 'rb')), file_names)
		tmp = itertools.chain.from_iterable(map(lambda x: x.molecules, tmp))
		self.molecules = list(filter(lambda x: x.header.atom_num < MAX_MOLECULE_SIZE, tmp))
		random.shuffle(self.molecules)

	def __len__(self):
		return len(self.molecules)

	def __getitem__(self, idx):
		# A tuple of (nodes, adjacency matrix, value)
		return (*mol_to_graph(self.molecules[idx]), torch.tensor(0) if self.molecules[idx].value<0 else torch.tensor(1))

INTERMEDIATE_LAYER_SIZE = 20
# Pyramid with 7 layers
NUM_LAYERS = 7

class SdfModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.network = model.PyramidGraphSage(NUM_LAYERS, [atom_dim] + [INTERMEDIATE_LAYER_SIZE]*NUM_LAYERS)
		self.final_layer_1 = nn.Linear(80, 15).cuda()
		self.final_layer_2 = nn.Linear(15, 2).cuda()

	def forward(self, nodes_adj):
		nodes = self.network(nodes_adj)

		# Extracting macro features from nodes via taking the 
		# min, max, mean and sum alongst every direction.
		# This is required because of the graph features being on a 
		# per-node basis.
		mx, mn, av, sm = torch.max(nodes, 1), torch.min(nodes, 1), torch.mean(nodes, 1), torch.sum(nodes, 1)
		inp = torch.cat((mx[0], mn[0], av, sm), 1)
		inp = F.relu(self.final_layer_1(inp))
		return self.final_layer_2(inp)

def train(file_names, epochs, test_files):
	print("Creating model")
	sdf_model = SdfModel().cuda()

	print("Creating training datasets")
	trainloaders = list(map(lambda x: torch.utils.data.DataLoader(
							MoleculeDataset([x]),
							batch_size=512,
							shuffle=False,
							num_workers=4), file_names))

	print("Creating test-set")
	testloader = torch.utils.data.DataLoader(
		MoleculeDataset(test_files),
		batch_size=512,
		shuffle=False,
		num_workers=4)

	optimizer = optim.Adam(sdf_model.parameters())
	criterion = nn.CrossEntropyLoss()

	print("Running propagations")
	running_loss = 0.0
	total_loss = 0.0
	for epoch in range(epochs):
		random.shuffle(trainloaders)
		for n, trainloader in enumerate(trainloaders):
			for i, data in enumerate(trainloader):
				# get the inputs
				nodes, adjs, labels = data

				nodes, adjs, labels = nodes.cuda(), adjs.cuda(), labels.cuda()
		
				# zero the parameter gradients
				optimizer.zero_grad()
		
				# forward + backward + optimize
				outputs = sdf_model((nodes, adjs))
	
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()
		
				# print statistics
				running_loss += loss.item()
				total_loss += loss.item()
				if True:
					print('[%d, %d, %5d] loss: %f' %
						  (epoch + 1, n + 1, i + 1, running_loss))
					running_loss = 0.0

		print("Total epoch loss: %f" % (total_loss,))
		total_loss = 0.0

		optimizer.zero_grad()
		true_positives = 0
		true_negatives = 0
		false_positives = 0
		false_negatives = 0

		for i, data in enumerate(testloader):
			nodes, adjs, labels = data
			nodes, adjs, labels = nodes.cuda(), adjs.cuda(), labels.cuda()
			outputs = sdf_model((nodes, adjs))
			loss = criterion(outputs, labels)
			running_loss += loss
			for j in range(outputs.shape[0]):
				if outputs[j][0] > outputs[j][1]:
					if labels[j] == 0:
						true_negatives += 1
					else:
						false_negatives += 1
				else:
					if labels[j] == 0:
						false_positives += 1
					else:
						true_positives += 1
			print("[test, %d]: test running loss: %f" % (i, running_loss))
			running_loss = 0
		print("true_pos: %d, true_neg: %d, false_pos: %d, false_neg: %d" % (true_positives, true_negatives, false_positives, false_negatives))
		optimizer.zero_grad()




