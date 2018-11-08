
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
NUM_LAYERS = 13

class SdfModel(nn.Module):
	def __init__(self, iterations=5):
		super().__init__()
		self.network = model.PyramidGraphSage(NUM_LAYERS, [atom_dim] + [INTERMEDIATE_LAYER_SIZE]*NUM_LAYERS)
		self.node_to_representations = nn.Linear(INTERMEDIATE_LAYER_SIZE, INTERMEDIATE_LAYER_SIZE)
		self.node_to_addresses = nn.Linear(INTERMEDIATE_LAYER_SIZE, INTERMEDIATE_LAYER_SIZE)
		self.attention = nn.Linear(INTERMEDIATE_LAYER_SIZE*3, INTERMEDIATE_LAYER_SIZE*3)
		self.final_layer_1 = nn.Linear(INTERMEDIATE_LAYER_SIZE, INTERMEDIATE_LAYER_SIZE)
		self.final_layer_2 = nn.Linear(INTERMEDIATE_LAYER_SIZE, 2)
		self.iterations = iterations
			

	def cuda(self):
		self.network = self.network.cuda()
		self.node_to_representations = self.node_to_representations.cuda()
		self.node_to_addresses = self.node_to_addresses.cuda()
		self.attention = self.attention.cuda()
		self.final_layer_1 = self.final_layer_1.cuda()
		self.final_layer_2 = self.final_layer_2.cuda()
		return self

	def forward(self, nodes_adj):
		nodes = self.network(nodes_adj)

		o = torch.zeros((nodes.shape[0], INTERMEDIATE_LAYER_SIZE))
		if torch.cuda.is_available():
			o = o.cuda()
		addr = o
		hidden = o

		# Extracting macro features from nodes
		nodes_rep = self.node_to_representations(nodes)
		nodes_addr = F.normalize(self.node_to_addresses(nodes), dim=2)
		for i in range(self.iterations):
			dp = torch.einsum('bja,ba->bj', (nodes_addr, addr))
			dp = torch.exp(dp) # softmax based attention
			dp = F.normalize(dp, dim=1)
			in_src = torch.einsum('bjv,bj->bv', (nodes_rep, dp))
			out = self.attention(torch.cat((in_src, addr, hidden), dim=1))
			in_aggregated, addr, hidden = out[:,:INTERMEDIATE_LAYER_SIZE] ,out[:,INTERMEDIATE_LAYER_SIZE:INTERMEDIATE_LAYER_SIZE*2], out[:,INTERMEDIATE_LAYER_SIZE*2:]
			hidden = F.relu(hidden)

		return self.final_layer_2(F.relu(self.final_layer_1(in_aggregated)))


def train(file_names, epochs, test_files):
	log = open("log.txt", "w")
	print("Creating model")
	sdf_model = SdfModel()
	if torch.cuda.is_available():
		sdf_model = sdf_model.cuda()

	print("Creating training datasets")
	trainloaders = list(map(lambda x: torch.utils.data.DataLoader(
							MoleculeDataset([x]),
							batch_size=256,
							shuffle=False,
							num_workers=1), file_names))

	print("Creating test-set")
	testloader = torch.utils.data.DataLoader(
		MoleculeDataset(test_files),
		batch_size=256,
		shuffle=False,
		num_workers=1)

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

				if torch.cuda.is_available():
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
					print('[%d,\t%d,\t%d] loss: %f' %
						  (epoch + 1, n + 1, i + 1, running_loss))
					running_loss = 0.0

		print("Total epoch loss: %f" % (total_loss,))
		total_loss = 0.0

		optimizer.zero_grad()
		true_positives = 0
		true_negatives = 0
		false_positives = 0
		false_negatives = 0
		running_loss = 0.0

		for i, data in enumerate(testloader):
			nodes, adjs, labels = data
			if torch.cuda.is_available():
				nodes, adjs, labels = nodes.cuda(), adjs.cuda(), labels.cuda()
			outputs = sdf_model((nodes, adjs))
			loss = criterion(outputs, labels)
			running_loss += loss.item()
			optimizer.zero_grad()
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
			print("[test,\t%d]: loss: %f" % (i, loss.item()))
		print("epoch:%d, loss:%f, true_pos: %d, true_neg: %d, false_pos: %d, false_neg: %d" % (epoch+1, running_loss, true_positives, true_negatives, false_positives, false_negatives))
		log.write("epoch:%d, loss:%f, true_pos: %d, true_neg: %d, false_pos: %d, false_neg: %d\n" % (epoch+1, running_loss, true_positives, true_negatives, false_positives, false_negatives))
		running_loss = 0.0
		optimizer.zero_grad()




