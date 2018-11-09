
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
# atom_dim is 66

MAX_MOLECULE_SIZE = 128

def mol_to_sparse(molecule):
	nodes_idx = []
	node_v = []
	for i in range(len(molecule.atoms)):
		nodes_idx.append([i,atoms_dict[molecule.atoms[i].symb]])
		node_v.append(1)
		if molecule.atoms[i].dd != 0:
			nodes_idx.append([i,atom_dim-2])
			node_v.append(molecule.atoms[i].dd)
		if molecule.atoms[i].ccc != 0 or i == 0:
			# if i=0, add term for size determining
			nodes_idx.append([i,atom_dim-1])
			node_v.append(molecule.atoms[i].ccc)

	edge_idx = []
	edge_v = []
	# Add size determining block
	edge_idx.append([len(molecule.atoms)-1]*2)
	edge_v.append(0.0)
	for bond in molecule.bonds:
		a,b = molecule.atoms[bond.fst-1], molecule.atoms[bond.snd-1]
		# bond.bond_type is 1,2,3... for single, double, triple... bond
		# If both atoms are carbon or neither of them, the edge is bidirectional
		# If one atom is carbon, the connection is directional from it to the 
		# second atom
		if (a.symb == 'C') == (b.symb == 'C'):
			edge_idx.append([bond.fst-1, bond.snd-1])
			edge_v.append(bond.bond_type/2)
			edge_idx.append([bond.snd-1, bond.fst-1])
			edge_v.append(bond.bond_type/2)
			continue
		if (a.symb == 'C'):
			edge_idx.append([bond.fst-1, bond.snd-1])
			edge_v.append(bond.bond_type)
			continue
		edge_idx.append([bond.snd-1, bond.fst-1])
		edge_v.append(bond.bond_type)

	nodes_idx = torch.LongTensor(nodes_idx).t()
	node_v = torch.FloatTensor(node_v)
	edge_idx = torch.LongTensor(edge_idx).t()
	edge_v = torch.FloatTensor(edge_v)

	nds = (nodes_idx, node_v)
	edg = (edge_idx, edge_v)

	return nds, edg


def sparse_list_to_sparse(sparse_list):
	indexes = list(map(lambda x: x[0], sparse_list))
	vals = list(map(lambda x: x[1], sparse_list))
	for i in range(len(sparse_list)):
		d0 = torch.LongTensor([i]*indexes[i].shape[1])
		indexes[i] = torch.stack((d0, indexes[i][0], indexes[i][1]), dim=0)
	indexes = torch.cat((*indexes,), dim=1)

	vals = torch.cat((*vals,), dim=0)
	return torch.sparse.FloatTensor(indexes, vals)




class MoleculeDataset:
	def __init__(self, file_names):
		tmp = map(lambda x: pickle.load(open(x, 'rb')), file_names)
		tmp = itertools.chain.from_iterable(map(lambda x: x.molecules, tmp))
		tmp = filter(lambda x: x.header.atom_num < MAX_MOLECULE_SIZE, tmp)
		self.molecules = list(map(lambda x: (mol_to_sparse(x), torch.tensor(0) if x.value<0 else torch.tensor(1)), tmp))
		del tmp # Free up a lot of RAM explicitly here

	def __len__(self):
		return len(self.molecules)

	def shuffle(self):
		random.shuffle(self.molecules)

	def batch_generator(self, batch_size):
		idx = 0
		l = len(self.molecules)
		while idx < l:
			batch = self.molecules[idx:idx+batch_size]
			batch_nodes = list(map(lambda x: x[0][0], batch))
			batch_adjs = list(map(lambda x: x[0][1], batch))
			vals = list(map(lambda x: x[1], batch))

			batch_nodes = sparse_list_to_sparse(batch_nodes)
			batch_adjs = sparse_list_to_sparse(batch_adjs)
			vals = torch.LongTensor(vals)
			
			yield batch_nodes, batch_adjs, vals
			idx += batch_size


	def __getitem__(self, idx):
		# A tuple of (nodes, adjacency matrix, value)
		return (*self.molecules[idx], torch.tensor(0) if self.molecules[idx].value<0 else torch.tensor(1))

INTERMEDIATE_LAYER_SIZE = 20
# Pyramid with 7 layers
NUM_LAYERS = 13

class SdfModel(nn.Module):
	def __init__(self, iterations=8):
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
	open("log.txt", "w").write("Model initializing...")
	print("Creating model")
	sdf_model = SdfModel()
	if torch.cuda.is_available():
		sdf_model = sdf_model.cuda()

	print("Creating training datasets")
	trainloader = MoleculeDataset(file_names)

	print("Creating test-set")
	testloader = torch.utils.data.DataLoader(
		MoleculeDataset(test_files),
		batch_size=256,
		shuffle=False,
		num_workers=1)

	optimizer = optim.Adam(sdf_model.parameters())
	criterion = nn.CrossEntropyLoss()

	print("Running propagations")
	open("log.txt", "a").write("\nModel initialized!")
	running_loss = 0.0
	total_loss = 0.0
	for epoch in range(epochs):
		trainloader.shuffle()
		for i, data in enumerate(trainloader.batch_generator(256)):
			# get the inputs
			nodes, adjs, labels = data
			if torch.cuda.is_available():
				nodes, adjs, labels = nodes.cuda(), adjs.cuda(), labels.cuda()
			nodes = nodes.to_dense()
			adjs = adjs.to_dense()
	
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
				print('[%d,\t%d] loss: %f' %
					  (epoch + 1, i + 1, running_loss))
				running_loss = 0.0

		print("Total epoch loss: %f" % (total_loss,))
		open("log.txt", "a").write("\nepoch: %d, Total epoch loss: %f" % (epoch+1, total_loss))
		total_loss = 0.0

		true_positives = 0
		true_negatives = 0
		false_positives = 0
		false_negatives = 0
		running_loss = 0.0

		testloader.shuffle()
		for i, data in enumerate(testloader.batch_generator(256)):
			nodes, adjs, labels = data
			if torch.cuda.is_available():
				nodes, adjs, labels = nodes.cuda(), adjs.cuda(), labels.cuda()
			optimizer.zero_grad()
			outputs = sdf_model((nodes, adjs))
			loss = criterion(outputs, labels)
			loss.backward()
			running_loss += loss.item()
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
		open("log.txt", "a").write("\nepoch:%d, loss:%f, true_pos: %d, true_neg: %d, false_pos: %d, false_neg: %d\n" % (epoch+1, running_loss, true_positives, true_negatives, false_positives, false_negatives))
		running_loss = 0.0
		optimizer.zero_grad()




