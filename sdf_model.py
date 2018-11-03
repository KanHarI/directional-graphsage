
import networkx

import torch
import torch.nn as nn
import torch.nn.functional as F

import model
import sdf_loader

atoms = ['B', 'Bi', 'Br', 'C', 'Cl', 'Cr', 'Cu', 'F', 'Hg', 'N', 'Na', 'Ni', 'O', 'P', 'Pt', 'Rh', 'S', 'Si', 'Sn', 'Zn']
atoms_dict = {atoms[i]:i for i in range(len(atoms))}
atom_dim = len(atoms) + 2 # +2 for mass delta and charge delta

def mol_to_graph(molecule):
	node_vals = torch.zeros((len(molecule.atoms), atom_dim), dtype=torch.float32)
	for i in range(len(molecule.atoms)):
		node_vals[i, atoms_dict[molecule.atoms[i].symb]] = 1
		node_vals[i, atom_dim-2] = molecule.atoms[i].dd # mass delta
		node_vals[i, atom_dim-1] = molecule.atoms[i].ccc # charge delta

	adj_matrix = torch.zeros((len(molecule.atoms),len(molecule.atoms)), dtype=torch.float32)
	for bond in molecule.bonds:
		a,b = molecule.atoms[bond.fst-1], molecule.atoms[bond.snd-1]
		if (a.symb == 'C') == (b.symb == 'C'):
			adj_matrix[bond.fst-1, bond.snd-1] = bond.bond_type/2 # 1,2,3... for single, double, triple...
			adj_matrix[bond.snd-1, bond.fst-1] = bond.bond_type/2 # halving to preserve the same average
			continue												# degree between different bonds
		if (a.symb == 'C'): # prefer "outbound" connection from carbon atoms
			adj_matrix[bond.fst-1, bond.snd-1] = bond.bond_type # as a way to give the network
			continue							# more information
		adj_matrix[bond.snd-1, bond.fst-1] = bond.bond_type
	
	return node_vals, adj_matrix


class SdfGraphLoader(SdfFile):
	def __init__(self, file_name):
		super().__init__(file_name)
		self.graphs = list(map(mol_to_graph, self.molecules))

class SdfModel(nn.Module):
	def __init__(self):
		self.input_node_size = atom_dim
