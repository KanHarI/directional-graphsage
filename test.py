
import model

import torch

import networkx

import sdf_loader


TEST_SET = "145total-connect.sdf"

TRAINING_SETS = ["1total-connect.sdf",
				"33total-connect.sdf",
				"41total-connect.sdf",
				"47total-connect.sdf",
				"81total-connect.sdf",
				"83total-connect.sdf",
				"109total-connect.sdf",
				"123total-connect.sdf"]


def main():
	atoms = set()
	all_sets = TRAINING_SETS + [TEST_SET]
	for file_name in map(lambda x: "NCI_full\\"+x, all_sets):
		sdf = sdf_loader.SdfFile(file_name)
		for molecule in sdf.molecules:
			for atom in molecule.atoms:
				atoms.add(atom.symb)
	print(atoms)
	
if __name__ == "__main__":
	main()
