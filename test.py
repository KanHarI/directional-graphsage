
import model

import torch

import networkx

import sdf_loader
import sdf_model


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
	train_sets = map(lambda x: "NCI_full\\" + x, TRAINING_SETS)
	sdf_model.train(train_sets, 1000)
	
if __name__ == "__main__":
	main()
