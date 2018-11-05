
import model

import torch

import sdf_loader
import sdf_model

import pickle


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
	for ts in TRAINING_SETS:
		pickle.dump(sdf_loader.SdfFile("NCI_full\\" + ts), open(TRAINING_SETS + ".pickle.bin", 'wb'))
	# train_sets = map(lambda x: "NCI_full\\" + x, TRAINING_SETS)
	# test_set = ["NCI_full\\" + TEST_SET]
	# sdf_model.train(train_sets, 100, test_set)
	
if __name__ == "__main__":
	main()
