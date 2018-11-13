
import model

import torch

import sdf_loader
import sdf_model


TEST_SET = ["NCI_pickled/145total-connect.sdf.pickle.bin"]

TRAINING_SETS = ["NCI_pickled/1total-connect.sdf.pickle.bin"]#,
				#"NCI_pickled/33total-connect.sdf.pickle.bin",
				#"NCI_pickled/41total-connect.sdf.pickle.bin",
				#"NCI_pickled/47total-connect.sdf.pickle.bin",
				#"NCI_pickled/81total-connect.sdf.pickle.bin",
				#"NCI_pickled/83total-connect.sdf.pickle.bin",
				#"NCI_pickled/109total-connect.sdf.pickle.bin",
				#"NCI_pickled/123total-connect.sdf.pickle.bin"]


def main():
	sdf_model.train(TRAINING_SETS, 200, TEST_SET)
	
if __name__ == "__main__":
	main()
