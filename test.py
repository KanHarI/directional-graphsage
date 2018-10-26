
import model

import torch

import networkx

def main():
	m = model.LstmReluGraphSage(1,1,2,2,2)
	g = networkx.DiGraph()
	g.add_node(1, vec=torch.tensor([2], dtype=torch.float32))
	g.add_node(2, vec=torch.tensor([3], dtype=torch.float32))
	g.add_edge(1,2, vec=torch.tensor([4], dtype=torch.float32))
	g2 = m.forward(g)


	print(g2.node[1])
	print(g2[1][2])
	print(g2.node[2])
	
if __name__ == "__main__":
	main()
