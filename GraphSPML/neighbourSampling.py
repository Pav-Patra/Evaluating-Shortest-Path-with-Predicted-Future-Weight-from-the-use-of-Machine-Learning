import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, is_undirected, to_undirected
from torch_geometric.loader import NeighborLoader
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import Incremental_Shortest_Path as ISP 
import os
import random

from torch_geometric.nn import NNConv, graclus


# test the capabilites of NeighborLoader which makes subgraphs from a graph

# first generate an example Data object

g = ISP.Graph()
numNodes = 100

g.generateGraph(numNodes)

g.drawGraph()

g.incrementalShortestPath()

#g.printTree()

plt.ion()
plt.show()

inp = int(input("Done"))

g.createStoreEdgList("99")

fileName = "graph99.edgelist"

with open('C:/Users/pavpa/OneDrive/Documents/CS Uni/cs310/Project Practice Code/node2vec-deepLearning-distApprox/data/subsetGraphs/'+fileName) as edgeFile:
    lines = edgeFile.readlines()

os.remove('C:/Users/pavpa/OneDrive/Documents/CS Uni/cs310/Project Practice Code/node2vec-deepLearning-distApprox/data/subsetGraphs/'+fileName)

src = []
dst = []
weight = []

for line in lines:
    vals = line.split()
    src.append(int(vals[0]))
    dst.append(int(vals[1]))
    weight.append(int(vals[2]))
    # since graph is undirected
    src.append(int(vals[1]))
    dst.append(int(vals[0]))
    weight.append(int(vals[2]))
print(src)
print(dst)
print(weight)

# form node embeddings by traversing the shortest path tree for g and adding the shortest path weights for each node number connection
vector = []
nodeVectors = []

# for node in g.trees:
#     vector = []
#     for path_weight in node:       # issue with unconnected nodes
#         if path_weight.child[1] is not None:
#             vector.append(float(path_weight.child[1]))
#         else:
#             vector.append(float('inf'))
#         print(f"src name: {path_weight.child[0]} weight: {path_weight.child[1]}")
#     nodeVectors.append(vector)

# print(nodeVectors)

for n in g.addedNodes:
    vec = [n]
    vector.append(vec) 

print(vector)


# define the pytorch_geometric Data class
edge_index = torch.tensor([src, dst], dtype=torch.long)
edge_weight = torch.tensor(weight, dtype=torch.long)
x = torch.tensor(vector, dtype=torch.float)

# Create a PyTorch Geometric Data object with edge weights
data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
print(data)
print(data.edge_attr)

batchNum = 2

loader1 = NeighborLoader(
    data,
    input_nodes=torch.tensor([random.randint(0, numNodes-1)]),
    num_neighbors=[-1,-1],    # all neighbours sampled for 2 hops
    batch_size=batchNum,
    replace=False,
    shuffle=False,
    subgraph_type='induced',    # directed=False deprecated, recommended to do this instead
)

loader2 = NeighborLoader(
    data,
    input_nodes=torch.tensor([random.randint(0, numNodes-1)]),
    num_neighbors=[-1,-1],    # all neighbours sampled for 2 hops
    batch_size=batchNum,
    replace=False,
    shuffle=False,
    subgraph_type='induced',    # directed=False deprecated, recommended to do this instead
)

loader3 = NeighborLoader(
    data,
    input_nodes=torch.tensor([random.randint(0, numNodes-1)]),
    num_neighbors=[-1,-1],    # all neighbours sampled for 2 hops
    batch_size=batchNum,
    replace=False,
    shuffle=False,
    subgraph_type='induced',    # directed=False deprecated, recommended to do this instead
)


batch1 = next(iter(loader1))
batch2 = next(iter(loader2))
bnatch3 = next(iter(loader3))

batches = [batch1, batch2, bnatch3]

for batch in batches:
    print(batch)
    print(batch.n_id[batch.edge_index])    # output edge_index of subgraph with original node lables 
    print(batch.n_id)
    print(batch.edge_attr)
    print(batch.batch_size)
    print(batch.x)

    subData = Data(x=batch.x, edge_index=batch.n_id[batch.edge_index], edge_attr=batch.edge_attr)

    subgraph = to_networkx(subData, to_undirected=True)  # issue where duplicate nodes are drawn. Might be the case
                                                         # because .edge_index contains 2 edges for each node pair to 
                                                         # represent an undirected graph
    # clear plot
    plt.clf()
    plt.cla()

    pos = nx.spring_layout(subgraph, seed=7)
    nx.draw(subgraph, with_labels=True)
    #labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx(subgraph, pos, )


    plt.show()

    inp = int(input("Done"))
