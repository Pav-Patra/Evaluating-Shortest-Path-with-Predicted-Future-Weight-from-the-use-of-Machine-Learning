import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import Incremental_Shortest_Path as ISP
import os
import random
from torch_geometric.nn import GCNConv 
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.loader import NeighborLoader

from torch_geometric.datasets import Planetoid

import networkx as nx



# purpose of this neural network is to take a graph dataset (trainable),
# convert to a tensor where the node attributes is an array of the distance of each node to all other nodes
# edge weights stored in node_attr
# determine whether the paths between two specified nodes has changed in relation the stored shortest path held in the node vectors


# may not be able to craft a neural network that can handle different node vector dimensions
# the original solution stored shortest path tree for each node to all other nodes as a matrix
# however for graphs woth different size the matrix would have different dimensions
# the solution to this would be to have each vector simply store the node number
# for a certain neighbourhood, check if src and dst are in it
# the old shortest path would be passed as a parameter into the nn
# convolutions on the edge embeddings to check whether the edge values between these 2 nodes have changed
# this indicates whether the ISP algo needs executing

# define a diskstra-like function which takes in src, dst edge_index and edge attributes and returns the true shortest path in a neighbourhood/graph embedding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, dropout_rate=0.5):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(num_features, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.convs.append(self.conv1)
        self.convs.append(self.conv2)


        self.p = dropout_rate

    def forward(self, src, dst, distances, data):      # src and dst nodes are the interested paths
        x, edge_index = data, data.edge_index
        
        # x.self.conv1(x, edge_index)
        # x = F.relu(x)
        # output = self.conv2(x, edge_index)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
            

        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu()) # we only need the representations of the target nodes
            x_all = torch.cat(xs, dim=0)
        return x_all


    
def data_to_networkx_graph(data):
    # manually convert tensor graph data to networkx weighted undirected graph

    plt.clf()
    plt.cla()
    g = nx.Graph()
    addedNodes = []
    addedEdges = []

    edge_index = data.edge_index 
    i = 0
    for i in range(len(edge_index[0])):
        x = int(edge_index[0][i].item())
        y = int(edge_index[1][i].item())
        print(x,y)

        if x not in addedNodes:
            g.add_node(x, pos=(0,0))
            addedNodes.append(x)

        if y not in addedNodes:
            g.add_node(y, pos=(0,0))
            addedNodes.append(y)

        if (x,y) not in addedEdges or (y,x) not in addedEdges:
            g.add_edge(x, y, weight=int(data.edge_attr[i].item()))
            addedEdges.append((x, y))

    pos = nx.spring_layout(g, seed=7)     # positions for all nodes - seed for reproducibility
    #pos = nx.get_node_attributes(self.draw_graph, 'pos')
    nx.draw(g, pos, with_labels=True)
    labels = nx.get_edge_attributes(g,'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

    inp = int(input("Done"))

    return g

    


def subgraph_path_weight(src, dst, subgraph_tensor):
    weight = 0     # impossible for a path to have 0 weight as each edge has weight > 1
                   # hence if the function returns 0 then  there is no path connection in the neigbourhood
    
    print(subgraph_tensor.n_id)
    print(subgraph_tensor.x)

    x = []
    for a in subgraph_tensor.n_id:
        x.append([a.item()])

    tensorX = torch.tensor(x, dtype=torch.float)
    
    subData = Data(x=tensorX, edge_index=subgraph_tensor.n_id[subgraph_tensor.edge_index], edge_attr=subgraph_tensor.edge_attr)
    print(subData)
    print(subData.x)
    print(subData.edge_index)
    print(subData.edge_attr)
    subgraph = data_to_networkx_graph(subData)
    print(subgraph.nodes)
    try:
        weight = nx.shortest_path_length(subgraph, source=src, target=dst, weight='weight') 
    except BaseException as e:
        weight = 0

    return weight
    
    

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

# gcn = GCN().to(device)
# optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
# criterion = nn.CrossEntropyLoss()    #????
# gcn = tra
    
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



vector = []
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
batch3 = next(iter(loader3))

batches = [batch1, batch2, batch3]

for batch in batches:
    print(batch)
    print(batch.n_id[batch.edge_index])    # output edge_index of subgraph with original node lables 
    print(batch.n_id)
    print(batch.edge_attr)
    print(batch.batch_size)
    print(batch.x)
    print(batch.y)
    edge_index = batch.n_id[batch.edge_index]

    src = int(random.randint(0, 99))
    dst = int(random.randint(0, 99))

    src2 = int(random.choice(edge_index[0]))
    dst2 = int(random.choice(edge_index[0]))

    print(f"Distance between {src} and {dst} is: {subgraph_path_weight(src, dst, batch)}")
    print(f"Distance between {src2} and {dst2} is: {subgraph_path_weight(src2, dst2, batch)}")

    subData = Data(x=batch.x, edge_index=edge_index, edge_attr=batch.edge_attr)
   

    

    #subgraph = to_networkx(subData, to_undirected=True)  # issue where duplicate nodes are drawn. Might be the case
                                                         # because .edge_index contains 2 edges for each node pair to 
                                                         # represent an undirected graph
    # clear plot

    # pos = nx.spring_layout(subgraph, seed=7)
    # nx.draw(subgraph, with_labels=True)
    # #labels = nx.get_edge_attributes(graph, 'weight')
    # nx.draw_networkx(subgraph, pos, )


    plt.show()

    inp = int(input("Done"))