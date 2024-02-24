from graph_dataset import ShortestPath
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

""""
My own file to test the nature of the ShortestPath dataset

I will identify how graphs are formed, their edge nature, how edge weights are expressed

"""


dataset = ShortestPath('train', 16, vary=True)

randomGraph = dataset.__getitem__(1)

print(dataset.__len__())
print(randomGraph)

print("x var:")
print(randomGraph.x)
print()
print("edge_index:")
print(randomGraph.edge_index)
print()
print("edge_attr:")
print(randomGraph.edge_attr)
print()
print("y var:")
print(randomGraph.y)
print()
print("noise:")
print(randomGraph.noise)

draw_graph = to_networkx(randomGraph, to_undirected=True)

pos = nx.spring_layout(draw_graph, seed=7)
nx.draw(draw_graph, with_labels=True)
#labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx(draw_graph, pos)
plt.show()


# node_features - all set to 0 for each of the rank number of nodes in the graph 
# edge_index - normal edge_index connecting nodes from number 0 to rand(2 to rank)


# dataset has taken shortcuts - for the number of nodes (n), there are n^2 edges