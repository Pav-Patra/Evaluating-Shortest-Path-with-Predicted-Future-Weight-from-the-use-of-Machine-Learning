import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

# add nodes
G.add_node(1, pos=(1,3))
G.add_node(2, pos=(2,3))
G.add_node(3, pos=(3,2))
G.add_node(4, pos=(1,-1))
G.add_node(5, pos=(2.1,-1))
G.add_node(6, pos=(-1,-1))
G.add_node(7, pos=(0,-2))
G.add_node(8, pos=(2,-2))

# add edges G
G.add_edge(1,2, weight=2)
G.add_edge(1,4, weight=1)
G.add_edge(1,5, weight=3)
G.add_edge(1,6, weight=1)
G.add_edge(2,3, weight=1)
G.add_edge(2,5, weight=1)
G.add_edge(3,8, weight=3)
G.add_edge(4,8, weight=2)
G.add_edge(5,8, weight=1)
G.add_edge(6,7, weight=1)
G.add_edge(7,8, weight=1)

pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

plt.show()