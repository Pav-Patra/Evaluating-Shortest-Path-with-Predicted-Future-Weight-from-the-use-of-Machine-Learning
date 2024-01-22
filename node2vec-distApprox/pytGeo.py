import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, is_undirected, to_undirected
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
#import GraphSPML.Incremental_Shortest_Path as ISP 

# Define a weighted undirected graph
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
edge_weight = torch.tensor([0.5, 0.5, 1.2, 1.2], dtype=torch.float)
x = torch.tensor([[2, 1], [5, 6], [3, 7]], dtype=torch.float)

# Create a PyTorch Geometric Data object with edge weights
data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

print(data)

print(f"Edge weightd: {data.edge_attr}")


graph = to_networkx(data, to_undirected=True)

pos = nx.spring_layout(graph, seed=7)
nx.draw(graph, with_labels=True)
labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
plt.show()

# get example dataset
dataset = Planetoid(root='tmp/Cora', name='Cora')

# a very basic Graph Convolutional Network
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

#train the model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# evaluate the model
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f"Accuracy: {acc:.4f}")





