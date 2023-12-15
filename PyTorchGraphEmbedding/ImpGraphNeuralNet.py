import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import KarateClub
import matplotlib.pyplot as plt

dataset = KarateClub()

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        result = self.conv1(x, edge_index)
        result = result.tanh()
        result = self.conv2(result, edge_index)
        result = result.tanh()
        result = self.conv3(result, edge_index)
        result = result.tanh()   # Final GNN embeddign space

        # Apply a final (linear) classifier
        out = self.classifier(result)

        return out, result
    
model = GCN()
print(model)


# Embedding the karate club metwork
_, result = model(dataset.x, dataset.edge_index)
print(f"Embedding shape: {list(result.shape)}")
print(result)

numpyResult = result.detach().numpy()

plt.scatter(numpyResult[:, 0], numpyResult[:, 1])
plt.xticks([])
plt.yticks([])
plt.show()
