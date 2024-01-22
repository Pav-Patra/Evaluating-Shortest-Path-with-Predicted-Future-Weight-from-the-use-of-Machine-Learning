import torch
import torch.nn as nn
import torch.nn.functional as F

class DijkstraNetwork(nn.Module):
    def __init__(self, num_nodes, feature_dim, hidden_dim):
        super(DijkstraNetwork, self).__init__()
        self.embedding = nn.Embedding(num_nodes, feature_dim)
        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, adjacency_matrix, source_node):
        node_features = self.embedding(torch.arange(adjacency_matrix.size(0)))
        source_features = node_features[source_node].unsqueeze(0).expand(adjacency_matrix.size(0), -1)

        # Concatenate node features and source features
        combined_features = torch.cat([node_features, source_features], dim=1)

        # First linear layer
        hidden = F.relu(self.linear1(combined_features))

        # Second linear layer for regression (predicting path weight)
        output = self.linear2(hidden)

        return output

# Example usage
num_nodes = 10
feature_dim = 16
hidden_dim = 32

# Assuming an adjacency matrix (replace with your own graph representation)
adjacency_matrix = torch.rand((num_nodes, num_nodes))
adjacency_matrix = (adjacency_matrix + adjacency_matrix.t()) / 2  # Ensure symmetry for an undirected graph

# Source node for prediction
source_node = 0

# Create the model
model = DijkstraNetwork(num_nodes, feature_dim, hidden_dim)

# Forward pass
predicted_weight = model(adjacency_matrix, source_node)
print("Predicted Shortest Path Weight:", predicted_weight.item())