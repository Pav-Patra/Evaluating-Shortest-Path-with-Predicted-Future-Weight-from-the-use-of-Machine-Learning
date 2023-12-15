import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a Concolution Neural Network Class which includes my own expansions 
# on the PyTorch Neural Neural Network module.
class GraphConvolutionLayer(nn.Module):
    # initialise Convolution Layers
    def __init__(self, input_features, output_features):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(in_features=input_features, out_features=output_features)

    def forward(self, adjacency_matrix, node_features, cumulative_weights):
        laplacian_matrix = make_laplacian(adjacency_matrix)
        graph_convolution = torch.matmul(laplacian_matrix, node_features)
        graph_convolution = F.relu(self.linear(graph_convolution))

        # Combining the graph convolution with the cumulative weighted paths of the n-ary trees
        combine_graph = torch.cat([graph_convolution, cumulative_weights], dim=1)
        return combine_graph


# helper funciton to compute Laplacian matric
def make_laplacian(adjacency_matrix):
    laplacian = adjacency_matrix
    return laplacian