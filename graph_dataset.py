""" graph_dataset.py
    Graph datasets for the IREM paper and DT_GNN

    Collaboratively developed for IREM: https://github.com/yilundu/irem_code_release
    
    Developed for IREM project

    Modified by Sean McLeish: https://github.com/mcleish7/DT-GNN

    Utilised by Pav Patra in custom SP_NN_model.py class
"""
import torch.utils.data as data
from torch_geometric.data import Data
import torch
import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path, maximum_flow
from scipy.sparse import csr_matrix
import random
import sys
import networkx

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)




class ShortestPath(data.Dataset):
    """
    Constructs a dataset at run time for the shortest path problem
    Inherits:
        torch.utils.data.Dataset (class)
    Attributes
    ----------
    h : int
        equal to rank
    w : int
        equal to rank
    vary : bool
        if set to True, the size of the graphs in the dataset will vary between 2 and rank
    rank : int
        the number of nodes in the graphs in this dataset
    split : bool
        splits the dataset into Train/Val if true (currently not used as dataset is created at run time)
    inp_dim : int
        the dimension of the input vector, used in the constructor for models
    out_dim : int
        the dimension of the output vector, used in the constructor for models

    Methods
    -------
    __init__(self, split, rank, vary=False)
        Constructor for the shortest path problem
    __getitem__(self, index)
        Returns a randomly generated graph
    __len(self)__
        Returns a large constant as the size of the dataset. As the data is generated at runtime this is a large constant.
    """

    def __init__(self, split, rank, vary=False):
        """
        Constructor for the shortest path problem

        Args:
            split (bool): Split the dataset into Train/Val if true
            rank (int): number of nodes in the graph
            vary (bool, optional): If set to True, the size of the graphs in the dataset will vary between 2 and rank
        """
        self.h = rank
        self.w = rank
        self.vary = vary
        self.rank = rank

        self.split = split
        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Args:
            index (int): an integer for data indexing, not used in this framework as data is created at runtime

        Returns:
            Pytorch Geometric Data: representation of an undirected weighted graph
        """
        if self.vary:
            rank = random.randint(2, self.rank)
        else:
            rank = self.rank

        graph = np.random.uniform(0, 1, size=[rank, rank]) # weights - NEEDS FIXING
        # print("Initial graph weights")
        # print(graph)

        graph = np.random.uniform(low=0, high=10, size=[rank, rank]) #weights (test)
        graph = np.round(graph) # test

        graph = graph + graph.transpose()
        np.fill_diagonal(graph, 0)

        """
            shortest_path takes in a graph in matrix form. In other words:

            if graph = [
                [0, 1, 2, 0],
                [0, 0, 0, 1],
                [2, 0, 0, 3],
                [0, 0, 0, 0]
            ]
            This graph contains 5 edges with weights:
            (0, 1)  =  1
            (0, 2)  =  2
            (1, 3)  =  1
            (2, 0)  =  2
            (2, 3)  =  3

        """


        graph_dist, graph_predecessors = shortest_path(csgraph=csr_matrix(graph), unweighted=False, directed=False, return_predecessors=True, method="FW")
        avg = np.mean(np.mean(graph_dist))

        node_features = torch.Tensor(np.zeros((rank, 1)))   # sets node embedding for each node to 0
        edge_index = torch.LongTensor(np.array(np.mgrid[:rank, :rank]).reshape((2, -1)))    # stores each formed edge in the graph as 2 separate arrays where [out_edge] [in_edge] 
                                                                                            # length of both arrays == total number of edges

        edge_features_context = torch.Tensor(np.tile(graph.reshape((-1, 1)), (1, 1)))   # stores the original weight of each edge in the graph matrix as a flattened 1D array
        edge_features_label = torch.Tensor(graph_dist.reshape((-1, 1)))     # flattens matrix of shortest path weights between each node into 1D array

        noise = torch.Tensor(edge_features_label.reshape((-1, 1)))

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features_context, y=edge_features_label, noise=noise)

        return data

    def __len__(self):
        """Return the total number of samples in the dataset. Returns a large constant as the dataset is randomly generated at runtime"""
        return int(1e6)

