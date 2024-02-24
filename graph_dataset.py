""" graph_dataset.py
    Graph datasets for the IREM paper and DT_GNN

    Collaboratively developed for IREM: https://github.com/yilundu/irem_code_release
    
    Developed for IREM project

    Edited by Sean McLeish for CS344 Discrete Maths Project 2023
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


class NoisyWrapper:

    def __init__(self, dataset, timesteps):

        self.dataset = dataset
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        self.inp_dim = dataset.inp_dim
        self.out_dim = dataset.out_dim

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        # self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod))
        # self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1. - alphas_cumprod))

        alphas_cumprod = np.linspace(1, 0, timesteps)
        self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1. - alphas_cumprod))
        self.extract = extract

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, *args, **kwargs):
        data = self.dataset.__getitem__(*args, **kwargs)
        y = data['y']

        t = torch.randint(1, self.timesteps, (1,)).long()
        t_next = t - 1
        noise = torch.randn_like(y)

        sample = (
            self.extract(self.sqrt_alphas_cumprod, t, y.shape) * y +
            self.extract(self.sqrt_one_minus_alphas_cumprod, t, y.shape) * noise
        )

        sample_next = (
            self.extract(self.sqrt_alphas_cumprod, t_next, y.shape) * y +
            self.extract(self.sqrt_one_minus_alphas_cumprod, t_next, y.shape) * noise
        )

        data['y_prev'] = sample.float()
        data['y'] = sample_next.float()

        return data


class Identity(data.Dataset):
    """
    Constructs a dataset at run time for the identity/edge copy problem
    Inherits:
        torch.utils.data.Dataset (class)
    Attributes
    ----------
    h : int
        equal to rank, the datasets used here are taken from the IREM paper so some variables are not used in this case
    w : int
        equal to rank, the datasets used here are taken from the IREM paper so some variables are not used in this case
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
        Constructor for the edge copy problem
    __getitem__(self, index)
        Returns a randomly generated graph
    __len(self)__
        Returns a large constant as the size of the dataset. As the data is generated at runtime this is a large constant.
    """
    def __init__(self, split, rank, vary=False):
        """
        Constructor for the edge copy problem

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

        R = np.random.uniform(-1, 1, (rank, rank))
        R_corrupt = R

        repeat = 128 // self.w + 1

        R_tile = np.tile(R, (1, repeat))

        node_features = torch.Tensor(np.zeros((rank, 1)))
        edge_index = torch.LongTensor(np.array(np.mgrid[:rank, :rank]).reshape((2, -1)))
        edge_features_context = torch.Tensor(np.tile(R_corrupt.reshape((-1, 1)), (1, 1)))
        edge_features_label = torch.Tensor(R.reshape((-1, 1)))
        noise = torch.Tensor(edge_features_label.reshape((-1, 1)))

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features_context, y=edge_features_label, noise=noise)

        return data

    def __len__(self):
        """Return the total number of samples in the dataset. Returns a large constant as the dataset is randomly generated at runtime"""
        return int(1e6)

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

class MaxFlow(data.Dataset):
    """
    Constructs a dataset at run time for the max flow problem
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
    inp_dim : int
        the dimension of the input vector, used in the constructor for models
    out_dim : int
        the dimension of the output vector, used in the constructor for models

    Methods
    -------
    __init__(self, split, rank, vary=False)
        Constructor for the max flow problem
    __getitem__(self, index)
        Returns a randomly generated graph
    __len(self)__
        Returns a large constant as the size of the dataset. As the data is generated at runtime this is a large constant.
    """
    def __init__(self, split, rank, vary=False):
        """
        Constructor for the max flow problem

        Args:
            split (bool): Split the dataset into Train/Val if true
            rank (int): number of nodes in the graph
            vary (bool, optional): If set to True, the size of the graphs in the dataset will vary between 2 and rank
        """
        self.h = rank
        self.w = rank
        self.vary = vary
        self.rank = rank

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

        # try testing with edge weights between 1 and 10
        graph = np.random.uniform(low=0, high=1, size=[rank, rank]) #weights

        np.fill_diagonal(graph, 0.0) #remove self loops

        G = networkx.from_numpy_array(graph)
        y = np.zeros((rank,rank))
        start = 0
        for i in range(0, rank-1):
            for j in range(i+1,rank):
                flow = networkx.algorithms.flow.maxflow.minimum_cut_value(G, i, j,capacity='weight') # using networkx algorithm to find maxflow between each pair of nodes
                y[i,j] = flow 
                y[j,i] = flow # using symmetry of max flow

        node_features = torch.Tensor(np.zeros((rank, 1)))
        edge_index = torch.LongTensor(np.array(np.mgrid[:rank, :rank]).reshape((2, -1)))

        edge_features_context = torch.Tensor(np.tile(graph.reshape((-1, 1)), (1, 1)))
        edge_features_label = torch.Tensor(y.reshape((-1, 1)))

        noise = torch.Tensor(edge_features_label.reshape((-1, 1)))
        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features_context, y=edge_features_label, noise=noise)
        return data

    def __len__(self):
        """Return the total number of samples in the dataset. Returns a large constant as the dataset is randomly generated at runtime"""
        return int(1e6)

class ConnectedComponents(data.Dataset):
    """
    Constructs a dataset at run time for the connected compoenents problem
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
        Constructor for the connected componenets problem
    __getitem__(self, index)
        Returns a randomly generated graph
    __len(self)__
        Returns a large constant as the size of the dataset. As the data is generated at runtime this is a large constant.
    """

    def __init__(self, split, rank, vary=False):
        """
        Constructor for the connected components problem

        Args:
            split (bool): Split the dataset into Train/Val if true
            rank (int): number of nodes in the graph
            vary (bool, optional): If set to True, the size of the graphs in the dataset will vary between 2 and rank
        """
        self.h = rank
        self.w = rank
        self.rank = rank
        self.vary = vary

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

        p = min((5/(2*rank)),0.3) # probability of creating an edge

        graph = networkx.fast_gnp_random_graph(rank, p) # uses networkx to generate a random graph

        connections = networkx.to_numpy_array(graph) # adjacency matrix of the random graph
        y = np.zeros((rank,rank))
        for i in range(0, rank-1):
            for j in range(i+1,rank):
                flow = networkx.has_path(graph,i,j)
                y[i,j] = flow
                y[j,i] = flow
        label = y

        node_features = torch.Tensor(np.zeros((rank, 1)))
        edge_index = torch.LongTensor(np.array(np.mgrid[:rank, :rank]).reshape((2, -1)))

        edge_features_context = torch.Tensor(np.tile(connections.reshape((-1, 1)), (1, 1)))
        edge_features_label = torch.Tensor(label.reshape((-1, 1)))
        noise = torch.Tensor(label.reshape((-1, 1)))

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features_context, y=edge_features_label, noise=noise)

        return data

    def __len__(self):
        """Return the total number of samples in the dataset. Returns a large constant as the dataset is randomly generated at runtime"""
        return int(1e6)
        
class ArticulationPoints(data.Dataset):
    """
    Constructs a dataset at run time for the articualtion point problem
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
    inp_dim : int
        the dimension of the input vector, used in the constructor for models
    out_dim : int
        the dimension of the output vector, used in the constructor for models

    Methods
    -------
    __init__(self, split, rank, vary=False)
        Constructor for the articulation point problem
    __getitem__(self, index)
        Returns a randomly generated graph
    __len(self)__
        Returns a large constant as the size of the dataset. As the data is generated at runtime this is a large constant.
    """
    def __init__(self, split, rank, vary=False):
        """
        Constructor for the articulation point problem

        Args:
            split (bool): Split the dataset into Train/Val if true
            rank (int): number of nodes in the graph
            vary (bool, optional): If set to True, the size of the graphs in the dataset will vary between 2 and rank
        """
        self.h = rank
        self.w = rank
        self.vary = vary
        self.rank = rank

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
        p = min((5/(2*rank)),0.3) # probability of two nodes being connected
        graph = networkx.fast_gnp_random_graph(rank, p) #using networkx to generate graph
        y = list(networkx.articulation_points(graph)) # list of indecies of nodes which are articualtion points
        a = np.zeros(rank, dtype="int64")
        np.put(a, y, np.ones(len(y), dtype="int64").tolist()) # numpy array of size |V| with ones in indecies which are articulation points
        y=a
        graph = networkx.to_numpy_array(graph) # adjacency matrix

        node_features = torch.Tensor(np.zeros((rank, 1)))
        edge_index = torch.LongTensor(np.array(np.mgrid[:rank, :rank]).reshape((2, -1)))

        edge_features_context = torch.Tensor(np.tile(graph.reshape((-1, 1)), (1, 1)))
        edge_features_label = torch.Tensor(y.reshape((-1, 1)))

        noise = torch.Tensor(edge_features_label.reshape((-1, 1)))
        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features_context, y=edge_features_label, noise=noise)
        
        return data

    def __len__(self):
        """Return the total number of samples in the dataset. Returns a large constant as the dataset is randomly generated at runtime"""
        return int(1e6)

class MaxFlowNumpy(data.Dataset):
    """
    Calls the SAVED dataset for the maxflow problem
    Have to save the data in a folder named data
    Please contact the authors on GitHub for access to the link for the datasets

    Inherits:
        torch.utils.data.Dataset (class)
    Attributes
    ----------
    h : int
        equal to rank
    w : int
        equal to rank
    graphs : numpy array
        the graphs in the dataset
    targets : numpy array
        the targets in the same order as the graphs in the dataset
    rank : int
        the number of nodes in the graphs in this dataset
    inp_dim : int
        the dimension of the input vector, used in the constructor for models
    out_dim : int
        the dimension of the output vector, used in the constructor for models

    Methods
    -------
    __init__(self, split, rank, vary=False)
        Constructor for the maxflow problem
    __getitem__(self, index)
        Returns a graph from the dataset
    __len(self)__
        Return the total number of samples in the dataset. Is fixed as this is a predefined dataset due to the time taken to create samples.
    """
    def __init__(self, split, rank):
        """
        Constructor for the articulation point problem

        Args:
            split (bool): Split the dataset into Train/Val if true
            rank (int): number of nodes in the graph
        """
        self.rank = rank
        self.graphs, self.targets = self.load_from_np(rank)

        self.h = rank
        self.w = rank
        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w

    def load_from_np(self, rank):
        """
        Returns numpy arrays of 10000 graphs of size rank and their targets 
        Args: 
            rank (int): size of the graphs
        Returns:
            two numpy arrays of the graphs adjacency matracies and targets
        """
        data = np.load('data/'+str(rank)+'_maxflow.npz')
        for item in data.keys():
            graphs, targets = data[item][0], data[item][1]
        return graphs, targets

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Args:
            index (int): an integer for data indexing, not used in this framework as data is created at runtime

        Returns:
            Pytorch Geometric Data: representation of an undirected weighted graph
        """
        rank = self.rank
        connections = self.graphs[index]
        label = self.targets[index]

        node_features = torch.Tensor(np.zeros((rank, 1)))
        edge_index = torch.LongTensor(np.array(np.mgrid[:rank, :rank]).reshape((2, -1)))

        edge_features_context = torch.Tensor(np.tile(connections.reshape((-1, 1)), (1, 1)))
        edge_features_label = torch.Tensor(label.reshape((-1, 1)))
        noise = torch.Tensor(label.reshape((-1, 1)))
        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features_context, y=edge_features_label, noise=noise)

        return data    
    
    def __len__(self):
        """Return the total number of samples in the dataset. Is fixed as this is a predefined dataset due to the time taken to create samples."""
        return 10000