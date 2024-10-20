""" graph_models.py
    Graph models for the IREM paper and DT_GNN

    Collaboratively developed for IREM: https://github.com/yilundu/irem_code_release
    
    Developed for IREM project

    Modified by Sean McLeish: https://github.com/mcleish7/DT-GNN

    Utilised by Pav Patra in custom SP_NN_model.py class
"""

import torch
from torch_geometric.nn import GINEConv, global_max_pool, GATv2Conv, GMMConv, GENConv, PDNConv, SAGEConv, EdgeConv, GATConv, knn_graph, MessagePassing
from torch import nn
import torch.nn.functional as F
import torch_geometric
import sys

import dt_1d_net as dt_models


class DT_recurrent(nn.Module):
    """
    Class for the DT GNN
    Inherits:
        torch.nn.Module (class)
    Attributes
    ----------
    connected : bool
        True if the dataset is for the connected componenets problem
    articulation: bool
        True if the dataset is for the articulation point problem
    random_noise: bool
        True if random noise initialisation is used during training
    edge_map: torch.nn.Linear
        Linear transformation of edge features
    conv1: torch.nn.Module
        The encoder, usually torch_geometric.nn.GCNConv but can be changed using self.exotic
    exotic: bool
        If true allows us to use one of MPNN or GAT as the encode for the encoder, Else we use torch_geometric.nn.GCNConv encoder
    recur: dt_1d_net.dt_net_recall_1d
        The Deep Thinking recurrent core
    conv2: torch_geometric.nn.GCNConv
        Currently not used, kept in for development purposes and to be able to use previously trained models
    conv3: torch_geometric.nn.GCNConv
        Decoder
    decode: torch.nn.Linear
        Transforms the node features returned into edge features
    m: torch.nn.Sigmoid
        Currently not used, kept in for development purposes
    Methods
    -------
    __init__(self, inp_dim, out_dim, random_noise=False, GAT_encoder=False, edge_MPNN_encoder=False, gcn_MPNN_encoder=False, connected=False, artdata=False)
        Constructor for the DT_recurrent class
    forward(self, inp, iters_to_do=10, iters_elapsed=0, interim_thought=None, **kwargs)
        Forward call of the network, used to apply the network to data
    apply_operator(self, inp)
        Applies an aggregation operator over the graph input, aggregates edge fretures for each nodes neighbourhood
    change_random_noise(self)
        Changes the random noise class variable to the opposite value, used to turn random noise off during validation testing
    """
    def __init__(self, inp_dim, out_dim, random_noise=False, GAT_encoder=False, edge_MPNN_encoder=False, gcn_MPNN_encoder=False, connected=False, artdata=False):
        """
        Constructor for the DT_recurrent class, for DT GNN

        Args:
            inp_dim (int): input dimension of data
            out_dim(int): output dimension of data
            random_noise (bool, optional): Set True if using random noise initialisation
            GAT_encoder (bool, optional): Set True to use a GAT encoder
            edge_MPNN_encoder (bool, optional): Set True to use a edge MPNN encoder
            gcn_MPNN_encoder (bool, optional): Set True to use a gcn MPNN encoder
            connected (bool, optional): Set True if solving the connected components problem
            artdata (bool, optional): Set True if solving the articulation point problem
        """
        super(DT_recurrent, self).__init__()
        self.connected = False
        if connected == True:
            self.connected = True
        self.articulation = False
        if artdata:
            self.articulation=True

        h = 128
        self.random_noise = random_noise
        self.edge_map = nn.Linear(1, h)

        self.exotic = True
        if GAT_encoder:
            self.conv1 = GAT(1,1, dt_call=True)
        elif edge_MPNN_encoder:
            self.conv1 = DynamicEdgeConv(1,1, dt_call=True)
        elif gcn_MPNN_encoder:
            self.conv1 = GCNConv(1,1, dt_call=True)
        else:
            self.exotic=False
            self.conv1 = torch_geometric.nn.GCNConv(1, 128)

        self.recur = dt_models.dt_net_recall_1d(width = h, in_channels = 128, graph=True)
        self.conv2 = torch_geometric.nn.GCNConv(128, 128)

        self.conv3 = torch_geometric.nn.GCNConv(128, 128)

        if self.articulation:
            self.decode = nn.Linear(128, 1)
        else:
            self.decode = nn.Linear((2*h), 1)
        
        self.m = nn.Sigmoid()

    def forward(self, inp, iters_to_do=10, iters_elapsed=0, interim_thought=None, **kwargs):
        """
        Foward call for DT GNN

        Args:
            inp (Data): the input graph in torch_geometric.data.Data format
            iters_to_do (int, optional): number of iterations to run for
            iters_elapsed (int, optional): number of iterations already completed
            interim_thought (torch.Tensor, optional): the tensor to be used in the skip connections
        Returns:
            all_outputs: the models output for the input data
            interim_thought: the tensor before the decoder is applied
        """
        all_outputs = []
        edge_attr = inp.edge_attr
        edge_index = inp.edge_index
        orginal_edge_index = inp.edge_index

        org_x = inp.x
        inp.x = self.apply_operator(inp) # applies aggregator over a nodes neighbourhood

        edge_embed = self.edge_map(edge_attr)
        if self.exotic:
            h = self.conv1(inp)
        else:
            h = self.conv1(inp.x, inp.edge_index)

        initial_h = h.T
        if self.random_noise and interim_thought == None:
            interim_thought = torch.randn((1,128,inp.x.shape[0]), device=f'cuda:{h.get_device()}') # random noise initialisation
        for i in range(0, iters_to_do):     

            _, interim_thought = self.recur(initial_h[None], interim_thought=interim_thought, iters_to_do=1) # Deep Thinking core

            h = torch.unsqueeze(torch.squeeze(interim_thought).T,0)

            h = F.relu(h) 
            h = self.conv3(h[0], inp.edge_index)
            if self.articulation: # articulation point identification is node based so we don't need to transform to edge features
                output = self.decode(h)
            else:
                hidden = h
                h1 = hidden[orginal_edge_index[0]]
                h2 = hidden[orginal_edge_index[1]]
                h = torch.cat([h1, h2], dim=-1)
                output = self.decode(h)

            all_outputs.append(output)

        inp.x = org_x # changes the node features back to their origional values as we overwrote them earlier
        return all_outputs, interim_thought

    def apply_operator(self, inp):
        """
        Applies an aggregator over a nodes neighbourhood and stores the output in the ndoes features

        Args:
            inp (Data): the input graph in torch_geometric.data.Data format
        Returns:
            torch.Tensor of size |V| of the aggregated edge features for each nodes neighbourhood
        """
        nx_g = torch_geometric.utils.to_networkx(inp, edge_attrs=["edge_attr"], to_undirected=True)
        operator = "min" # take the minimum value of all adjacent edges as the nodes features
        if self.connected == True: # for connected components use max aggregation
            operator = "max"
        store = []
        # Currently written versbosely for development purposes
        for node in nx_g.nodes():
            edges = nx_g.edges(node, data=True)
            # print(edges)
            operand = 0.0
            if operator == "min":
                operand = float("inf")
            for node1, node2, data in edges:
                if operator == "max":
                    if data["edge_attr"] > operand:
                        operand = data["edge_attr"]
                elif operator == "sum":
                    operand += data["edge_attr"]
                elif operator == "min":
                    # print(data["edge_attr"][0])
                    # print(operand)
                    if data["edge_attr"][0] < operand:
                        operand = data["edge_attr"][0]
                else:
                    print("operator not implemented")
                    sys.exit()
            store.append(operand)
        return torch.unsqueeze(torch.Tensor(store), dim=-1).cuda()

    def change_random_noise(self):
        """
        Changes the random noise class variable to the opposite value
        """
        self.random_noise = not self.random_noise




    
class GAT(torch.nn.Module):
    """
    Class for the GATv2 model
    Based on code from: https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/runtime/gcn.py
    Inherits:
        torch.nn.Module (class)
    Attributes
    ----------
    connected : bool
        True if the dataset is for the connected componenets problem
    dt_call: bool
        True if we are calling this class from the DT GNN class
    edge_map: torch.nn.Linear
        maps edge features into a fixed size space
    conv0: torch.nn.Linear
        linear transformation of node features
    conv1: torch.nn.GATv2Conv
        GATv2 PyTorch Geometric layer
    conv2: torch.nn.GATv2Conv
        GATv2 PyTorch Geometric layer
    decode: torch.nn.Linear
        helps transform node features to edge features
    m: torch.nn.Sigmoid
        sigmoid layer, not currently used
    Methods
    -------
    __init__(self, in_channels, out_channels, dt_call=False, connected=False)
        Constructor for the GAT class
    forward(self, data)
        Forward call of the network, used to apply the network to data
    """
    def __init__(self, in_channels, out_channels, dt_call=False, connected=False):
        """
        Constructor for GAT model

        Args:
            in_channels (int): number of channels in
            out_channels (int): number of channels out
            dt_call (bool, optional): True if being called from the DT model
            connected (bool, optional): True if dataset is connected components
        """
        super(GAT, self).__init__()
        self.connected = False
        if connected == True:
            self.connected = True

        self.dt_call=dt_call
        in_channels, out_channels, size = 128, 128, 128
        
        self.edge_map = nn.Linear(1, size)
        self.conv0 = nn.Linear(1, size)
        self.conv1 = GATv2Conv(in_channels, 8, heads=8, dropout=0.6, edge_dim=size)
        self.conv2 = GATv2Conv(8 * 8, out_channels, dropout=0.6, edge_dim=size)
        self.decode = nn.Linear((2*size), 1)
        self.m = nn.Sigmoid()

    def forward(self, data):
        """
        Foward call for GAT model

        Args:
            data (Data): the input graph in torch_geometric.data.Data format
        Returns:
            the models output for the input data
        """
        x, edge_index = data.x, data.edge_index
        edge_embed = self.edge_map(data.edge_attr)
        x = self.conv0(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_embed)
        x =  F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_embed)
        # x = F.log_softmax(x, dim=1)
        if self.dt_call:
            return x
        hidden = x

        h1 = hidden[edge_index[0]]
        h2 = hidden[edge_index[1]]

        h = torch.cat([h1, h2], dim=-1)
        output = self.decode(h)
        return output


    
class DynamicEdgeConv(EdgeConv):
    """
    Class for the edge MPNN model
    Based on code from: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    Inherits:
        EdgeConv (class)
    Attributes
    ----------
    connected : bool
        True if the dataset is for the connected componenets problem
    dt_call: bool
        True if we are calling this class from the DT GNN class
    k: int
        Number of neighbours to be used
    decode: torch.nn.Linear
        helps transform node features to edge features
    m: torch.nn.Sigmoid
        sigmoid layer, not currently used
    Methods
    -------
    __init__(self, in_channels, out_channels)
        Constructor for the edge MPNN class
    forward(self, data)
        Forward call of the network, used to apply the network to data
    """
    def __init__(self, in_channels, out_channels, k=6, dt_call=False, connected=False):
        """
        Constructor for edge MPNN model

        Args:
            in_channels (int): number of channels in, currently not used but kept to keep all method signitures the same
            out_channels (int): number of channels out, currently not used but kept to keep all method signitures the same
            dt_call (bool, optional): True if being called from the DT model
            connected (bool, optional): True if dataset is connected components
            k (int, optional): number of neighbours to use in K-NN
        """
        self.dt_call=dt_call
        self.connected = False
        if connected == True:
            self.connected = True
        in_channels = 1
        out_channels = 128
        super().__init__(in_channels, out_channels)
        self.k = k
        self.decode = nn.Linear((2*128), 1)
        self.m = nn.Sigmoid()

    def forward(self, x, batch=None):
        """
        Foward call for edge MPNN model

        Args:
            x (Data): the input graph in torch_geometric.data.Data format
            batch (Batch): used if using the pytorch geometric batching method
        Returns:
            the models output for the input data
        """
        edge_index = knn_graph(x.x, self.k, batch, loop=False, flow=self.flow) # k nearest neighbours
        hidden = super().forward(x.x, edge_index) # base class forward call
        if self.dt_call:
            return hidden
        edge_index = x.edge_index
        h1 = hidden[edge_index[0]]
        h2 = hidden[edge_index[1]]

        h = torch.cat([h1, h2], dim=-1)
        output = self.decode(h)
        return output

class GCNConv(MessagePassing):
    """
    Class for the gcn MPNN model
    Based on code from: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    Inherits:
        torch_geometric.nn.MessagePassing (class)
    Attributes
    ----------
    connected : bool
        True if the dataset is for the connected componenets problem
    dt_call: bool
        True if we are calling this class from the DT GNN class
    lin: torch.nn.Linear
        Linear transformation of the network
    bias: torch.nn.Parameter
        Bias of the network
    decode: torch.nn.Linear
        helps transform node features to edge features
    m: torch.nn.Sigmoid
        sigmoid layer, not currently used
    Methods
    -------
    __init__(self, in_channels, out_channels)
        Constructor for the gcn MPNN class
    reset_parameters(self)
        resets the linear layer and bias to small random weights and 0 respectively
    forward(self, data)
        Forward call of the network, used to apply the network to data
    message(self, x_i, x_j)
        Message call of the network, used to pass messages between nodes
    """
    def __init__(self, in_channels, out_channels, dt_call=False, connected=False):
        """
        Constructor for gcn MPNN model

        Args:
            in_channels (int): number of channels in, currently not used but kept to keep all method signitures the same
            out_channels (int): number of channels out, currently not used but kept to keep all method signitures the same
            dt_call (bool, optional): True if being called from the DT model
            connected (bool, optional): True if dataset is connected components
        """
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.connected = False
        if connected == True:
            self.connected = True

        self.dt_call=dt_call
        in_channels, out_channels = 1, 128
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.decode = nn.Linear(256, 1)
        self.reset_parameters()
        self.m = nn.Sigmoid()

    def reset_parameters(self):
        """
        Resets parameters of message passing part of the model
        """
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, inp):
        """
        Foward call for gcn MPNN model

        Args:
            inp (Data): the input graph in torch_geometric.data.Data format
        Returns:
            the models output for the input data
        """
        x = inp.x
        edge_index = inp.edge_index
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        if self.dt_call:
            return out + self.bias
        # Step 6: Apply a final bias vector.
        hidden = out + self.bias
        edge_index = inp.edge_index
        h1 = hidden[edge_index[0]]
        h2 = hidden[edge_index[1]]

        h = torch.cat([h1, h2], dim=-1)
        output = self.decode(h)
        return output

    def message(self, x_j, norm):
        """
        Message call for gcn mpnn model

        Args:
            x_j (torch.Tensor): has shape [E, in_channels]
            norm (torch.Tensor): normalises node features.
        Returns:
            the transformaton of the two inputs
        """
        return norm.view(-1, 1) * x_j