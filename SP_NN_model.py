import torch 
import os
import torch.nn.functional as F
from graph_dataset import ShortestPath
from graph_models import DT_recurrent
import random
import argparse
from argparse import Namespace
import os.path as osp
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from torch.optim import Adam
from scipy.sparse.csgraph import shortest_path, dijkstra
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_networkx
import Incremental_Shortest_Path as ISP


"""
    This file is created by Pav Patra

    Callable external class which allows the model to be called on test graph data.
    Allows the approximation of the shortest path between 2 nodes
"""

class SP_NN:
    # constructor
    def __init__(self):

        self.FLAGS = Namespace(train=False, cuda=True, vary=False, no_replay_buffer=False, dataset='shortestpath', logdir='outputs/shortestpath', exp='DT_sp', resume_iter=10000, batch_size=1, num_epoch=10000, lr=0.0001, 
                               log_interval=10, save_interval=1000, alpha=0.3, prog=True, json_name='tracker', plot=True, plot_name='65_10_test', plot_folder='plots/testing_plots', transfer_learn=False, transfer_learn_model=None, 
                               random_noise=False, data_workers=0, filter_dim=64, rank=15, num_steps=40, step_lr=100.0, latent_dim=64, decoder=False, gen=False, gen_rank=-5, recurrent=False, dt=True, gat=False, edge_mpnn=False, 
                               gcn_mpnn=False, ponder=False, no_truncate_grad=False, iterative_decoder=False, lr_throttle=False, nodes=1, gpus=1, node_rank=0, capacity=50000, infinite=True, edge_mpnn_encoder=False, gcn_mpnn_encoder=False, 
                               gat_encoder=False, replay_buffer=True)
        self.model = None
        self.device = torch.device('cuda')

        self.logdir = osp.join(self.FLAGS.logdir, self.FLAGS.exp)

        dataset = ShortestPath('train', self.FLAGS.rank, vary=self.FLAGS.vary)

        self.model, self.optimizer = self.init_model(self.FLAGS, dataset, self.device)

        model_path = osp.join(self.logdir, "model_best.pth".format(self.FLAGS.resume_iter))
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model created")

        self.model.eval()

        self.gdataset1 = None
        self.gdataset2 = None


    """
        initialise the first instance of the formed ISP graph
    """
    def declareFirstDataset(self, g):
        # create the data loader objects
        # to test just one graph on the model, set batch_size=1 and testing loop must end when counter > 0

        first_dataset = [self.create_isp_dataset(g,0)]

        # instead of directly finding the shortest path using the model, use an
        # initial graph and identify whether there has been a change in the new graph
        self.gdataset1 = DataLoader(first_dataset, num_workers=self.FLAGS.data_workers, batch_size=1, pin_memory=False, drop_last=True)

        print("First Dataset Declared")


    """
        Main function that takes the graph and runs the model against this graph
        gdataset1 is the version of the graph previously computed
        gdataset2 is the potentially updated graph
    """    
    def run(self, g, src, dst):
        # create the data loader objects
        # to test just one graph on the model, set batch_size=1 and testing loop must end when counter > 0

        live_dataset = [self.create_isp_dataset(g,src)]

        # instead of directly finding the shortest path using the model, use an
        # initial graph and identify whether there has been a change in the new graph
        self.gdataset2 = DataLoader(live_dataset, num_workers=self.FLAGS.data_workers, batch_size=1, pin_memory=False, drop_last=True)

        test_array, transpose_mean = self.start_tests(self.gdataset1, self.model, self.FLAGS, step=self.FLAGS.resume_iter, test_call=True)     # change test_call to false for less rigorous testing
        test_array2, transpose_mean2 = self.start_tests(self.gdataset2, self.model, self.FLAGS, step=self.FLAGS.resume_iter, test_call=True)

        # take the transpose matrices produced from bpth datasets and find the difference
        transpose_difference = transpose_mean - transpose_mean2
        # this essentially returns numerical values for each index in the adjacency matrix of the graph g
        # any non-zero values indicate a change in shortest path length

        change = self.findChange(transpose_difference, src, dst)
        # remove small values < 1e-6
        change = round(np.abs(change), 6)

        
        return change


    def test(self):
        print(self.FLAGS)



    def init_model(self, FLAGS, dataset, device):
        # method to initialise the shortest path model
        connecteddata = False
        if FLAGS.dataset == "connected":
            connecteddata = True
        articulationdata = False
        if FLAGS.dataset == "articulation":
            articulationdata = True
        
        # declare model class from graph_models 
        model = DT_recurrent(dataset.inp_dim, dataset.out_dim, random_noise=FLAGS.random_noise, connected=connecteddata, artdata = articulationdata)

        base_params = [p for n, p in model.named_parameters()]
        recur_params = []
        iters = 1
        all_params = [{"params": base_params}]

        model.to(device)

        optimizer = Adam(all_params, lr=FLAGS.lr, weight_decay=2e-6, eps=10**-6) # Adam optimiser used for all models
        return model, optimizer



    def gen_answer(self, inp, FLAGS, model, pred, scratch, num_steps):
        # Method to call the forward call of the model

        preds = []
        im_grads = []
        energies = []
        im_sups = []

        preds = []
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
        state = None
        preds, state = model.forward(inp, iters_to_do = num_steps, iterim_though = None)

        return pred, preds, im_grads, energies, scratch



    def start_tests(self, test_dataloader, model, FLAGS, step=0, gen=False, test_call=False, train_call=False, rig_call=False, alpha_rig_call=False):
        """
            Method to test a model
        """
        changed_rn = False # records if we have had to change random noise initialisation
        if FLAGS.dt and not(test_call or rig_call):
            if FLAGS.random_noise:
                model.change_random_noise()
                changed_rn = True

        global best_test_error, best_gen_test_error
        lim = 10 # number of iterations used in validation testing
        if test_call or rig_call or alpha_rig_call:
            lim = 10 # more rigourous testing if we are testing models  (CHANGED THIS FROM 100 to 10 as selected for shortest path TESTING)
        if FLAGS.cuda:
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        replay_buffer = None
        dist_list = []
        energy_list = []
        min_dist_energy_list = []

        model.eval()
        counter = 0

        with torch.no_grad():
            for data in test_dataloader:
                
                data = data.to(dev)
                im = data['y']
                
                # data['y'] holds the true shortest paths in the graph using scipy.sparse.csgraph.shortest_path
                # research how scipy.sparse.csgraph.shortest_path works


                pred = (torch.rand_like(data.edge_attr) - 0.5) * 2

                scratch = torch.zeros_like(pred)   # not used in gen_answer/forward pass???

                # pred variable is not used at all in gen_answer and model's forward function
                # the model's output is stored in preds
                # test output by passing one graph from the dataset into the dataloader

                pred, preds, im_grad, energies, scratch = self.gen_answer(data, FLAGS, model, pred, scratch, FLAGS.num_steps) # testing so no need for dependency graph
                #print("preds after forward pass test:")
                #print(len(preds))   # preds always contaains 40 tensors
                #print(len(preds[1]))  # the size of each tensor varies for different graph sizes.
                                    # it is always equal to the number of edges

                preds = torch.stack(preds, dim=0)   
                
                s_preds = preds.cpu().numpy()[0]
               

                energies = torch.stack(energies, dim=0).mean(dim=1).mean(dim=-1) # used for the EBM

                dist = (preds - im[None, :])
            

                dist = torch.pow(dist, 2).mean(dim=-1)
            
                # highest accuracy is viewed in the first index of dist
                dist_transpose = torch.transpose(dist, 0, 1)


                dist = dist.mean(dim=-1)

                dist_transpose_mean = dist_transpose.mean(dim=-1)
                ##print(dist_transpose_mean)
                

                min_idx = energies.argmin(dim=0)

                dist_min = dist[min_idx]
                min_dist_energy_list.append(dist_min)

                dist_list.append(dist.detach())
                energy_list.append(energies.detach())

                counter = counter + 1
                
                if counter > lim: # as the datasets are generated at run time we have to stop the testing manually
                    break
        dist = torch.stack(dist_list, dim=0).mean(dim=0)
       
        energies = torch.stack(energy_list, dim=0).mean(dim=0)
        min_dist_energy = torch.stack(min_dist_energy_list, dim=0).mean()
        if FLAGS.dt:
            min_dist_energy = dist[-1] # take the last output of the model as the final answer for deep thinking models

        print("Testing..................")
        print("last step error: ", dist)
        print("energy values: ", energies)
        print('test at step %d done!' % step)
        
        model.train()

        if changed_rn: # changed random noise back if it was chnaged at the beginning of the method
            model.change_random_noise()
        if test_call:
            return dist, dist_transpose_mean
        
    """
        This function takes in a ISP.Graph as input
    """
    def create_isp_dataset(self, g, intNode):
        
        # get total number of added nodes to define the rank
        rank = len(g.addedNodes)
        

        # define the graph's empty adjacency matrix (0 means no connection)
        graph = np.zeros((rank,rank))
        graph_dist_matrix = np.zeros((rank,rank))


        edgeArray = np.array(np.mgrid[:rank, :rank]).reshape((2, -1))

        # fill adjacency matrix with edge weights
        # glitch for nodes with 0 edges
        # use try catch block to analyse issue and perform an overcome to this
        for u in range(rank):
            for v, weight in g.graph[u]:
                graph[u][v] = weight

            # also get =shortest path distance between u and v to fill graph_dist_matrix
            for v in range(rank):            
                graph_dist_matrix[u][v] = g.findShortestPath(u, u, v)

                # construct edge_attr
                #weightArray.append([weight])


        edgeWeight = torch.Tensor(np.tile(graph.reshape((-1, 1)), (1, 1)))     # stores the original weight of each edge in the graph matrix as a flattened 1D array


        print("Done")
        # use shortest path results for .y variable
    
        # model requires some change in y variable to produce any change in output predictions
        
        # for the given target node, compute dijkstra's to all other destinations
        #for i in nodes:
        sp_val = dijkstra(csgraph=csr_matrix(graph), unweighted=False, directed=False, indices=intNode, return_predecessors=False, min_only=True)
             #sp_vals.append(sp_val)

       

        # replace sp_val in graph_dist_matrix

        # update distnce matrix with new results
        for i in range(rank):
                graph_dist_matrix[intNode][i] = sp_val[i]
       

        node_features = torch.Tensor(np.zeros((rank, 1))) # sets node embedding for each node to 0


        edge_index = torch.LongTensor(edgeArray)    # stores each formed edge in the graph as 2 separate arrays where [out_edge] [in_edge] 
                                                # length of both arrays == total number of edges
        
        edge_features_context = edgeWeight   # stores the original weight of each edge in the graph matrix as a flattened 1D array

        edge_features_label = torch.Tensor(graph_dist_matrix.reshape((-1, 1)))

        noise = torch.Tensor(edge_features_label.reshape((-1, 1)))

        
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features_context, y=edge_features_label, noise=noise)
 

        return data


    # this function takes in a transpose_mean array calculated from 2 predictions, a src node and dst node
    # the function determines whether there has been a change in path legth between the 2 nodes from both graphs
    # this helps notify the ISP algorithm whether the entire graph requires updating
    def findChange(self, transpose_mean, src, dst):
        rank = transpose_mean.size(dim=0)
        rank = int(np.sqrt(rank))
     
        np_mean = transpose_mean

        np_mean = torch.reshape(np_mean, (rank, rank))



        change = np_mean[src, dst].item()

        change2 = np_mean[dst, src].item()

        if(round(np.abs(change2), 6) >= 1e-6):
            change = change2

        return change








def data_to_networkx_graph(data):
    # manually convert tensor graph data to networkx weighted undirected graph

    plt.clf()
    plt.cla()
    g = nx.Graph()
    addedNodes = []
    addedEdges = []

    edge_index = data.edge_index 
    for i in range(len(edge_index[0])):
        x = int(edge_index[0][i].item())
        y = int(edge_index[1][i].item())

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
    
    nx.draw(g, pos, with_labels=True)
    labels = nx.get_edge_attributes(g,'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

    inp = int(input("Done"))

    return g



def matrix_to_networkx_graph(data):
    """
        Takes in a Data object which contains a graph holding weights as a noEdgesXnoEdges matrix
        Constructs the newtworkx graph using this information
        Note matrix weights of 0 means no edge connection, hence all edge weights must be greater than 0

        must assume edge_attr is flattened by a n by b dimensional matrix
    """

    g = nx.Graph()
    addedNodes = []
    addedEdges = []

    # extract edge connections
    edge_index = data.edge_index
    # extract edge weights 
    edge_attr = data.edge_attr

    # convert to numpy arrays
    edge_index = edge_index.numpy()
    edge_attr = edge_attr.numpy()

    rank = int(np.sqrt(len(edge_attr)))

    matrix_edge_attr = edge_attr.reshape((rank, rank))

    for i in range(len(edge_index[0])):
        x = int(edge_index[0][i])
        y = int(edge_index[1][i])
        

        if x not in addedNodes:
            g.add_node(x, pos=(0,0))
            addedNodes.append(x)

        if y not in addedNodes:
            g.add_node(y, pos=(0,0))
            addedNodes.append(y)

        if ((x,y) not in addedEdges or (y,x) not in addedEdges) and x != y:
            g.add_edge(x, y, weight=matrix_edge_attr[x][y])
            addedEdges.append((x, y))

    pos = nx.spring_layout(g, seed=7)     # positions for all nodes - seed for reproducibility
    
    nx.draw(g, pos, with_labels=True)
    labels = nx.get_edge_attributes(g,'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

    return g


    


def subgraph_path_weight(src, dst, subgraph_tensor):
    weight = 0     # impossible for a path to have 0 weight as each edge has weight > 1
                   # hence if the function returns 0 then  there is no path connection in the neigbourhood
    

    x = []
    for a in subgraph_tensor.n_id:
        x.append([a.item()])

    tensorX = torch.tensor(x, dtype=torch.float)
    
    subData = Data(x=tensorX, edge_index=subgraph_tensor.n_id[subgraph_tensor.edge_index], edge_attr=subgraph_tensor.edge_attr)
   
    subgraph = data_to_networkx_graph(subData)
   
    try:
        weight = nx.shortest_path_length(subgraph, source=src, target=dst, weight='weight') 
    except BaseException as e:
        weight = 0

    return weight






if __name__ == "__main__":
   
    g = ISP.Graph()
    rank = 10
    model = SP_NN()
    g.generateGraph(rank)
    g.incrementalShortestPath()
    print(g.graph)
    model.declareFirstDataset(g)
    changeVals = []
    for _ in range(10):
        for _ in range(10):
            g.changeGraph()
        change = model.run(g, src=0, dst=4)
        print("Change Value:")
        print(change)
        changeVals.append(change)
    print(changeVals)




"""

    Script Ran:
    python test_model.py --exp=DT_sp --logdir=shortestpath --num_steps=40 --dataset=shortestpath --cuda --infinite  --resume_iter=10000 --dt --lr 0.0001 --prog --alpha=0.3 --plot_name=65_10_test --plot_folder=testing_plots --plot --rank=15 --gen_rank=-5 --batch_size=1

    *FLAGS"
        Altered:
            --exp=DT_sp
            --logdir=shortestpath
            --num_steps=40
            --dataset=shortestpath
            --cuda
            --infinite
            --resume_iter=10000
            --dt
            --lr 0.0001
            --prog
            --alpha=0.3
            --plot_name=65_10_test
            --plot_folder=testing_plots
            --rank=15
            --gen_rank=-5
            --batch_size=1
        Defaults:



"""

    
