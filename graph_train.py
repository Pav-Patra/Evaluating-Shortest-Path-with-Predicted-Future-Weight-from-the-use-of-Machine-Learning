""" graph_train.py
    Training and testing code from graph models

    Collaboratively developed for IREM: https://github.com/yilundu/irem_code_release
    
    Developed for IREM project

    Edited by Sean McLeish for CS344 Discrete Maths Project 2023
"""
import torch
from graph_models import GraphEBM, GraphFC, GraphPonder, GraphRecurrent, DT_recurrent, GAT, DynamicEdgeConv, GCNConv
import torch.nn.functional as F
import os
from graph_dataset import Identity, ConnectedComponents, ShortestPath, MaxFlow, ArticulationPoints, MaxFlowNumpy
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim import Adam, SparseAdam, NAdam, RAdam, AdamW
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.data.batch import Batch
import os.path as osp
from torch.nn.utils import clip_grad_norm
import numpy as np
import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
from torchvision.utils import make_grid
#import seaborn as sns
from torch_geometric.nn import global_mean_pool
import sys
from random import randrange
import gc


def worker_init_fn(worker_id):
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))

class ReplayBuffer(object):
    # Developed for IREM project
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, inputs):
        batch_size = len(inputs)
        if self._next_idx >= len(self._storage):
            self._storage.extend(inputs)
        else:
            if batch_size + self._next_idx < self._maxsize:
                self._storage[self._next_idx:self._next_idx +
                              batch_size] = inputs
            else:
                split_idx = self._maxsize - self._next_idx
                self._storage[self._next_idx:] = inputs[:split_idx]
                self._storage[:batch_size - split_idx] = inputs[split_idx:]
        self._next_idx = (self._next_idx + batch_size) % self._maxsize

    def _encode_sample(self, idxes):
        datas = []
        for i in idxes:
            data = self._storage[i]
            datas.append(data)

        return datas

    def sample(self, batch_size, no_transform=False, downsample=False):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1)
                 for _ in range(batch_size)]
        return self._encode_sample(idxes), torch.Tensor(idxes)

    def set_elms(self, data, idxes):
        if len(self._storage) < self._maxsize:
            self.add(data)
        else:
            for i, ix in enumerate(idxes):
                self._storage[ix] = data[i]


"""Parse input arguments"""
parser = argparse.ArgumentParser(description='Train EBM model')

parser.add_argument('--train', action='store_true', help='whether or not to train')
parser.add_argument('--cuda', action='store_true', help='whether to use cuda or not')
parser.add_argument('--vary', action='store_true', help='vary size of graph')
parser.add_argument('--no_replay_buffer', action='store_true', help='utilize a replay buffer')

parser.add_argument('--dataset', default='identity', type=str, help='dataset to evaluate')
parser.add_argument('--logdir', default='cachedir', type=str, help='location where log of experiments will be stored')
parser.add_argument('--exp', default='default', type=str, help='name of experiments')

# training
parser.add_argument('--resume_iter', default=0, type=int, help='iteration to resume training')
parser.add_argument('--batch_size', default=512, type=int, help='size of batch of input to use')
parser.add_argument('--num_epoch', default=10000, type=int, help='number of epochs of training to run')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for training')
parser.add_argument('--log_interval', default=10, type=int, help='log outputs every so many batches')
parser.add_argument('--save_interval', default=1000, type=int, help='save outputs every so many batches')
parser.add_argument('--alpha', default=1.0, type=float, help='alpha hyperparameter')
parser.add_argument('--prog', action='store_true', help='use progressive loss')
parser.add_argument('--json_name', default='tracker', type=str, help='name of json file losses are written to')
parser.add_argument('--plot', action='store_true', help='plot the testing loss in graph')
parser.add_argument('--plot_name', default=None, type=str, help='prefix of file plot is written to')
parser.add_argument('--plot_folder', default=None, type=str, help='folder of file plot is written to')
parser.add_argument('--transfer_learn', action='store_true', help='do transfer learning') 
parser.add_argument('--transfer_learn_model', default=None, type=str, help='model path to learn from')
parser.add_argument('--random_noise', action='store_true', help='use random noise initialisation for interim thought')

# data
parser.add_argument('--data_workers', default=4, type=int, help='Number of different data workers to load data in parallel')

# EBM specific settings
parser.add_argument('--filter_dim', default=64, type=int, help='number of filters to use')
parser.add_argument('--rank', default=10, type=int, help='rank of matrix to use')
parser.add_argument('--num_steps', default=5, type=int, help='Steps of gradient descent for training')
parser.add_argument('--step_lr', default=100.0, type=float, help='step size of latents')
parser.add_argument('--latent_dim', default=64, type=int, help='dimension of the latent')
parser.add_argument('--decoder', action='store_true', help='utilize a decoder to output prediction')
parser.add_argument('--gen', action='store_true', help='evaluate generalization')
parser.add_argument('--gen_rank', default=5, type=int, help='Add additional rank for generalization')
parser.add_argument('--recurrent', action='store_true', help='utilize a decoder to output prediction')
parser.add_argument('--dt', action='store_true', help='use deep thinking net')
parser.add_argument('--gat', action='store_true', help='use GAT')
parser.add_argument('--edge_mpnn', action='store_true', help='use edge MPNN')
parser.add_argument('--gcn_mpnn', action='store_true', help='use gcn MPNN')
parser.add_argument('--ponder', action='store_true', help='utilize a decoder to output prediction')
parser.add_argument('--no_truncate_grad', action='store_true', help='not truncate gradient')
parser.add_argument('--iterative_decoder', action='store_true', help='utilize a decoder to output prediction')
parser.add_argument('--lr_throttle', action='store_true', help='utilize throttle in the decoder')

# Distributed training hyperparameters
parser.add_argument('--nodes', default=1, type=int, help='number of nodes for training')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus per nodes')
parser.add_argument('--node_rank', default=0, type=int, help='rank of node')
parser.add_argument('--capacity', default=50000, type=int, help='number of elements to generate')
parser.add_argument('--infinite', action='store_true', help='makes the dataset have an infinite number of elements')

#encoder choice for DT net, usually not used
parser.add_argument('--edge_mpnn_encoder', action='store_true', help='utilize a edge mpnn encoder')
parser.add_argument('--gcn_mpnn_encoder', action='store_true', help='utilize a gcn mpnn encoder')
parser.add_argument('--gat_encoder', action='store_true', help='utilize a gat encoder')

# only store loss values less than this number
best_test_error = 10.0
best_gen_test_error = 10.0


def average_gradients(model):
    # Developed for IREM project
    size = float(dist.get_world_size())

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def gen_answer(inp, FLAGS, model, pred, scratch, num_steps, create_graph=True):
    """
    Method to call the forward call of model

    Args:
        inp (Data): the data to run the network on
        FLAGS (dict): the input flags to the program
        model (torch.nn.Module): the model
        pred (torch.Tensor): the previous output, is random is the model does not need it
        scratch (torch.Tensor): 0 vector same shape as pred
        num_steps (int): number of steps to run for if an iterative model
        create_graph (bool, optional): create the dependency graph for the model
    Returns:
        pred: the last prediction
        preds: the array of predictions
        im_grads: the gradients
        energies: the energies of the model at each prediction
        scratch: updated verison of input scratch after the model has run
    """
    preds = []
    im_grads = []
    energies = []
    im_sups = []

    if FLAGS.decoder:
        print("decoder")
        pred = model.forward(inp)
        preds = [pred]
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
    elif FLAGS.recurrent:
        print("recurrent")
        preds = []
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
        state = None
        for i in range(num_steps):
            pred, state = model.forward(inp, state)
            preds.append(pred)
    elif FLAGS.dt: # DT GNN
        print("dt")
        preds = []
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
        state = None
        preds, state = model.forward(inp, iters_to_do = num_steps, iterim_though = None)
    elif FLAGS.gat:
        print("gat")
        pred = model.forward(inp)
        preds = [pred]
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
    elif FLAGS.edge_mpnn:
        print("edge_mpnn")
        pred = model.forward(inp)
        preds = [pred]
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
    elif FLAGS.gcn_mpnn:
        print("gcn_mpnn")
        pred = model.forward(inp)
        preds = [pred]
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
    elif FLAGS.ponder:
        print("ponder")
        preds = model.forward(inp, iters=num_steps)
        pred = preds[-1]
        im_grad = torch.zeros(1)
        im_grads = [im_grad]
        energies = [torch.zeros(1)]
        state = None
    elif FLAGS.iterative_decoder:
        print("iterative decoder")
        for i in range(num_steps):
            energy = torch.zeros(1)

            out_dim = model.out_dim

            im_merge = torch.cat([pred, inp], dim=-1)
            pred = model.forward(im_merge) + pred

            energies.append(torch.zeros(1))
            im_grads.append(torch.zeros(1))
    else: # IREM
        print("IREM")
        with torch.enable_grad():
            pred.requires_grad_(requires_grad=True)
            s = inp.size()
            scratch.requires_grad_(requires_grad=True)

            for i in range(num_steps):

                energy = model.forward(inp, pred)

                if FLAGS.no_truncate_grad:
                    im_grad, = torch.autograd.grad([energy.sum()], [pred], create_graph=create_graph)
                else:
                    if i == (num_steps - 1):
                        im_grad, = torch.autograd.grad([energy.sum()], [pred], create_graph=True)
                    else:
                        im_grad, = torch.autograd.grad([energy.sum()], [pred], create_graph=False)
                pred = pred - FLAGS.step_lr * im_grad

                preds.append(pred)
                energies.append(energy)
                im_grads.append(im_grad)

    return pred, preds, im_grads, energies, scratch


def ema_model(model, model_ema, mu=0.999):
    # Developed for IREM project
    for (model, model_ema) in zip(model, model_ema):
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            param_ema.data[:] = mu * param_ema.data + (1 - mu) * param.data


def sync_model(model):
    # Developed for IREM project
    size = float(dist.get_world_size())

    for model in model:
        for param in model.parameters():
            dist.broadcast(param.data, 0)


def init_model(FLAGS, device, dataset):
    """
    Method to initialise a model

    Args:
        FLAGS (dict): the input flags to the program
        device (str): CUDA or CPU
        dataset (Data): dataset object, for setting the size of the models inputs and outputs
    Returns:
        the models and optimiser
    """
    connecteddata = False
    if FLAGS.dataset == "connected":
        connecteddata = True
    articulationdata = False
    if FLAGS.dataset == "articulation":
        articulationdata = True
    if FLAGS.decoder:
        model = GraphFC(dataset.inp_dim, dataset.out_dim)
    elif FLAGS.ponder:
        model = GraphPonder(dataset.inp_dim, dataset.out_dim)
    elif FLAGS.recurrent:
        model = GraphRecurrent(dataset.inp_dim, dataset.out_dim)
    # The iterative decoder causes an error currently, we have contacted the IREM developers about this
    # elif FLAGS.iterative_decoder: 
    #     model = IterativeFC(dataset.inp_dim, dataset.out_dim, False)
    elif FLAGS.dt:
        # If statements allow us to change the encoder of the model at runtime, usually not used
        encoder = "GINEConv_encoder"
        if FLAGS.gcn_mpnn_encoder:
            encoder = "gcn_mpnn_encoder"
            model = DT_recurrent(dataset.inp_dim, dataset.out_dim, random_noise=FLAGS.random_noise, gcn_MPNN_encoder=True, connected=connecteddata)
        elif FLAGS.edge_mpnn_encoder:
            encoder = "edge_mpnn_encoder"
            model = DT_recurrent(dataset.inp_dim, dataset.out_dim, random_noise=FLAGS.random_noise, edge_MPNN_encoder=True, connected=connecteddata)
        elif FLAGS.gat_encoder:
            encoder = "gat_encoder"
            model = DT_recurrent(dataset.inp_dim, dataset.out_dim, random_noise=FLAGS.random_noise, GAT_encoder=True, connected=connecteddata)
        else:
            model = DT_recurrent(dataset.inp_dim, dataset.out_dim, random_noise=FLAGS.random_noise, connected=connecteddata, artdata = articulationdata)
        print(f"for DT-GNN, encoder is {encoder}, and connected is {connecteddata}")

        if FLAGS.lr_throttle: # learning rate throttling, as in the Deep Thinking models, usually not used
            base_params = [p for n, p in model.named_parameters() if "recur" not in n]
            recur_params = [p for n, p in model.named_parameters() if "recur" in n]

            recur_names = [n for n, p in model.named_parameters() if "recur" in n]
            non_recur_names = [n for n, p in model.named_parameters() if "recur" not in n]
            iters = FLAGS.num_steps
            lr = FLAGS.lr
            all_params = [{"params": base_params}, {"params": recur_params, "lr": lr / iters}]
        else:
            base_params = [p for n, p in model.named_parameters()]
            recur_params = []
            iters = 1
            all_params = [{"params": base_params}]
    elif FLAGS.gat:
        model = GAT(dataset.inp_dim, dataset.out_dim, connected=connecteddata, dt_call = articulationdata)
    elif FLAGS.edge_mpnn:
        model = DynamicEdgeConv(dataset.inp_dim, dataset.out_dim, k=6, connected=connecteddata, dt_call = articulationdata)
    elif FLAGS.gcn_mpnn:
        model = GCNConv(dataset.inp_dim, dataset.out_dim, connected=connecteddata, dt_call = articulationdata)
    else:
        model = GraphEBM(dataset.inp_dim, dataset.out_dim, False)
    model.to(device)

    if not FLAGS.dt: 
        all_params = model.parameters()
    optimizer = Adam(all_params, lr=FLAGS.lr, weight_decay=2e-6, eps=10**-6) # Adam optimiser used for all models
    return model, optimizer


def test(train_dataloader, model, FLAGS, step=0, gen=False, test_call=False, train_call=False, rig_call=False, alpha_rig_call=False):
    """
    Method to test a model

    Args:
        train_dataloader (DataLoader): The data loader to get the data required
        model (torch.nn.Module): model to test
        FLAGS (dict): the input flags to the program
        step (int, optional): iteration number
        gen (bool, optional): Can be used to distinguish between the out of distribution and in distribution testing
        test_call (bool, optional): If true we are calling this method during testing
        train_call (bool, optional): If true we are calling this method during training
        rig_call (bool, optional): If true we are calling this method from the testing rig file
        alpha_rig_call (bool, optional): If true we are calling this method from the alpha testing rig file
    Returns:
        the models output and enegies
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
    print(len(train_dataloader))
    print(lim)
    with torch.no_grad():
        for data in train_dataloader:
            print(counter)
            data = data.to(dev)
            im = data['y']
            print("data['y']:")
            print(im)

            pred = (torch.rand_like(data.edge_attr) - 0.5) * 2.
            print("pred:")
            print(pred)

            scratch = torch.zeros_like(pred)
            print("scratch:")
            print(scratch)

            pred_init = pred
            pred, preds, im_grad, energies, scratch = gen_answer(data, FLAGS, model, pred, scratch, FLAGS.num_steps, create_graph=False) # testing so no need for dependency graph
            print("preds after forward pass test:")
            print(preds)

            preds = torch.stack(preds, dim=0)
            print("Stack preds:")
            print(preds)
            energies = torch.stack(energies, dim=0).mean(dim=-1).mean(dim=-1) # used for the EBM 

            dist = (preds - im[None, :])
            print("dist:")
            print(dist)

            dist = torch.pow(dist, 2).mean(dim=-1)
            print("dist pow:")
            print(dist)

            dist = dist.mean(dim=-1)
            print("dist mean:")
            print(dist)
            min_idx = energies.argmin(dim=0)
            
            dist_min = dist[min_idx]
            min_dist_energy_list.append(dist_min)

            dist_list.append(dist.detach())
            energy_list.append(energies.detach())
            print(dist_list)
            print(energy_list)

            counter = counter + 1
            if counter > lim: # as the datasets are generated at run time we have to stop the testing manually
                break
    dist = torch.stack(dist_list, dim=0).mean(dim=0)
    print("Final dist:")
    print(dist)
    energies = torch.stack(energy_list, dim=0).mean(dim=0)
    min_dist_energy = torch.stack(min_dist_energy_list, dim=0).mean()
    if FLAGS.dt:
        min_dist_energy = dist[-1] # take the last output of the model as the final answer for deep thinking models

    print("Testing..................")
    print("last step error: ", dist)
    print("energy values: ", energies)
    print('test at step %d done!' % step)
    if gen:
        best_gen_test_error = min(best_gen_test_error, min_dist_energy.item())
        print("best gen test error: ", best_gen_test_error)
    else:
        best_test_error = min(best_test_error, min_dist_energy.item())
        print("best test error: ", best_test_error)
    model.train()
    if changed_rn: # changed random noise back if it was chnaged at the beginning of the method
        model.change_random_noise()
    if test_call:
        return dist
    if train_call:
        return min_dist_energy
    if rig_call or alpha_rig_call:
        return dist, energies

def get_output_for_prog_loss(inputs, max_iters, net):
    """
    Implementation of the progressive part of the progressive loss

    Args:
        inputs (Data): The data loader to get the data required
        max_iters (int): maximum number of iters to run for
        net (torch.nn.Module): model to evaluate
    Returns:
        outputs: the models output after a random number of iterations
        k: the ranomd number of iterations we tracked gradients for
    """
    # get features from n iterations to use as input
    n = randrange(0, max_iters)

    # do k iterations using intermediate features as input
    k = randrange(1, max_iters - n + 1)
    if n > 0:
        _, interim_thought = net(inputs, iters_to_do=n)
        interim_thought = interim_thought.detach()
    else:
        interim_thought = None

    outputs, _ = net(inputs, iters_elapsed=n, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k

def progressive_loss(model, data, FLAGS, outputs_max_iters, alpha):
    """
    Implementation of progressive loss

    Args:
        model (torch.nn.Module): model to evaluate
        data (Data): The data loader to get the data required
        FLAGS (dict): input parameters to the program
        outputs_max_iters (int): maximum number of iters to run for
        alpha (float): alpha value for progressive loss
    Returns:
        the progressive loss value
    """
    inputs = data
    targets = data.y
    max_iters = FLAGS.num_steps
    net = model
    criterion = torch.nn.MSELoss(reduction="none") # using MSE loss
    if alpha != 1:
        outputs_max_iters = outputs_max_iters[:, -1:]
        outputs_max_iters = torch.squeeze(outputs_max_iters)
        targets_squeezed = torch.squeeze(targets)
        loss_max_iters = criterion(outputs_max_iters, targets_squeezed)
    else:
        loss_max_iters = torch.zeros_like(targets).float()

    # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
    # so we save time by setting it equal to 0).
    if alpha != 0:
        outputs, k = get_output_for_prog_loss(inputs, max_iters, net)
        outputs = outputs[-1]
        loss_progressive = criterion(outputs, targets)
    else:
        loss_progressive = torch.zeros_like(targets).float()

    loss_max_iters_mean = loss_max_iters.mean()
    loss_progressive_mean = loss_progressive.mean()

    loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean

    return loss

def get_lr(optimizer):
    # Get the learning rate from the optimiser
    for param_group in optimizer.param_groups:
        return param_group['lr']

def cross_entropy(pred, soft_targets):
    """
    Implementation of cross entropy loss for real values 
    Based on: https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720

    Args:
        pred (torch.Tensor): prediction from a model
        soft_targets (torch.Tensor): target of model
    Returns:
        the cross entropy loss of the inputs
    """
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

def train(train_dataloader, test_dataloader, gen_dataloader, logger, model, optimizer, FLAGS, logdir, rank_idx):
    """
    Main training method

    Args:
        train_dataloader (DataLoader): training data dataloader
        test_dataloader (DataLoader): testing data dataloader, usually same data size as train data
        gen_dataloader (DataLoader): out of distribution data dataloader, doesn't value to be out of distribution
        logger (SummaryWriter): logger to write output to
        model (torch.nn.Module): model to train
        optimizer (torch.nn.optim): optimsier to use in training
        FLAGS (dict): dictionary of input parameters
        logdir (str): file to write outputs to
        rank_idx (int): used if training across mutliple machines
    """
    print(f"{FLAGS.dataset},{FLAGS.rank},{FLAGS.gen_rank}") # prints summary of most important flags
    alpha = FLAGS.alpha
    it = FLAGS.resume_iter
    optimizer.zero_grad()
    num_steps = FLAGS.num_steps
    losses=[]
    dev = torch.device("cuda")  
    replay_buffer = ReplayBuffer(10000)
    steps = [10001]
    if FLAGS.dt: # controls stepping lr for DT models 
        steps = [1000,3000,6000]
        if FLAGS.dataset == "connected" or FLAGS.dataset == "maxflow":
            steps = [3000,5000,8000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=10**-2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=0) # left commented for future users to experiment
    for epoch in range(FLAGS.num_epoch):
        for data in train_dataloader:
            torch.cuda.empty_cache()
            pred = (torch.rand_like(data.edge_attr) - 0.5) * 2
            data['noise'] = pred
            scratch = torch.zeros_like(pred)
            nreplay = FLAGS.batch_size

            if FLAGS.replay_buffer and len(replay_buffer) >= FLAGS.batch_size:
                data_list, levels = replay_buffer.sample(32)
                new_data_list = data.to_data_list()

                ix = int(FLAGS.batch_size * 0.5)

                nreplay = len(data_list) + 40
                data_list = data_list + new_data_list
                data = Batch.from_data_list(data_list)
                pred = data['noise']
            else:
                levels = np.zeros(FLAGS.batch_size)
                replay_mask = (
                    np.random.uniform(
                        0,
                        1,
                        FLAGS.batch_size) > 1.0)

            data = data.to(dev)
            pred = data['noise']
            pred, preds, im_grads, energies, scratch = gen_answer(data, FLAGS, model, pred, scratch, num_steps) # gets model prediction
            energies = torch.stack(energies, dim=0)

            # Choose energies to be consistent at the last step
            preds = torch.stack(preds, dim=1)
            
            im_grads = torch.stack(im_grads, dim=1)
            im = data['y']

            if FLAGS.ponder:
                im_loss = torch.pow(preds[:, :] - im[:, None, :], 2).mean(dim=-1).mean(dim=-1)
            else:
                im_loss = torch.pow(preds[:, -1:] - im[:, None, :], 2).mean(dim=-1).mean(dim=-1)
            
            im_loss = im_loss.mean() # MSE loss
            # ce_loss = cross_entropy(preds[:, -1:],im[:, None, :]) # cross entropy loss
            prog_loss = torch.ones(1) 

            loss = im_loss
            if FLAGS.prog: # progressive loss
                assert FLAGS.dt
                prog_loss = progressive_loss(model, data, FLAGS, preds, alpha)
                loss = prog_loss

            loss.backward()

            losses.append(loss.item())

            if FLAGS.replay_buffer: 
                data['noise'] = preds[:, -1].detach()
                data_list = data.cpu().to_data_list()

                replay_buffer.add(data_list[:nreplay])

            if FLAGS.gpus > 1:
                average_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10) # gradient clipping
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if it > 10000: # stops training after 10000 steps
                name = f"{FLAGS.logdir}/{FLAGS.exp}/loss_{FLAGS.json_name}.txt"
                with open(name, 'w+') as f: # storing the data as we do not expect reach the end of the loop in the set runtime
                    f.write(f"{FLAGS.dataset},{FLAGS.rank},{FLAGS.gen_rank} \n{losses}")
                sys.exit()

            if it % FLAGS.log_interval == 0 and rank_idx == 0: # frequency of logging output
                loss = loss.item()
                kvs = {}

                kvs['im_loss'] = im_loss.mean().item()

                string = f"Iteration {it:03d} | "

                for k, v in kvs.items():
                    string += "%s: %.6f" % (k,v)
                    logger.add_scalar(k, v, it)

                # print(string) #prints MSE loss
                # print("ce loss is ", ce_loss.item())
                if FLAGS.prog:
                    string += f" | prog loss: {prog_loss.item():.6f}"
                string += f" | lr: {get_lr(optimizer)}"
                print(string)
            if it % FLAGS.save_interval == 0 and rank_idx == 0 and it>0:
                model_path = osp.join(logdir, "model_latest.pth".format(it))
                ckpt = {'FLAGS': FLAGS}

                ckpt['model_state_dict'] = model.state_dict()
                ckpt['optimizer_state_dict'] = optimizer.state_dict()

                torch.save(ckpt, model_path)

                print("Testing performance .......................")
                in_dist_loss = test(test_dataloader, model, FLAGS, step=it, gen=False, train_call=True)

                print("Generalization performance .......................")
                out_dist_loss = test(gen_dataloader, model, FLAGS, step=it, gen=True, train_call=True)
                if ((in_dist_loss == best_test_error) and (best_gen_test_error == out_dist_loss)):
                    model_path = osp.join(logdir, "model_best.pth".format(it))
                    ckpt = {'FLAGS': FLAGS}
                    ckpt['model_state_dict'] = model.state_dict()
                    ckpt['optimizer_state_dict'] = optimizer.state_dict()
                    torch.save(ckpt, model_path)
            it += 1


def main_single(rank, FLAGS):
    """
    Main method for a single machine job

    Args:
        rank (int): size of the data to train on
        FLAGS (dict): dictionary of input parameters
    """
    # used if this is part of a multi machine job
    rank_idx = FLAGS.node_rank * FLAGS.gpus + rank 
    world_size = FLAGS.nodes * FLAGS.gpus 

    # selects the correct data sets
    if FLAGS.dataset == 'identity':
        dataset = Identity('train', FLAGS.rank, vary=FLAGS.vary)
        test_dataset = Identity('test', FLAGS.rank)
        gen_dataset = Identity('test', FLAGS.rank+FLAGS.gen_rank)
    elif FLAGS.dataset == 'connected':
        dataset = ConnectedComponents('train', FLAGS.rank, vary=FLAGS.vary)
        test_dataset = ConnectedComponents('test', FLAGS.rank)
        gen_dataset = ConnectedComponents('test', FLAGS.rank+FLAGS.gen_rank)
    elif FLAGS.dataset == 'shortestpath':
        dataset = ShortestPath('train', FLAGS.rank, vary=FLAGS.vary)
        test_dataset = ShortestPath('test', FLAGS.rank)
        gen_dataset = ShortestPath('test', FLAGS.rank+FLAGS.gen_rank)
    elif FLAGS.dataset == 'maxflow':
        dataset = MaxFlow('train', FLAGS.rank, vary=FLAGS.vary)
        test_dataset = MaxFlow('test', FLAGS.rank)
        gen_dataset = MaxFlow('test', FLAGS.rank+FLAGS.gen_rank)
    elif FLAGS.dataset == 'articulation':
        dataset = ArticulationPoints('train', FLAGS.rank, vary=FLAGS.vary)
        test_dataset = ArticulationPoints('test', FLAGS.rank)
        gen_dataset = ArticulationPoints('test', FLAGS.rank+FLAGS.gen_rank)

    # sets the testing data to be the gen data if required
    if FLAGS.gen:
        test_dataset = gen_dataset

    shuffle = True
    sampler = None

    # used if this is part of a multi machine job
    if world_size > 1:
        group = dist.init_process_group(backend='nccl', init_method='tcp://localhost:8113', world_size=world_size, rank=rank_idx, group_name="default")

    # torch.cuda.set_device(rank)
    device = torch.device('cuda')

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    FLAGS_OLD = FLAGS
    # Used to set up for transfer learning
    if FLAGS.transfer_learn:
        FLAGS.transfer_learn_model = "outputs/"+FLAGS.transfer_learn_model
        model_path = osp.join(FLAGS.transfer_learn_model, "model_best.pth")

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        # have to manually overwrite some FLAGS
        FLAGS = checkpoint['FLAGS']
        FLAGS.resume_iter = 0
        FLAGS.save_interval = FLAGS_OLD.save_interval
        FLAGS.nodes = FLAGS_OLD.nodes
        FLAGS.gpus = FLAGS_OLD.gpus
        FLAGS.node_rank = FLAGS_OLD.node_rank
        FLAGS.train = True
        FLAGS.batch_size = FLAGS_OLD.batch_size
        FLAGS.num_steps = FLAGS_OLD.num_steps
        FLAGS.exp = FLAGS_OLD.exp
        FLAGS.no_truncate_grad = FLAGS_OLD.no_truncate_grad
        FLAGS.rank = FLAGS_OLD.rank
        FLAGS.gen_rank = FLAGS_OLD.gen_rank
        FLAGS.step_lr = FLAGS_OLD.step_lr
        FLAGS.dataset = FLAGS_OLD.dataset
        FLAGS.dt = FLAGS_OLD.dt
        FLAGS.lr=FLAGS_OLD.lr
        FLAGS.prog = FLAGS_OLD.prog
        FLAGS.alpha = FLAGS_OLD.alpha
        FLAGS.random_noise = FLAGS_OLD.random_noise
        FLAGS.edge_mpnn = FLAGS_OLD.edge_mpnn
        FLAGS.edge_mpnn_encoder = FLAGS_OLD.edge_mpnn_encoder
        FLAGS.gcn_mpnn_encoder = FLAGS_OLD.gcn_mpnn_encoder
        FLAGS.gat_encoder = FLAGS_OLD.gat_encoder

        model, optimizer = init_model(FLAGS, device, dataset)
        state_dict = model.state_dict()

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        #freezes all weights but the encoder and decoder
        for name, param in model.named_parameters():
            param.requires_grad = False
            if (name == "conv3.lin.weight") or (name == "conv3.bias") or (name == "conv1.bias") or (name == "conv1.lin.weight"):
                param.requires_grad = True
    # used to set up for testing
    elif FLAGS.resume_iter != 0:
        # opens saved model
        try: 
            model_path = osp.join(logdir, "model_best.pth".format(FLAGS.resume_iter))
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        except:
            print("model did not converge...")
            print("!!!defaulting to model latest path!!!")
            model_path = osp.join(logdir, "model_latest.pth".format(FLAGS.resume_iter))
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        # for max flow the data is too large to generate at runtime so we used presaved examples
        if FLAGS.dataset == 'maxflow':
            print("maxflow numpy data being used")
            dataset = MaxFlowNumpy('train', FLAGS.rank)
            test_dataset = MaxFlowNumpy('test', FLAGS.rank)
            gen_dataset = MaxFlowNumpy('test', FLAGS.rank+FLAGS.gen_rank)
        FLAGS = checkpoint['FLAGS']
        FLAGS.resume_iter = FLAGS_OLD.resume_iter
        FLAGS.save_interval = FLAGS_OLD.save_interval
        FLAGS.nodes = FLAGS_OLD.nodes
        FLAGS.gpus = FLAGS_OLD.gpus
        FLAGS.node_rank = FLAGS_OLD.node_rank
        FLAGS.train = FLAGS_OLD.train
        FLAGS.batch_size = FLAGS_OLD.batch_size
        FLAGS.num_steps = FLAGS_OLD.num_steps
        FLAGS.exp = FLAGS_OLD.exp
        FLAGS.no_truncate_grad = FLAGS_OLD.no_truncate_grad
        FLAGS.gen_rank = FLAGS_OLD.gen_rank
        FLAGS.rank = FLAGS_OLD.rank
        FLAGS.step_lr = FLAGS_OLD.step_lr
        FLAGS.plot = FLAGS_OLD.plot
        FLAGS.plot_name = FLAGS_OLD.plot_name
        FLAGS.plot_folder = FLAGS_OLD.plot_folder
        FLAGS.dataset = FLAGS_OLD.dataset
        FLAGS.random_noise = FLAGS_OLD.random_noise
        FLAGS.edge_mpnn = FLAGS_OLD.edge_mpnn
        FLAGS.edge_mpnn_encoder = FLAGS_OLD.edge_mpnn_encoder
        FLAGS.gcn_mpnn_encoder = FLAGS_OLD.gcn_mpnn_encoder
        FLAGS.gat_encoder = FLAGS_OLD.gat_encoder
        FLAGS.lr_throttle = FLAGS_OLD.lr_throttle

        model, optimizer = init_model(FLAGS, device, dataset)
        state_dict = model.state_dict()

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # training a new model
    else:
        model, optimizer = init_model(FLAGS, device, dataset)
    #counts number of parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"This model has {pytorch_total_params/1e6:0.3f} million parameters.")
    # used if this is part of a multi machine job
    if FLAGS.gpus > 1:
        sync_model(model)

    # creates the data loader objects
    train_dataloader = DataLoader(dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=shuffle, worker_init_fn=worker_init_fn)
    test_dataloader = DataLoader(test_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=False, drop_last=True, worker_init_fn=worker_init_fn)
    gen_dataloader = DataLoader(gen_dataset, num_workers=FLAGS.data_workers, batch_size=FLAGS.batch_size // 2, shuffle=True, pin_memory=False, drop_last=True, worker_init_fn=worker_init_fn)

    logger = SummaryWriter(logdir)
    it = FLAGS.resume_iter

    if FLAGS.train:
        model.train()
    else:
        model.eval()

    if FLAGS.train:
        train(train_dataloader, test_dataloader, gen_dataloader, logger, model, optimizer, FLAGS, logdir, rank_idx)
    else: # testing
        test_array = test(test_dataloader, model, FLAGS, step=FLAGS.resume_iter, test_call=True)     # change test_call to false for less rigorous testing
        gen_array = test(gen_dataloader, model, FLAGS, step=FLAGS.resume_iter, gen=True, test_call=True)     # change test_call to false for less rigorous testing
        print(f"test_array for rank {FLAGS.rank} is {test_array}\n gen_array for gen_rank {FLAGS.rank+FLAGS.gen_rank} is {gen_array}")
        # if not training we create a graph of the testing outputs
        print(test_array)
        print(gen_array)
        if FLAGS.plot:
            in_dist = test_array.cpu().detach().numpy()
            out_dist = gen_array.cpu().detach().numpy()
            print("in_dist:")
            print(in_dist)
            print("out_dist:")
            print(out_dist)
            if FLAGS.dataset == "identity":
                data = "edge copy"
            elif FLAGS.dataset == "connected":
                data = "connected components"
            elif FLAGS.dataset == "shortestpath":
                data = "shortest path"
            else:
                data = "max flow"
            f, ax = plt.subplots(1, 1)
            if in_dist.size == 1: # for models with only one output use a scatter plot
                ax.scatter(0, in_dist, label=f"{FLAGS.rank} nodes for {data} experiment")
                ax.scatter(0, out_dist, label=f"{FLAGS.gen_rank+FLAGS.rank} nodes for {data} experiment")
            else: # for models with iterative outputs use a line graph
                ax.plot(in_dist, label=f"{FLAGS.rank} nodes for {data} experiment")
                print("Outputs of model from in_dist[in_dist]:")
                print(in_dist)
                ax.plot(out_dist, label=f"{FLAGS.gen_rank+FLAGS.rank} nodes for {data} experiment")
                print("Outputs of model from out_dist[out_dist]:")
                print(out_dist)
            ax.set(ylabel='MSE Loss', xlabel='Test-Time Iterations')
            ax1 = ax.twinx()
            ax1.set_ylim(ax.get_ylim())
            
            print("Outputs of model from in_dist[in_dist.size-1]")
            print(in_dist[in_dist.size-1])
            ax1.set_yticks([out_dist[out_dist.size-1], in_dist[in_dist.size-1]]) # plots the output of the model on a seperate model
            graph_file = f"{FLAGS.plot_name}_{FLAGS.dataset}_rank-{FLAGS.rank}_gen-{FLAGS.gen_rank+FLAGS.rank}_alpha-{str(FLAGS.alpha)[2:]}" # name of file
            folder = FLAGS.plot_folder
            if not osp.exists(folder):
                os.makedirs(folder)
            save_path = os.path.join(folder,graph_file)
            ax.legend(loc="upper left", bbox_to_anchor=(1, 0.5))
            plt.legend(loc='best')
            plt.grid()
            plt.savefig(save_path, bbox_inches="tight", dpi=500)


def main():
    """
    Main method for single and mutli machine job
    """
    FLAGS = parser.parse_args()
    FLAGS.replay_buffer = not FLAGS.no_replay_buffer
    FLAGS.logdir = "outputs/"+FLAGS.logdir
    if FLAGS.plot_folder != None:
        FLAGS.plot_folder = "plots/"+FLAGS.plot_folder
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)

    if not osp.exists(logdir):
        os.makedirs(logdir)
        
    if FLAGS.gpus > 1:
        mp.spawn(main_single, nprocs=FLAGS.gpus, args=(FLAGS,))
    else:
        main_single(0, FLAGS)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except:
        pass
    main()