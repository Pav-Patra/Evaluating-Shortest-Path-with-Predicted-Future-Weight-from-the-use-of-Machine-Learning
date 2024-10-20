""" dt_net_1d.py
    DeepThinking 1D convolutional neural network.

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.
    
    Developed for DeepThinking project
    October 2021

    Modified by Sean McLeish: https://github.com/mcleish7/DT-GNN

    Utilised by Pav Patra in custom SP_NN_model.py class
"""

import torch
from torch_geometric.nn import GINEConv, global_max_pool, GATConv, GMMConv, GENConv, PDNConv, SAGEConv, EdgeConv
from torch import nn
import torch.nn.functional as F
import torch_geometric
import sys

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914

class BasicBlock1D(nn.Module):
    """Basic residual block class 1D"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(nn.Conv1d(in_planes, self.expansion * planes,
                                                    kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class DTNet1D(nn.Module):
    """DeepThinking 1D Network model class"""

    def __init__(self, block, num_blocks, width, recall, group_norm=False, in_channels=1, graph = False, **kwargs):
        super().__init__()
        self.graph = graph
        self.width = int(width)
        self.recall = recall
        self.group_norm = group_norm

        proj_conv = nn.Conv1d(in_channels, width, kernel_size=3,
                              stride=1, padding=1, bias=False)

        conv_recall = nn.Conv1d(width + in_channels, width, kernel_size=3,
                                stride=1, padding=1, bias=False)

        if self.recall:
            recur_layers = [conv_recall, nn.ReLU()]
        else:
            recur_layers = []

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        head_conv1 = nn.Conv1d(width, width, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv1d(width, int(width/2), kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv1d(int(width/2), 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)
        self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                  head_conv2, nn.ReLU(),
                                  head_conv3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do=300, interim_thought=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2))).to(x.device)
        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)

            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)
            all_outputs[:, i] = out

        if self.training:
            return out, interim_thought
        if self.graph:
            return out, interim_thought
        return all_outputs


def dt_net_1d(width, **kwargs):
    return DTNet1D(BasicBlock1D, [2], width, recall=False)


def dt_net_recall_1d(width, **kwargs):
    try:
        if kwargs["graph"] == False:
            return DTNet1D(BasicBlock1D, [2], width, recall=True, in_channels=kwargs["in_channels"], graph =kwargs["graph"])
        else:
            # Adds extra 2 layers in core for graph models
            return DTNet1D(BasicBlock1D, [2,2], width, recall=True, in_channels=kwargs["in_channels"], graph =kwargs["graph"])
    except:
        return DTNet1D(BasicBlock1D, [2], width, recall=True, in_channels=kwargs["in_channels"], graph =False)