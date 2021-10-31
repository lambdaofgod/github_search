"""
this code is adapted from
unsupervised GraphSAGE example
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_sage_unsup.py
"""
import os
import tqdm
from typing import Union
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

import fasttext
from mlutil.feature_extraction import embeddings

import livelossplot

from gensim.models import KeyedVectors


import torch
import torch.nn as nn
from torch import optim
from torch import Tensor
import torch.nn.functional as F
from torch_cluster import random_walk

from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.nn import SAGEConv


class ResidualSAGEConv(SAGEConv):
    def __init__(self, **kwargs):
        super(ResidualSAGEConv, self).__init__(**kwargs)

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None
    ) -> Tensor:
        """ """
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)
            out += x_r

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out


class SAGENeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        pos_batch = random_walk(row, col, batch, walk_length=1, coalesced=False)[:, 1]

        n_edges = self.adj_t.size(1)
        neg_batch = torch.randint(0, n_edges, (batch.numel(),), dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(SAGENeighborSampler, self).sample(batch)


class SAGE(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, num_layers, sage_layer_cls=SAGEConv
    ):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(
                sage_layer_cls(in_channels=in_channels, out_channels=hidden_channels)
            )

    def forward(self, x, adjs):
        """
        x - embeddings of nodes
        adjs -
            list of ((edge_index, edge_data, size))
            data for edges (edge_index contains data in COO format)
            2 lists
            - one for neighboring vertices (positive samples)
            - one for random vertices (negative samples)
        """
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        out, pos_out, neg_out = x.split(x.size(0) // 3, dim=0)
        return out, pos_out, neg_out

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def loss(self, out, pos_out, neg_out):
        pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss = -pos_loss - neg_loss
        return loss


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.convs = torch.nn.ModuleList(
            [
                SAGEConv(in_channels, hidden_channels),
                SAGEConv(hidden_channels, hidden_channels),
                SAGEConv(hidden_channels, hidden_channels),
            ]
        )

        self.activations = torch.nn.ModuleList()
        self.activations.extend(
            [
                nn.PReLU(hidden_channels),
                nn.PReLU(hidden_channels),
                nn.PReLU(hidden_channels),
            ]
        )

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x_target = x  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x


def graph_infomax_corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


def graph_infomax_summary(z, *args, **kwargs):
    return torch.sigmoid(z.mean(dim=0))


def train(model, data, train_loader, optimizer, device):
    model.train()

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        x = data.x[n_id].to(device)
        outs = model(x, adjs)
        size = outs[0].size(0)
        loss = model.loss(*outs)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * size

    return total_loss / data.num_nodes


@torch.no_grad()
def get_val_loss(model, val_loader):
    model.eval()
    for batch_size, n_id, adjs in val_loader:
        adjs = [adj.to(device) for adj in adjs]
        x = data.x[n_id].to(device)
        outs = model(x, adjs)

        loss = unsupervised_graphsage_loss(model, x[n_id], adjs)
        total_loss += float(loss) * out.size(0)

    return total_loss
