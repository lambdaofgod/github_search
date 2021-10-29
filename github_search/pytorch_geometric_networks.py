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
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class PygGraphWrapper:
    """
    holds pytorch_geometric dataset and utils for mapping vertices to names
    """

    def __init__(
        self, featurizer, records_df, source_col="source", destination_col="destination"
    ):
        self.featurizer = featurizer
        self.records_df = records_df
        self.source_col = source_col
        self.destination_col = destination_col

        vertices = (
            pd.concat([records_df[source_col], records_df[destination_col]])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        self.vertex_mapping = pd.Series(data=vertices.index, index=vertices.values)
        self.inverse_vertex_mapping = vertices
        edge_index_source = self.vertex_mapping.loc[records_df[source_col]].values
        edge_index_destination = self.vertex_mapping.loc[
            records_df[destination_col]
        ].values
        edge_index = torch.tensor(
            np.row_stack([edge_index_source, edge_index_destination])
        )
        features = featurizer(self.vertex_mapping.index)
        self.dataset = Data(torch.tensor(features), torch.tensor(edge_index))

    def get_sub_dataset_wrapper(self, vertex_subset):
        records_subdf = self.records_df[
            self.records_df[self.source_col].isin(vertex_subset)
            | self.records_df[self.destination_col].isin(vertex_subset)
        ]
        return PygGraphWrapper(self.featurizer, records_subdf)

    def get_vertex_embeddings(self, vertex_subset, model):
        sub_dataset_wrapper = self.get_sub_dataset_wrapper(vertex_subset)
        features = (
            model.full_forward(
                sub_dataset_wrapper.dataset.x, sub_dataset_wrapper.dataset.edge_index
            )
            .cpu()
            .detach()
            .numpy()
        )
        return features[sub_dataset_wrapper.vertex_mapping.loc[vertex_subset]]


class ResidualSAGEConv(SAGEConv):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

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
        self, in_channels, hidden_channels, num_layers, sage_layer_cls=ResidualSAGEConv
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
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


def unsupervised_graphsage_loss(model, xs, adjs):
    out = model(xs, adjs)
    out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

    pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
    neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
    loss = -pos_loss - neg_loss
    return loss, out.size(0)


def train(model, data, train_loader, optimizer, device):
    model.train()

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        x = data.x[n_id].to(device)
        loss, size = unsupervised_graphsage_loss(model, x, adjs)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * size

    return total_loss / data.num_nodes


@torch.no_grad()
def get_val_loss(model, val_loader):
    model.eval()
    for batch_size, n_id, adjs in val_loader:
        adjs = [adj.to(device) for adj in adjs]
        loss = unsupervised_graphsage_loss(model, x[n_id], adjs)
        total_loss += float(loss) * out.size(0)

    return total_loss
