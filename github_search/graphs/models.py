import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as ptgnn
from typing import Protocol

ptgnn.DeepGraphInfomax


class GraphNeuralNetwork(Protocol):
    """Base class for GNNs that can both embed graphs and be used for only node embedding"""

    def forward(self, x, edge_index, batch):
        """get graph embedding after pooling nodes"""

    def forward_pre_pooling(self, x, edge_index):
        """get graph before pooling"""

    def forward_without_pooling(self, x, edge_index, batch=None):
        """run graph without pooling, resulting in node embeddings"""



class GCNEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        n_node_features,
        graph_conv_cls=ptgnn.SAGEConv,
    ):
        super(GCNEncoder, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = graph_conv_cls(n_node_features, hidden_channels)
        self.prelu = nn.PReLU()
        self.conv2 = graph_conv_cls(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.prelu(x)
        x = self.conv2(x, edge_index)
        return x



class GCN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        n_node_features,
        final_layer_size,
        graph_conv_cls=ptgnn.SAGEConv,
    ):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = graph_conv_cls(n_node_features, hidden_channels)
        self.lin = torch.nn.Linear(2 * hidden_channels, final_layer_size)
        self.init_weights()

    def forward_pre_pooling(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        return x

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.forward_pre_pooling(x, edge_index)
        # 2. Readout layer
        x_max = ptgnn.global_max_pool(x, batch)  # [batch_size, hidden_channels]
        x_mean = ptgnn.global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = torch.column_stack([x_max, x_mean])
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x

    def forward_without_pooling(self, x, edge_index, batch=None):
        x = self.forward_pre_pooling(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(torch.column_stack([x, x]))

    def init_weights(self, init_fn=nn.init.orthogonal_):
        graph_layers = [self.conv1]
        nn.init.orthogonal_(self.lin.weight)
        for layer in graph_layers:
            init_fn(layer.lin_l.weight)
            init_fn(layer.lin_r.weight)
