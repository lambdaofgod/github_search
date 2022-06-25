import pickle

import livelossplot
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.nn as ptgnn
import torch_geometric.data as ptg_data
import tqdm
from torch import nn
import multiprocessing as mp


class GCN(torch.nn.Module):
    def __init__(
        self, hidden_channels, n_node_features, n_classes, graph_conv_cls=ptgnn.SAGEConv
    ):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.conv1 = graph_conv_cls(n_node_features, hidden_channels)
        self.conv2 = graph_conv_cls(hidden_channels, hidden_channels)
        self.conv3 = graph_conv_cls(hidden_channels, hidden_channels)
        self.lin = nn.Linear(2 * hidden_channels, n_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x_max = ptgnn.global_max_pool(x, batch)  # [batch_size, hidden_channels]
        x_mean = ptgnn.global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = torch.column_stack([x_max, x_mean])
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


def train(model, loader, device, plot_interval=20):

    loss_plot = livelossplot.PlotLosses()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    model.train()

    for (i, data) in tqdm.auto.tqdm(
        enumerate(loader, len(loader))
    ):  # Iterate in batches over the training dataset.
        out = model(
            data.x.to(device), data.edge_index.to(device), data.batch.to(device)
        )  # Perform a single forward pass.
        labels = data.label.to(devic e)
        loss = criterion(out, labels)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        accuracy = (out.argmax(dim=1) == labels).detach().cpu().numpy().mean()
        loss_plot.update({"loss": loss.item(), "accuracy": accuracy})
        optimizer.zero_grad()  # Clear gradients.
        if i % plot_interval == 0:
            loss_plot.send()
        scheduler.step(loss)
    return loss_plot


def run_classification(upstream, product, hidden_channels):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GCN(hidden_channels=32, n_node_features=200, n_classes=17).to(device)

    train_dataset = pickle.load(open(upstream["gnn.prepare_datasets"]["train"], "rb"))
    train_loader = ptg_data.DataLoader(train_dataset)
    train(model, train_loader, device)
    pd.Series([]).to_csv(str(product))
