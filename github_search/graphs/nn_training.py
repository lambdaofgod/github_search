import ast
import logging
import pickle
from typing import Callable, List, Tuple, Union

import livelossplot
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.data as ptg_data
import torch_geometric.nn as ptgnn
import tqdm
from github_search import paperswithcode_task_areas, utils
from sklearn import metrics, preprocessing
from torch import nn


logging.basicConfig(level="INFO")

GraphDataList = List[ptg_data.Data]
GraphDataListWithLabels = List[Tuple[ptg_data.Data, Union[str, List[str]]]]


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


def train_gnn(
    model: nn.Module,
    loader: ptg_data.DataLoader,
    device: str,
    loss_fn: nn.Module,
    accuracy_fn,
    labels_dtype: np.dtype,
    plot_fig_path: str,
):

    loss_plot = livelossplot.PlotLosses(
        outputs=[livelossplot.outputs.MatplotlibPlot(figpath=plot_fig_path)]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = loss_fn.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    model.train()

    with tqdm.auto.tqdm(enumerate(loader), total=len(loader)) as pbar:
        for (i, data) in pbar:
            out = model(
                data.x.to(device), data.edge_index.to(device), data.batch.to(device)
            )  # Perform a single forward pass.
            labels = torch.tensor(np.array(data.encoded_label).astype(labels_dtype)).to(
                device
            )
            loss = loss_fn(out, labels)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            accuracy = accuracy_fn(data.encoded_label, out.detach().cpu().numpy())
            pbar.set_description(f"iteration {i}, loss: {round(loss.item(), 3)}")
            loss_plot.update({"loss": loss.item(), "accuracy": accuracy})
            optimizer.zero_grad()  # Clear gradients.
            scheduler.step(loss)
    loss_plot.send()
    return loss_plot


def add_label_encoded_class(graph_data: ptg_data.Data, label: int):
    graph_data.encoded_label = label
    return graph_data


def get_data_list_with_encoded_classes(
    graph_data_list: GraphDataListWithLabels, label_encoder
):
    labels = [label for g, label in graph_data_list]
    if not hasattr(label_encoder, "classes_"):
        label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)
    graph_data_list_with_labels = [
        add_label_encoded_class(graph_data, encoded_label)
        for ((graph_data, __), encoded_label) in zip(graph_data_list, encoded_labels)
    ]
    return (graph_data_list_with_labels, label_encoder)


def prepare_repos_metadata_df(repo_df, classification_column, area_tasks_path):
    if classification_column == "area":
        area_tasks_df = pd.read_csv(area_tasks_path)
        repo_df = paperswithcode_task_areas.prepare_paperswithcode_with_areas_df(
            repo_df, area_tasks_df
        )
    repo_df = repo_df[["repo", classification_column]].set_index("repo")[
        classification_column
    ]
    if classification_column == "tasks":
        repo_df = repo_df.apply(ast.literal_eval)
    return repo_df


def get_dataset_splits_with_labels(
    dataset: GraphDataList,
    train_path: str,
    test_path: str,
    classification_column: str,
    area_tasks_path: str,
):
    [train_repos_with_tasks, test_repos_with_tasks] = [
        prepare_repos_metadata_df(
            pd.read_csv(path), classification_column, area_tasks_path
        )
        for path in [train_path, test_path]
    ]
    train_dataset_with_labels, test_dataset_with_labels = [
        [
            (g, repos_with_tasks.loc[[g.graph_name]].iloc[0])
            for g in dataset
            if g.graph_name in repos_with_tasks.index
        ]
        for repos_with_tasks in [train_repos_with_tasks, test_repos_with_tasks]
    ]
    return train_dataset_with_labels, test_dataset_with_labels


class GNNConfig:

    label_encoders = {
        "tasks": preprocessing.MultiLabelBinarizer(),
        "area": preprocessing.LabelEncoder(),
    }
    accuracy_functions = {
        "tasks": utils.get_multilabel_samplewise_topk_accuracy,
        "area": utils.get_accuracy_from_scores,
    }
    loss_functions = {"tasks": nn.BCEWithLogitsLoss(), "area": nn.CrossEntropyLoss()}
    labels_dtype = {"tasks": np.float32, "area": np.int64}


def run_classification(
    upstream, product, hidden_channels, classification_column, batch_size=32
):
    assert (
        classification_column in GNNConfig.label_encoders.keys()
    ), "unsupported classification type"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"using {device}")
    area_tasks_path = str(upstream.get("prepare_area_grouped_tasks"))
    dataset = pickle.load(open(upstream["gnn.prepare_dataset"], "rb"))

    train_metadata_path, test_metadata_path = [
        str(upstream["prepare_repo_train_test_split"][split_part])
        for split_part in ["train", "test"]
    ]
    (
        train_dataset_with_labels,
        test_dataset_with_labels,
    ) = get_dataset_splits_with_labels(
        dataset,
        train_metadata_path,
        test_metadata_path,
        classification_column,
        area_tasks_path,
    )

    train_dataset, label_encoder = get_data_list_with_encoded_classes(
        train_dataset_with_labels, GNNConfig.label_encoders[classification_column]
    )
    test_dataset, __ = get_data_list_with_encoded_classes(
        test_dataset_with_labels, label_encoder
    )
    dim = train_dataset[0].x.shape[1]
    logging.info(f"using {len(train_dataset)} examples in train set")
    logging.info(f"using {len(test_dataset)} examples in test set")
    logging.info(f"fitting model with {len(label_encoder.classes_)} labels")
    pd.Series(label_encoder.classes_).to_csv("/tmp/output_classes.txt")
    logging.info(f"fitting model with {len(label_encoder.classes_)} labels")

    model = GCN(
        hidden_channels=hidden_channels,
        n_node_features=dim,
        n_classes=len(label_encoder.classes_),
    ).to(device)
    train_loader = ptg_data.DataLoader(train_dataset, batch_size)
    train_gnn(
        model,
        train_loader,
        device,
        GNNConfig.loss_functions[classification_column],
        GNNConfig.accuracy_functions[classification_column],
        GNNConfig.labels_dtype[classification_column],
        str(product)
    )


def run_multilabel_classification(upstream, product, hidden_channels, batch_size=32):
    run_classification(upstream, product, hidden_channels, "tasks", batch_size)


def run_area_classification(upstream, product, hidden_channels, batch_size=32):
    run_classification(upstream, product, hidden_channels, "area", batch_size)
