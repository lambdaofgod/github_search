import ast
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import findkit
import livelossplot
import numpy as np
import pandas as pd
import sentence_transformers
import torch
import torch.nn.functional as F
import torch_geometric.data as ptg_data
import torch_geometric.nn as ptgnn
import tqdm
from mlutil.feature_extraction import embeddings
from sklearn import base, metrics, preprocessing
from toolz import partial
from torch import nn

from github_search import logging_setup, paperswithcode_task_areas, utils
from github_search.graphs.models import GCN
from github_search.graphs.training_config import (
    AreaClassificationTrainingConfig,
    GNNTrainingConfig,
    MultilabelTaskClassificationTrainingConfig,
    SimilarityModelTrainingConfig,
)

GraphDataList = List[ptg_data.Data]
GraphDataListWithLabels = List[Tuple[ptg_data.Data, Union[str, List[str]]]]


def train_gnn(
    model: nn.Module,
    dataset: GraphDataList,
    epochs: int,
    batch_size: int,
    device: str,
    config: GNNTrainingConfig,
    plot_fig_path: str,
    model_path: str,
):
    loss_plot = livelossplot.PlotLosses(
        outputs=[livelossplot.outputs.MatplotlibPlot(figpath=plot_fig_path)]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = config.loss_function.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=20
    )

    model.train()

    losses = []
    for __ in tqdm.auto.tqdm(range(epochs)):
        train_loader = ptg_data.DataLoader(dataset, batch_size, shuffle=True)
        with tqdm.auto.tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
            for (i, data) in pbar:
                target = config.preprocess_target(data, device)
                out = model(
                    data.x.to(device), data.edge_index.to(device), data.batch.to(device)
                )  # Perform a single forward pass.
                loss = loss_fn(out, target)  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                accuracy = config.accuracy_function(
                    data.encoded_label, out.detach().cpu().numpy()
                )
                losses.append(loss.item())
                smoothed_loss = np.mean(losses[-20:])
                pbar.set_description(
                    f"iteration {i}, loss: {round(smoothed_loss.item(), 3)}"
                )
                loss_plot.update({"loss": smoothed_loss, "accuracy": accuracy})
                optimizer.zero_grad()  # Clear gradients.
                scheduler.step(loss)
    loss_plot.send()
    model.save(model_path)
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


def run_classification(
    upstream, product, hidden_channels, classification_column, batch_size=32, epochs=1
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = (
        AreaClassificationTrainingConfig()
        if classification_column == "name"
        else MultilabelTaskClassificationTrainingConfig()
    )
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
        train_dataset_with_labels,
        config.label_encoder,
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
        final_layer_size=len(label_encoder.classes_),
    ).to(device)
    train_gnn(
        model,
        train_dataset,
        epochs,
        batch_size,
        device,
        config,
        str(product["plot_path"]),
        str(product["model_path"]),
    )


def run_multilabel_classification(upstream, product, hidden_channels, batch_size=32):
    run_classification(upstream, product, hidden_channels, "tasks", batch_size)


def run_area_classification(upstream, product, hidden_channels, batch_size=32):
    run_classification(upstream, product, hidden_channels, "area", batch_size)


def run_label_similarity_model(
    upstream,
    product,
    sentence_transformer_model_name,
    hidden_channels,
    batch_size=256,
    epochs=3,
):
    query_column = "least_common_task"
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
        query_column,
        area_tasks_path,
    )
    dim = dataset[0].x.shape[1]
    n_classes = len(set(l for (g, l) in train_dataset_with_labels))
    model = GCN(
        hidden_channels=hidden_channels,
        n_node_features=dim,
        final_layer_size=n_classes,
    ).to(device)
    embedder = findkit.feature_extractor.SentenceEncoderFeatureExtractor(
        sentence_transformers.SentenceTransformer(sentence_transformer_model_name)
    )
    config = SimilarityModelTrainingConfig.from_embedder_and_model(embedder, model)

    train_dataset, label_encoder = get_data_list_with_encoded_classes(
        train_dataset_with_labels, config.label_encoder
    )
    # test_dataset, __ = get_data_list_with_encoded_classes(
    #    test_dataset_with_labels, label_encoder
    # )
    logging.info(f"using {len(train_dataset)} examples in train set")
    # logging.info(f"using {len(test_dataset)} examples in test set")
    logging.info(f"fitting model with {len(label_encoder.classes_)} labels")
    pd.Series(label_encoder.classes_).to_csv("/tmp/output_classes.txt")
    logging.info(f"fitting model with {len(label_encoder.classes_)} labels")

    train_gnn(
        model,
        train_dataset,
        epochs,
        batch_size,
        device,
        config,
        str(product["plot_path"]),
        str(product["model_path"]),
    )
