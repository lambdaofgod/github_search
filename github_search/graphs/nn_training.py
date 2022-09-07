import ast
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union

import findkit
import h5py
import livelossplot
import numpy as np
import pandas as pd
import sentence_transformers
import torch
import torch.nn.functional as F
import torch_geometric.data as ptg_data
import torch_geometric.nn as ptgnn
import tqdm
from fastai.text import all as fastai_text
from findkit.feature_extractor import FastAITextFeatureExtractor
from github_search import logging_setup, paperswithcode_task_areas, utils
from github_search.graphs import datasets, models
from github_search.graphs.training_config import (
    AreaClassificationTrainingConfig,
    GNNTrainingConfig,
    MultilabelTaskClassificationTrainingConfig,
    SimilarityModelTrainingConfig,
)
from mlutil.feature_extraction import embeddings
from sklearn import base, metrics, preprocessing
from toolz import partial
from torch import nn

GraphDataListWithLabels = List[Tuple[ptg_data.Data, Union[str, List[str]]]]


def train_gnn(
    model: models.GraphNeuralNetwork,
    dataset: datasets.GraphDatasetLike,
    epochs: int,
    batch_size: int,
    device: str,
    config: GNNTrainingConfig,
    plot_fig_path: str,
    model_path: str,
    accumulation_steps: int,
):
    loss_plot = livelossplot.PlotLosses(
        outputs=[livelossplot.outputs.MatplotlibPlot(figpath=plot_fig_path)]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = config.loss_function.to(device)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001 
    )

    model.train()

    losses = []
    for __ in tqdm.auto.tqdm(range(epochs)):
        train_loader = ptg_data.DataLoader(dataset, batch_size, shuffle=True)
        with tqdm.auto.tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
            for (i, data) in pbar:
                target = config.preprocess_target(data)
                out = model(
                    data.x.to(device), data.edge_index.to(device), data.batch.to(device)
                )  # Perform a single forward pass.
                loss = loss_fn(out, target)  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                accuracy = config.accuracy_function(
                    config.get_labels(data), out.detach().cpu().numpy()
                )
                losses.append(loss.item())
                smoothed_loss = np.mean(losses[-20:])
                pbar.set_description(
                    f"iteration {i}, loss: {round(smoothed_loss.item(), 3)}"
                )
                loss_plot.update({"loss": smoothed_loss, "accuracy": accuracy})
                if (i + 1) % accumulation_steps == 0:
                    optimizer.zero_grad()  # Clear gradients.
                    scheduler.step(loss)
    loss_plot.send()
    torch.save(model, model_path)
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
    dataset: datasets.GraphDatasetLike,
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
            and len(g.tasks) > 0
            and type(g.least_common_task) is str
        ]
        for repos_with_tasks in [train_repos_with_tasks, test_repos_with_tasks]
    ]
    return train_dataset_with_labels, test_dataset_with_labels


def run_classification(
    upstream,
    product,
    dataset,
    hidden_channels,
    classification_column,
    gnn_class=models.GCN,
    batch_size=32,
    epochs=1,
    accumulation_steps=4,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = (
        MultilabelTaskClassificationTrainingConfig(device)
        if classification_column == "task"
        else AreaClassificationTrainingConfig(device)
    )
    logging.info(f"using {device}")
    area_tasks_path = str(upstream.get("prepare_area_grouped_tasks"))

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

    model = models.GCN(
        hidden_channels=hidden_channels,
        n_node_features=dim,
        final_layer_size=len(label_encoder.classes_),
    ).to(device)
    train_gnn(
        model,
        train_dataset,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        config=config,
        plot_fig_path=str(product["plot_path"]),
        model_path=str(product["model_path"]),
        accumulation_steps=2,
    )


def run_multilabel_classification(upstream, product, hidden_channels, batch_size=32):
    dataset = pickle.load(open(upstream["gnn.prepare_dataset_splits"]["train"], "rb"))
    run_classification(upstream, product, dataset, hidden_channels, "tasks", batch_size)


def run_area_classification(upstream, product, hidden_channels, batch_size=32):
    dataset = pickle.load(open(upstream["gnn.prepare_dataset_splits"]["train"], "rb"))
    run_classification(upstream, product, dataset, hidden_channels, "area", batch_size)


def run_dependency_area_classification(
    upstream, product, hdf5_dataset_path, hidden_channels, batch_size=32, epochs=2
):
    h5file = h5py.File(hdf5_dataset_path, "r")
    dataset = datasets.HDF5Dataset(h5file, ["area"])
    run_classification(
        upstream,
        product,
        dataset,
        hidden_channels,
        classification_column="area",
        batch_size=batch_size,
        epochs=epochs,
    )


def run_label_similarity_model(
    upstream,
    product,
    similarity_model_params,
    n_features,
    ulmfit_path,
    gnn_class=models.GCN,
    accumulation_steps=5,
):
    hidden_channels = similarity_model_params["hidden_channels"]
    epochs = similarity_model_params["epochs"]
    batch_size = similarity_model_params["batch_size"]
    query_column = "least_common_task"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"using {device}")
    area_tasks_path = str(upstream.get("prepare_area_grouped_tasks"))
    dataset = datasets.load_dataset(
        upstream["gnn.prepare_dataset_with_rnn"], ["area", "least_common_task", "tasks"]
    )

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
    model = gnn_class(
        hidden_channels=hidden_channels,
        n_node_features=dim,
        final_layer_size=n_features,
    ).to(device)
    fastai_learner = fastai_text.load_learner(ulmfit_path)
    embedder = FastAITextFeatureExtractor.build_from_learner(
        fastai_learner, max_length=48
    )
    config = SimilarityModelTrainingConfig.from_embedder_and_model(
        embedder, model, device
    )

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
        dataset,
        epochs,
        batch_size,
        device,
        config,
        str(product["plot_path"]),
        str(product["model_path"]),
        accumulation_steps=accumulation_steps,
    )
