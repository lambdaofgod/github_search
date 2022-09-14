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
    AreaClassificationTrainingConfig, GNNTrainingConfig,
    MultilabelTaskClassificationTrainingConfig, SimilarityModelTrainingConfig)
from mlutil.feature_extraction import embeddings
from sklearn import base, metrics, preprocessing
from toolz import partial
from torch import nn
from github_search.graphs.nn_training import *


def run_infomax(upstream, product, hidden_channels, n_features):
    dataset = datasets.load_dataset(
        upstream["gnn.prepare_dataset_with_rnn"], ["area", "least_common_task", "tasks"]
    )
    device = "cuda"
    dim = dataset[0].x.shape[1]
    model = models.GCNEncoder(
        hidden_channels=hidden_channels,
        n_node_features=dim,
    ).to(device)
    train_infomax(
        model,
        n_features,
        dataset,
        epochs=1,
        device=device,
        plot_fig_path=str(product["plot_path"]),
        model_path=str(product["model_path"]),
        accumulation_steps=4,
    )


def run_multilabel_classification(upstream, product, hidden_channels, batch_size=32):
    dataset = pickle.load(open(upstream["gnn.prepare_dataset_splits"]["train"], "rb"))
    run_classification(upstream, product, dataset, hidden_channels, "tasks", batch_size)


def run_area_classification(upstream, product, hidden_channels, batch_size=32):
    dataset = pickle.load(open(upstream["gnn.prepare_dataset_splits"]["train"], "rb"))
    run_classification(upstream, product, dataset, hidden_channels, "area", batch_size)


def run_dependency_area_classification(
    upstream, product, hidden_channels, batch_size=32, epochs=2
):
    dataset = datasets.load_dataset(
        upstream["gnn.prepare_dataset_with_rnn"], ["area", "least_common_task", "tasks"]
    )
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
