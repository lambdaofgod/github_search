import ast
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Union

import livelossplot
import numpy as np
import pandas as pd
import sentence_transformers
import torch
import torch.nn.functional as F
import torch_geometric.nn as ptgnn
import tqdm
from github_search import paperswithcode_task_areas, utils
from mlutil.feature_extraction import embeddings
from sklearn import base, metrics, preprocessing
from toolz import partial
from torch import nn
from github_search.graphs.data_preparation import make_trivial_graph


@dataclass
class GNNTrainingConfig(ABC):

    label_encoder: base.BaseEstimator
    loss_function: nn.Module
    labels_dtype: torch.dtype

    @classmethod
    @abstractmethod
    def accuracy_function(cls, encoded_label: np.ndarray, model_output: np.ndarray):
        pass

    def preprocess_target(self, data, device):
        return torch.Tensor(np.array(data.encoded_label).astype(self.labels_dtype)).to(
            device
        )


@dataclass
class MultilabelTaskClassificationTrainingConfig(GNNTrainingConfig):

    loss_function: nn.Module = field(default_factory=nn.BCEWithLogitsLoss)
    label_encoder: base.BaseEstimator = field(default_factory=preprocessing.MultiLabelBinarizer)
    labels_dtype: torch.dtype = field(default=np.float32)

    @classmethod
    def accuracy_function(cls, encoded_label, model_output):
        return utils.get_multilabel_samplewise_topk_accuracy(
            encoded_label, model_output
        )


@dataclass
class AreaClassificationTrainingConfig(GNNTrainingConfig):

    loss_function: nn.Module = field(default_factory=nn.CrossEntropyLoss())
    label_encoder: base.BaseEstimator = field(default_factory=preprocessing.LabelEncoder())
    labels_dtype: torch.dtype = field(default=np.int64)

    @classmethod
    def accuracy_function(cls, encoded_label, model_output):
        return utils.get_accuracy_from_scores(encoded_label, model_output)


class multiple_negatives_ranking_loss(nn.Module):
    def __init__(self, similarity_fn=sentence_transformers.util.cos_sim, scale=20):
        super(multiple_negatives_ranking_loss, self).__init__()
        self.scale = scale
        self.similarity_fn = similarity_fn

    def forward(self, embs, other_embs, labels=None):
        if labels is None:
            labels = torch.arange(embs.shape[0]).to(embs.device)
        assert embs.shape == other_embs.shape
        similarity = self.similarity_fn(embs, other_embs)
        return F.cross_entropy(similarity * self.scale, labels)


@dataclass
class SimilarityModelTrainingConfig(GNNTrainingConfig):

    embedder: embeddings.EmbeddingVectorizer
    model: nn.Module
    label_encoder: base.BaseEstimator
    loss_function: nn.Module
    labels_dtype: torch.dtype

    @staticmethod
    def from_embedder_and_model(
        embedder: embeddings.EmbeddingVectorizer,
        model: nn.Module,
        label_encoder_cls=preprocessing.LabelEncoder,
        loss_function_cls=multiple_negatives_ranking_loss,
        labels_dtype=torch.float32,
    ):
        return SimilarityModelTrainingConfig(
            embedder=embedder,
            model=model,
            loss_function=loss_function_cls(),
            label_encoder=label_encoder_cls(),
            labels_dtype=labels_dtype,
        )

    def preprocess_target(self, data, device):
        graph_data = make_trivial_graph(data.label, self.embedder).to(device)
        return self.model(
            graph_data.x, graph_data.edge_index, torch.arange(len(data.label)).cuda()
        )

    @classmethod
    def accuracy_function(cls, encoded_label, model_output):
        return utils.get_accuracy_from_scores(encoded_label, model_output)
