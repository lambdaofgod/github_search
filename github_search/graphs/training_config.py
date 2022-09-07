import ast
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import sentence_transformers
import torch
import torch.nn.functional as F
from mlutil.feature_extraction import embeddings
from sklearn import base, preprocessing
from torch import nn

from github_search import paperswithcode_task_areas, utils
from github_search.graphs import label_preprocessing


@dataclass
class GNNTrainingConfig(ABC):
    @classmethod
    @abstractmethod
    def accuracy_function(cls, encoded_label: np.ndarray, model_output: np.ndarray):
        pass

    @classmethod
    def get_labels(cls, data):
        return data.encoded_label

    @abstractmethod
    def preprocess_target(self, data):
        pass


@dataclass
class ClassificationConfig(GNNTrainingConfig):
    def preprocess_target(self, data):
        target = (
            torch.Tensor(np.array(data.encoded_label))
            .to(self.device)
            .type(self.labels_dtype)
        )
        return target


@dataclass
class MultilabelTaskClassificationTrainingConfig(ClassificationConfig):

    device: str
    loss_function: nn.Module = field(default_factory=nn.BCEWithLogitsLoss)
    label_encoder: base.BaseEstimator = field(
        default_factory=preprocessing.MultiLabelBinarizer
    )
    labels_dtype: torch.dtype = field(default=torch.float32)

    @classmethod
    def accuracy_function(cls, encoded_label, model_output):
        return utils.get_multilabel_samplewise_topk_accuracy(
            encoded_label, model_output
        )


@dataclass
class AreaClassificationTrainingConfig(ClassificationConfig):

    device: str
    loss_function: nn.Module = field(default_factory=nn.CrossEntropyLoss)
    label_encoder: base.BaseEstimator = field(
        default_factory=preprocessing.LabelEncoder
    )
    labels_dtype: torch.dtype = field(default=torch.int64)

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
    device: str

    @staticmethod
    def from_embedder_and_model(
        embedder: embeddings.EmbeddingVectorizer,
        model: nn.Module,
        device: str,
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
            device=device,
        )

    def preprocess_target(self, data):
        graph_data = label_preprocessing.make_graph_from_label_list(
            data.tasks, self.embedder
        ).to(self.device)
        return self.model(graph_data.x, graph_data.edge_index, graph_data.batch)

    @classmethod
    def accuracy_function(cls, encoded_label, model_output):
        return utils.get_accuracy_from_scores(encoded_label, model_output)

    def get_labels(self, data):
        return np.arange(len(data))
