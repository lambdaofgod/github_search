from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import numpy as np
import sentence_transformers
import torch
from torch import nn
from github_search.ir import models
from github_search.utils import kwargs_only
from mlutil import sentence_transformers_utils
from collections import Counter
import pickle
import tqdm
from enum import Enum
import logging
from github_search.neural_bag_of_words.tokenization import TokenizerWithWeights
from github_search.neural_bag_of_words.models import NBOWModel


class EmbedderDataConfig:
    pass


@dataclass
class SentenceTransformerDataConfig:
    model_name: str


@dataclass
class ImplementedEmbedderDataConfig:
    encoding_fn: Callable[[List[str]], np.ndarray]
    tokenizer: TokenizerWithWeights
    max_length: int


def get_data_config(model_str, encoding_fn, max_length, tokenizer):
    if ModelPairConfig.is_model_implemented(model_str):
        return ImplementedEmbedderDataConfig(
            encoding_fn=encoding_fn, max_length=max_length, tokenizer=tokenizer
        )
    else:
        return SentenceTransformerDataConfig(model_str)


@dataclass
class ModelPairConfig:
    query_config: EmbedderDataConfig
    document_config: EmbedderDataConfig
    query_model_type: str
    document_model_type: str

    @staticmethod
    def is_model_implemented(model):
        return model in ["nbow", "single_layer_transformer"]


class EmbedderFactory:
    def __init__(self, data_config: ModelPairConfig):
        self.config = data_config

    @classmethod
    def make_from_train_val_config(
        cls,
        train_val_config,
        encoding_fn,
        max_length,
        query_tokenizer,
        document_tokenizer,
    ):
        query_embedder_type = train_val_config.query_embedder
        document_embedder_type = train_val_config.document_embedder

        query_config = get_data_config(
            train_val_config.query_embedder, encoding_fn, 100, query_tokenizer
        )
        document_config = get_data_config(
            train_val_config.document_embedder,
            encoding_fn,
            max_length,
            document_tokenizer,
        )
        model_config = ModelPairConfig(
            query_config, document_config, query_embedder_type, document_embedder_type
        )
        return cls(model_config)

    def get_embedder_pair(self):
        query_embedder, dim = self.get_embedder_with_target_dim(
            self.config.query_config, None
        )
        document_embedder, _ = self.get_embedder_with_target_dim(
            self.config.document_config, dim
        )
        return models.EmbedderPair(
            query_embedder=query_embedder, document_embedder=document_embedder
        )

    def get_embedder_with_target_dim(self, config, target_dim):
        if type(config) is str:
            embedder = sentence_transformers.SentenceTransformer(config)
            target_dim = embedder.get_sentence_embedding_dimension()
        else:
            embedder = self.get_nbow_embedder(
                config=config,
                target_dim=target_dim,
            )
            target_dim = None
        return embedder, target_dim

    def get_nbow_embedder(self, config, target_dim: Optional[int]):
        nbow = NBOWModel.make_nbow_from_encoding_fn(
            tokenizer_with_weights=config.tokenizer,
            encoding_fn=config.encoding_fn,
        )
        return nbow.make_sentence_transformer_nbow_model(
            tokenizer_with_weights=config.tokenizer,
            max_seq_length=config.max_length,
            target_dim=target_dim,
        )

    def get_single_attention_layer():
        pass
