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
from bounter import bounter

import logging
from github_search.neural_bag_of_words.tokenization import TokenizerWithWeights


@dataclass
class EmbedderDataConfig:

    encoding_fn: Callable[[List[str]], np.ndarray]
    tokenizer: TokenizerWithWeights
    max_length: int


@dataclass
class QueryDocumentDataConfig:

    query_config: Union[str, EmbedderDataConfig]
    document_config: EmbedderDataConfig


@dataclass
class EmbedderFactory:

    data_config: QueryDocumentDataConfig

    def get_embedder_pair(self, query_data, document_data):

        if type(self.data_config.query_config) is not str:
            query_embedder = self.get_embedder(
                data=query_data,
                data_config=self.data_config.query_config,
                target_dim=None,
            )
            target_dim = None
        else:
            query_embedder = sentence_transformers.SentenceTransformer(
                self.data_config.query_config
            )
            target_dim = query_embedder.get_sentence_embedding_dimension()
        document_embedder = self.get_embedder(
            document_data, self.data_config.document_config, target_dim
        )
        return models.EmbedderPair(
            query_embedder=query_embedder, document_embedder=document_embedder
        )

    def get_embedder(
        self, data, data_config: EmbedderDataConfig, target_dim: Optional[int]
    ):
        nbow = NBOWData.make_from_encoding_fn(
            tokenizer_with_weights=data_config.tokenizer,
            encoding_fn=data_config.encoding_fn,
        )
        return nbow.make_sentence_transformer_nbow_model(
            tokenizer_with_weights=data_config.tokenizer,
            max_seq_length=data_config.max_length,
            target_dim=target_dim,
        )


@dataclass
class NBOWData:

    weights: torch.Tensor
    embeddings: torch.nn.Module

    @classmethod
    def make_from_encoding_fn(
        cls,
        tokenizer_with_weights,
        encoding_fn,
        padding_idx: int = 0,
        device="cuda",
        scaling="log",
    ):
        if scaling == "log":
            df_weights = np.log2(tokenizer_with_weights.frequency_weights + 1)
        inv_weights = 1 / df_weights
        weights = torch.Tensor(np.clip(inv_weights, 0, 1)).to(device)
        sorted_vocab = tokenizer_with_weights.tokenizer.get_vocab()
        embedded_vocab = encoding_fn(sorted_vocab)
        embeddings = nn.Embedding.from_pretrained(embedded_vocab).to(device)
        return cls(weights, embeddings)

    def make_sentence_transformer_nbow_model(
        self,
        tokenizer_with_weights: TokenizerWithWeights,
        target_dim=None,
        pooling_mean=True,
        pooling_max=False,
        max_seq_length: int = 1000,
    ) -> sentence_transformers.SentenceTransformer:
        """
        creates a model that wraps NBOW that conforms to sentence transformer interface
        """

        vocab = tokenizer_with_weights.tokenizer.get_vocab()
        tokenizer = tokenizer_with_weights.tokenizer
        token_weights_dict = {
            word: weight
            for (word, weight) in zip(vocab, self.weights.cpu().numpy().tolist())
        }
        weights_layer = sentence_transformers.models.WordWeights(
            vocab, token_weights_dict
        )

        embeddings_layer = sentence_transformers.models.WordEmbeddings(
            tokenizer, self.embeddings.weight, max_seq_length=max_seq_length
        )
        modules = [
            embeddings_layer,
            weights_layer,
            sentence_transformers.models.Pooling(
                embeddings_layer.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=pooling_mean,
                pooling_mode_max_tokens=pooling_max,
            ),
        ]
        if target_dim is not None:
            modules.append(
                sentence_transformers.models.Dense(
                    in_features=embeddings_layer.embeddings_dimension,
                    out_features=target_dim,
                )
            )

        return sentence_transformers.SentenceTransformer(modules=modules)
