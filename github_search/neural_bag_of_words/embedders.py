from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
import sentence_transformers
import torch
import tqdm
from github_search.ir.models import *
from github_search.neural_bag_of_words.layers import NBOWLayer
from github_search.neural_bag_of_words.training_utils import NBOWPair
from github_search.utils import kwargs_only
from mlutil import sentence_transformers_utils

NBOWToEmbedderConverter = Callable[
    [NBOWLayer], sentence_transformers.SentenceTransformer
]


def make_sentence_transformer_nbow_model(
    nbow_layer,
    vocab: List[str],
    tokenize_fn: Callable[[str], List[str]],
    token_weights: np.ndarray,
    pooling_mean=True,
    pooling_max=False,
    max_seq_length: int = 1000,
) -> sentence_transformers.SentenceTransformer:
    """
    creates a model that wraps NBOW that conforms to sentence transformer interface
    """
    tokenizer = sentence_transformers_utils.CustomTokenizer(
        vocab, tokenize_fn=tokenize_fn
    )

    token_weights_dict = {word: weight for (word, weight) in zip(vocab, token_weights)}
    weights_layer = sentence_transformers.models.WordWeights(vocab, token_weights_dict)

    embeddings_layer = sentence_transformers.models.WordEmbeddings(
        tokenizer, nbow_layer.embedding.weight, max_seq_length=max_seq_length
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

    return sentence_transformers.SentenceTransformer(modules=modules)


def make_embedders_from_nbow_pair(
    nbow_pair: NBOWPair,
    *,
    query_embedder_fn: NBOWToEmbedderConverter,
    document_embedder_fn: NBOWToEmbedderConverter,
):
    return EmbedderPair(
        query_embedder=query_embedder_fn(nbow_pair.query_nbow),
        document_embedder=document_embedder_fn(nbow_pair.document_nbow),
    )
