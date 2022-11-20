from dataclasses import dataclass, field

import torch
import numpy as np
import tqdm
from github_search.neural_bag_of_words.models import NBOWLayer
from github_search.neural_bag_of_words.data import (
    NBOWNumericalizer,
    pack_sequences_as_tensors,
)
from mlutil import sentence_transformers_utils
import sentence_transformers

from typing import List, Callable


def make_sentence_transformer_nbow_model(
    nbow_layer,
    vocab: List[str],
    tokenize_fn: Callable[[str], List[str]],
    token_weights: np.ndarray,
    pooling_mean=True,
    pooling_max=False,
    max_seq_length: int = 1000 
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
