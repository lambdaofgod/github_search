from dataclasses import dataclass, field

import numpy as np
import sentence_transformers
import torch
from torch import nn
from mlutil import sentence_transformers_utils
from github_search.neural_bag_of_words.tokenization import TokenizerWithWeights


import torch
from sentence_transformers import SentenceTransformer, models


class SingleAttentionLayer(nn.Module):

    def __init__(self, embedding_dim, n_heads=24, device="cuda"):
        super(SingleAttentionLayer, self).__init__()
        self.device = device
        self.attn_layer = nn.MultiheadAttention(embedding_dim, n_heads, batch_first=True).to(device)

    def forward(self, features):
        token_embeddings = features['token_embeddings']
        embeddings, mask = self.attn_layer(token_embeddings, token_embeddings, token_embeddings)
        features.update({"token_embeddings": embeddings})
        return features


@dataclass
class NBOWModel:

    weights: torch.Tensor
    embeddings: torch.nn.Module

    @classmethod
    def make_nbow_from_encoding_fn(
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
