from dataclasses import dataclass
from operator import itemgetter

import numba
import numpy as np
import pandas as pd
import torch
from sklearn import feature_extraction
from torch import nn

EPS = 1e-6


class NBOWLayer(nn.Module):
    def __init__(self, token_weights: torch.FloatTensor, embedding: nn.Embedding):
        super().__init__()
        assert len(token_weights) == embedding.num_embeddings
        self.token_weights = token_weights
        self.embedding = embedding

    def forward(self, idxs, mask):
        embs = self.embedding(idxs) * mask.unsqueeze(-1)
        token_weights = self.token_weights[idxs] * mask
        return torch.einsum("ijk,ij->ik", embs, token_weights)


class NBOW:
    def __init__(self, vectorizer, weights, embeddings, device="cuda"):
        self.vectorizer = vectorizer
        self.nbow_layer = NBOWLayer(weights, embeddings).to(device)
        self.device = device

    def __repr__(self):
        return f"NBOW(\n  vectorizer={self.vectorizer.__repr__()},\n  nbow_layer={str(self.nbow_layer)})"

    @classmethod
    def initialize_from_embeddings(cls, texts, embeddings, device="cuda", **kwargs):
        tfidf, weights = cls.fit_tfidf(texts, device, **kwargs)
        return NBOW(tfidf, weights, embeddings)

    @classmethod
    def initialize(cls, texts, embedding_dim, device="cuda", **kwargs):
        tfidf, weights = cls.fit_tfidf(texts, device, **kwargs)
        num_embeddings = len(tfidf.vocabulary_)
        embeddings = nn.Embedding(num_embeddings, embedding_dim)
        return NBOW(tfidf, weights, embeddings)

    @classmethod
    def initialize_with_encoding_fn(cls, texts, encoding_fn, device="cuda", **kwargs):
        tfidf, weights = cls.fit_tfidf(texts, device, **kwargs)
        sorted_vocab = list(
            dict(sorted(tfidf.vocabulary_.items(), key=itemgetter(1))).keys()
        )
        embedding_weights = encoding_fn(sorted_vocab)
        embeddings = nn.Embedding.from_pretrained(embedding_weights)
        return NBOW(tfidf, weights, embeddings)

    @classmethod
    def fit_tfidf(cls, texts, device, **kwargs):
        vectorizer = feature_extraction.text.TfidfVectorizer(**kwargs)
        vectorizer.fit(texts)
        weights = torch.Tensor(vectorizer.idf_).to(device)
        return vectorizer, 1.0 / weights + EPS

    def get_tensors(self, texts):
        sparse_texts = self.vectorizer.transform(texts)
        token_idxs, mask = self.get_input_batch_with_mask_from_sparse(sparse_texts)
        return token_idxs, mask

    def get_input_batch_with_mask_from_sparse(self, sparse_texts):
        nnz_idxs = np.where(sparse_texts.todense())
        max_count = pd.Series(nnz_idxs[0]).value_counts().max()
        token_idxs = gather_idxs(nnz_idxs, max_count)
        return torch.LongTensor(token_idxs).to(self.device), torch.IntTensor(
            token_idxs > 0
        ).to(self.device)

    def transform(self, texts):
        token_idxs, mask = self.get_tensors(texts)
        return self.nbow_layer(token_idxs, mask)


@numba.jit
def gather_idxs(nnz_idxs, max_count):
    row_idxs = np.unique(nnz_idxs[0])
    m = np.zeros((len(row_idxs), max_count), dtype="int64")
    nnz_x = nnz_idxs[0]
    nnz_y = nnz_idxs[1]
    for i in row_idxs:
        values = nnz_y[nnz_x == i]
        m[i, : len(values)] = values
    return m
