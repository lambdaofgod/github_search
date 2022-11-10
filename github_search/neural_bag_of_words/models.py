from dataclasses import dataclass
from operator import itemgetter

import numba
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn import base, feature_extraction
from torch import nn

EPS = 1e-6


class NBOWLayer(nn.Module):
    def __init__(
        self,
        token_weights: torch.FloatTensor,
        embedding: nn.Embedding,
        padding_idx: int = 0,
    ):
        super().__init__()
        assert len(token_weights) == embedding.num_embeddings
        self.token_weights = token_weights
        self.embedding = embedding
        self.padding_idx = padding_idx

    def forward(self, idxs):
        mask = 1 * (idxs != self.padding_idx)
        embs = self.embedding(idxs) * mask.unsqueeze(-1)
        token_weights = self.token_weights[idxs] * mask
        return torch.einsum("ijk,ij->ik", embs, token_weights)

    def to(self, device):
        self.embedding = self.embedding.to(device)
        self.token_weights = self.token_weights.to(device)
        return self


# WARNING
# assumes that each row has at least one nonzero value
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
