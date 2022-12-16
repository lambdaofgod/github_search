import torch
from torch import nn
import numpy as np
import numba


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

    @classmethod
    def make_from_encoding_fn(
        cls,
        vocab,
        df_weights,
        encoding_fn,
        padding_idx: int = 0,
        device="cuda",
        scaling="log",
    ):
        if scaling == "log":
            df_weights = np.log2(df_weights + 1)
        inv_weights = 1 / df_weights
        weights = torch.Tensor(np.clip(inv_weights, 0, 1)).to(device)
        sorted_vocab = vocab.vocab.itos_
        embedded_vocab = encoding_fn(sorted_vocab)
        embeddings = nn.Embedding.from_pretrained(embedded_vocab).to(device)
        return cls(weights, embeddings)


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
