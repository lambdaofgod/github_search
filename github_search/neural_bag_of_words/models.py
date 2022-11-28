from dataclasses import dataclass
from operator import itemgetter

import numba
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn import base, feature_extraction
from torch import nn
import itertools

import pytorch_lightning as pl
import torch
import torch.utils
from quaterion.loss import MultipleNegativesRankingLoss


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


class PairwiseNBOWModule(pl.LightningModule):
    def __init__(
        self,
        nbow_query: NBOWLayer,
        nbow_document: NBOWLayer,
        max_len: int = 2000,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        max_grad_norm: float = 1.0,
        device="cuda",
    ):
        """
        lightning module for training neural bag of words model
        query and document models are potentially different NBOWLayers
        """
        super().__init__()
        self.nbow_query = nbow_query
        self.nbow_document = nbow_document
        self.loss = MultipleNegativesRankingLoss()
        self.weight_decay = weight_decay
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self._device = device
        self.max_len = max_len

    def get_params(self):
        return list(
            itertools.chain.from_iterable(
                (
                    self.nbow_query.nbow_layer.embedding.parameters(),
                    self.nbow_document.nbow_layer.embedding.parameters(),
                )
            )
        )

    def configure_optimizers(self):
        parameters = list(
            itertools.chain.from_iterable(
                [self.nbow_query.parameters(), self.nbow_document.parameters()]
            )
        )
        for p in parameters:
            p.requires_grad = True
        optimizer = torch.optim.Adam(
            parameters, lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @classmethod
    def get_mask(cls, nums, padding_num):
        return (nums != padding_num) * 1

    def training_step(self, batch, batch_idx, name="train"):
        query_nums, document_nums = batch
        query_emb = self.nbow_query(query_nums[:, : self.max_len])
        document_emb = self.nbow_document(document_nums[:, : self.max_len])
        embs = torch.cat([query_emb, document_emb])
        ## DO POPRAWY
        pairs = torch.row_stack(
            [
                torch.arange(len(query_emb)),
                torch.arange(len(query_emb)) + len(query_emb),
            ]
        ).T
        loss = self.loss(embs, pairs, None, None)
        self.log(f"{name}_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, name="validation")
