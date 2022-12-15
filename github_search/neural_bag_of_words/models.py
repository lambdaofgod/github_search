import itertools

import pandas as pd
from typing import Protocol
import numba
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils
from quaterion.loss import MultipleNegativesRankingLoss, TripletLoss
from torch import nn

EPS = 1e-6


LOSSES = {
    "multiple_negatives_ranking_loss": MultipleNegativesRankingLoss,
    "triplet_loss": TripletLoss,
}


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


def get_lengths_series(idxs, padding_value):
    non_pad_x, non_pad_y = torch.where(idxs != padding_value)
    __, indices = torch.unique(non_pad_x, return_inverse=True, sorted=True)
    lengths = torch.bincount(indices)
    return pd.Series(lengths.cpu().numpy())


class Checkpointer(Protocol):
    def save_epoch_checkpoint(self, nbow_query, nbow_document, epoch):
        pass

    def get_metrics_df(self, nbow_query, nbow_document, epoch) -> pd.DataFrame:
        pass


class PairwiseNBOWModule(pl.LightningModule):
    def __init__(
        self,
        nbow_query: NBOWLayer,
        nbow_document: NBOWLayer,
        checkpointer: Checkpointer,
        loss_function_name: str,
        padding_value: int,
        max_len: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        max_grad_norm: float = 1.0,
        device="cuda",
        max_query_len=None,
        max_eval_len=2000,
    ):
        """
        lightning module for training neural bag of words model
        query and document models are potentially different NBOWLayers
        """
        super().__init__()
        self.nbow_query = nbow_query
        self.nbow_document = nbow_document
        self.loss = LOSSES[loss_function_name]()
        self.weight_decay = weight_decay
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self._device = device
        self.max_len = max_len
        self.max_query_len = max_len if max_query_len is None else max_query_len
        self._i = 0
        self.checkpointer = checkpointer
        self.padding_value = padding_value
        self.max_eval_len = max_eval_len

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

    @classmethod
    def get_random_token_nums(cls, token_nums, max_nums):
        random_idxs = np.random.randint(0, high=token_nums.shape[1], size=max_nums)
        return token_nums[:, random_idxs]

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self.checkpointer.save_epoch_checkpoint(
            self.nbow_query, self.nbow_document, self._i
        )
        metrics_dict = self.checkpointer.get_metrics_df(
            self.nbow_query, self.nbow_document, self._i
        ).to_dict()
        del metrics_dict["name"]
        self.log("ir_metrics", metrics_dict)
        self.log("accuracy@10", metrics_dict["accuracy@10"])
        self._i += 1
        return self.step(batch, batch_idx, name="validation")

    def step(self, batch, batch_idx, name):
        query_nums, document_nums = batch

        document_lengths = get_lengths_series(document_nums, self.padding_value)
        if name == "train":
            document_nums = self.truncate_by_length(
                document_nums, document_lengths, self.max_len, self.max_eval_len
            )
        else:
            document_nums = document_nums[:, : self.max_eval_len]

        model_inputs = self.prepare_model_inputs(query_nums, document_nums)
        loss = self.loss(**model_inputs, labels=None, subgroups=None)
        self._log_to_neptune(name, loss, document_nums, document_lengths)
        return loss

    def prepare_model_inputs(self, query_nums, document_nums):
        query_emb = self.nbow_query(query_nums[:, : self.max_query_len])
        document_emb = self.nbow_document(document_nums)
        embs = torch.cat([query_emb, document_emb])
        ## DO POPRAWY
        pairs = torch.row_stack(
            [
                torch.arange(len(query_emb)),
                torch.arange(len(query_emb)) + len(query_emb),
            ]
        ).T
        return {"embeddings": embs, "pairs": pairs}

    def _log_to_neptune(self, name, loss, document_nums, document_lengths):
        if name == "train":
            max_document_length = document_nums.shape[1]
            median_document_length = document_lengths.median()
            pad_ratio = ((document_nums == self.padding_value) * 1.0).mean()
            self.log(f"{name}_loss", loss)
            self.log(f"{name}_pad_ratio", pad_ratio)
            self.log(f"{name}_batch_max_document_length", max_document_length)
            self.log(f"{name}_batch_median_document_length", median_document_length)
        else:
            self.log(f"{name}_loss", loss)

    @classmethod
    def truncate_by_length(cls, nums, lengths, truncation_mode, max_length):
        if truncation_mode == "median":
            median = int(lengths.median())
            return nums[:, :median]
        elif truncation_mode == "mean":
            mean = int(lengths.mean())
            truncated_mean = min(mean, max_length)
            return nums[:, :truncated_mean]
        elif type(truncation_mode) is int:
            max_length = truncation_mode
            return nums[:, :max_length]
