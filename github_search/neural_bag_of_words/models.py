import itertools

from sentence_transformers.util import batch_to_device
import pandas as pd
from typing import Protocol
import numba
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils
from torch import nn
from github_search.neural_bag_of_words.layers import NBOWLayer
from github_search.neural_bag_of_words import losses
from github_search.ir.models import EmbedderPair

EPS = 1e-6


LOSSES = {
    "multiple_negatives_ranking_loss": losses.MultipleNegativesRankingLossWrapper(),
    "mse_loss": losses.EmbeddingMSELoss(),
}


class Checkpointer(Protocol):
    def save_epoch_checkpoint(self, embedder_pair: EmbedderPair):
        pass

    def get_metrics_df(self, embedder_pair: EmbedderPair, epoch: int) -> pd.DataFrame:
        pass


class PairwiseEmbedderModule(pl.LightningModule):
    def __init__(
        self,
        embedder_pair: EmbedderPair,
        checkpointer: Checkpointer,
        loss_function_name: str,
        max_len: int,
        max_query_len: int,
        validation_metric_name: str,
        lr: float = 1e-3,
        train_query_embedder=True,
        weight_decay: float = 1e-6,
        max_grad_norm: float = 1.0,
        device="cuda",
        max_eval_len=2000,
    ):
        """
        lightning module for training neural bag of words model
        query and document models are potentially different NBOWLayers
        """
        super().__init__()
        self.validation_metric_name = validation_metric_name
        self.embedder_pair = embedder_pair
        self.query_embedder = embedder_pair.query_embedder
        self.document_embedder = embedder_pair.document_embedder
        self.loss = LOSSES[loss_function_name]
        self.weight_decay = weight_decay
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self._device = device
        self.max_document_len = max_len
        self.max_query_len = max_query_len
        self._i = 0
        self.checkpointer = checkpointer
        self.max_eval_len = max_eval_len
        self.train_query_embedder = train_query_embedder
        self.shuffle_documents = True

    def get_params(self):
        query_params = list(self.embedder_pair.query_embedder.parameters())
        document_params = list(self.embedder_pair.document_embedder.parameters())
        if self.train_query_embedder:
            return query_params + document_params
        else:
            return document_params

    def configure_optimizers(self):
        parameters = self.get_params()
        for p in parameters:
            p.requires_grad = True
        optimizer = torch.optim.Adam(
            parameters, lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @classmethod
    def get_random_token_nums(cls, token_nums, max_nums):
        random_idxs = np.random.randint(0, high=token_nums.shape[1], size=max_nums)
        return token_nums[:, random_idxs]

    def training_step(self, batch, batch_idx):
        """
        lightning training step
        """
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self.checkpointer.save_epoch_checkpoint(self.embedder_pair, self._i)
        metrics_dict = self.checkpointer.get_metrics_df(
            self.embedder_pair, self._i
        ).to_dict()
        del metrics_dict["name"]
        self.log("ir_metrics", metrics_dict, batch_size=1)
        validation_metric_value = metrics_dict[self.validation_metric_name]
        self.log(self.validation_metric_name, validation_metric_value, batch_size=1)
        self._i += 1
        __ = self.step(batch, batch_idx, name="validation")
        return validation_metric_value

    def step(self, batch, batch_idx, name):
        """
        step that also logs document length in training
        """
        queries_batch, documents_batch = batch

        query_embeddings = self.get_query_embeddings(queries_batch)
        documents_tokenized = self.prepare_token_inputs(
            documents_batch,
            self.max_document_len,
            self.shuffle_documents,
        )
        document_embeddings = self.get_document_embeddings(documents_tokenized)
        loss_inputs = self.loss.prepare_loss_inputs(
            query_embeddings, document_embeddings
        )
        loss_value = self.loss(loss_inputs)
        self._log_to_neptune(
            name,
            loss_value,
            documents_tokenized["sentence_lengths"],
            documents_tokenized["attention_mask"],
        )
        return loss_value

    def prepare_token_inputs(self, batch, max_seq_length, shuffle):
        batch_length = batch["input_ids"].shape[1]
        shuffled_idxs = torch.randperm(batch_length)
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                tensor = batch[key]
                if len(tensor.shape) > 1:
                    if shuffle:
                        tensor = tensor[:, shuffled_idxs]
                    tensor = tensor[:, :max_seq_length]
                batch[key] = tensor.to(self.device)
        return batch

    def get_query_embeddings(self, queries):
        return self.query_embedder(
            self.prepare_token_inputs(queries, self.max_query_len, shuffle=False)
        )["sentence_embedding"]

    def get_document_embeddings(self, documents_tokenized):
        return self.document_embedder(documents_tokenized)["sentence_embedding"]

    def _log_to_neptune(self, name, loss, document_lengths, attention_mask):
        if name == "validation":
            max_document_length = document_lengths.max()
            median_document_length = torch.median(document_lengths)
            pad_ratio = 1 - (1.0 * attention_mask).mean()
            self.log(f"{name}_loss", loss, batch_size=1)
            self.log(f"{name}_pad_ratio", pad_ratio, batch_size=1)
            self.log(
                f"{name}_batch_max_document_length", max_document_length, batch_size=1
            )
            self.log(
                f"{name}_batch_median_document_length",
                median_document_length,
                batch_size=1,
            )
        else:
            self.log(f"{name}_loss", loss, batch_size=1)

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
