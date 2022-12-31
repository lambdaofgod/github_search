import itertools

import numpy as np
import torch
from quaterion.loss import MultipleNegativesRankingLoss, TripletLoss
from torch import nn


class MultipleNegativesRankingLossWrapper:

    _loss = MultipleNegativesRankingLoss()

    def __call__(self, loss_inputs):
        return self._loss(**loss_inputs, labels=None, subgroups=None)

    def prepare_loss_inputs(self, query_embeddings, document_embeddings):
        """
        prepare inpots for our nonstandard loss function that works on pairs
        """
        embeddings = torch.cat([query_embeddings, document_embeddings])
        pairs = torch.row_stack(
            [
                torch.arange(len(query_embeddings)),
                torch.arange(len(query_embeddings)) + len(query_embeddings),
            ]
        ).T
        return {"embeddings": embeddings, "pairs": pairs}


class EmbeddingMSELoss:

    _loss = nn.MSELoss()

    def __call__(self, loss_inputs):
        return self._loss(loss_inputs[0], loss_inputs[1])

    def prepare_loss_inputs(self, query_embeddings, document_embeddings):
        return (query_embeddings, document_embeddings)
