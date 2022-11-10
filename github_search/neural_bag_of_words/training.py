import itertools

import pytorch_lightning as pl
import torch
import torch.utils
from github_search.neural_bag_of_words.models import NBOWLayer
from quaterion.loss import MultipleNegativesRankingLoss


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

    def training_step(self, batch, batch_idx):
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
        self.log("train_loss", loss)
        return loss
