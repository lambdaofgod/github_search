import torch
from torch import nn


class NBOW(nn.Module):
    def __init__(self, token_weights: torch.FloatTensor, embedding: nn.Embedding):
        super().__init__()
        assert len(token_weights) == embedding.num_embeddings
        self.token_weights = token_weights
        self.embedding = embedding

    def forward(self, idxs, mask):
        embs = self.embedding(idxs) * mask.unsqueeze(-1)
        token_weights_weights = self.token_weights[idxs] * mask
        return torch.einsum("ijk,ij->ik", embs, token_weights_weights)
