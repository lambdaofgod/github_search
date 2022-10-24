import numpy as np
import torch
from github_search.neural_bag_of_words import models


def test_nbow_layer_shapes():
    vocab_size = 5
    dims = 10
    seq_length = 2
    idf = torch.ones(vocab_size)
    emb = torch.nn.Embedding(vocab_size, dims)
    nbow_model = models.NBOWLayer(idf, emb)

    idxs = torch.arange(seq_length).reshape(1, -1)
    mask = torch.ones(seq_length).reshape(1, -1)

    assert nbow_model(idxs, mask).shape == (1, dims)


def test_nbow_with_encoding_fn():
    emb_size = 5

    def encoding_fn(tokens):
        return torch.ones((len(tokens), emb_size))

    texts = ["foo bar", "foo baz"]
    nbow = models.NBOW.initialize_with_encoding_fn(texts, encoding_fn)
    assert nbow.transform(texts).shape == (len(texts), emb_size)
