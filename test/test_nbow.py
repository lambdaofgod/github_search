import torch
from github_search import neural_bag_of_words


def test_nbow_shapes():
    vocab_size = 5
    dims = 10
    seq_length = 2
    idf = torch.ones(vocab_size)
    emb = torch.nn.Embedding(vocab_size, dims)
    nbow_model = neural_bag_of_words.NBOW(idf, emb)

    idxs = torch.arange(seq_length).reshape(1, -1)
    mask = torch.ones(seq_length).reshape(1, -1)

    assert nbow_model(idxs, mask).shape == (1, dims)
