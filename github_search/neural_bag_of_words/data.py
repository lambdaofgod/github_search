import random
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Dict

from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.utils
import torchtext
from github_search import python_tokens


Tokenizer = Callable[[List[str]], List[List[str]]]


def prepare_dependency_texts(dependency_records_df):
    dependency_records_df = dependency_records_df[
        dependency_records_df["source"] != "<ROOT>"
    ]
    return dependency_records_df.groupby("repo")["destination"].agg(" ".join)


def split_tokenizer(s):
    return s.split()


def shuffle_list(l):
    return np.array(l, dtype=np.int32)[np.random.permutation(len(l))]


def pack_sequences_as_tensors(sequences, padding_value, shuffle):
    np_tensors = [shuffle_list(s) if shuffle else s for s in sequences]
    return torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(t) for t in np_tensors],
        batch_first=True,
        padding_value=padding_value,
    )


def collate_fn(
    batch, padding_value_query, padding_value_document, shuffle=False, device="cuda"
):
    queries, documents = zip(*batch)
    return pack_sequences_as_tensors(queries, padding_value_query, shuffle=False).to(
        device
    ), pack_sequences_as_tensors(documents, padding_value_document, shuffle).to(device)


@dataclass
class NBOWNumericalizer:

    tokenizer: Tokenizer
    vocab: torchtext.vocab.Vocab

    def numericalize_texts(self, texts):
        tokenized_texts = [self.tokenizer(t) for t in texts]
        numericalized_texts = [
            list(it)
            for it in list(
                torchtext.data.functional.numericalize_tokens_from_iterator(
                    self.vocab, tokenized_texts
                )
            )
        ]
        return numericalized_texts

    @classmethod
    def build_from_texts(cls, texts: List[str], tokenizer: Tokenizer, min_freq=5):
        tokenized_texts = [tokenizer(t) for t in texts]
        vocab = torchtext.vocab.build_vocab_from_iterator(
            tokenized_texts, min_freq=min_freq
        )
        if not "<pad>" in vocab.vocab.itos_:
            vocab.vocab.append_token("<pad>")
        vocab.set_default_index(vocab["<pad>"])
        return NBOWNumericalizer(tokenizer, vocab)

    def get_padding_idx(self):
        return self.vocab.vocab["<pad>"]


class QueryDocumentDataset(torch.utils.data.Dataset):
    def __init__(
        self, queries, documents, shuffle_documents=True, max_split_token_length=1000
    ):
        assert len(queries) == len(documents)

        self.shuffle_documents = shuffle_documents
        assert len(queries) == len(documents)
        self.len = len(documents)
        self.queries = [" ".join(q.split()[:max_split_token_length]) for q in queries]
        self.documents = [
            " ".join(d.split()[:max_split_token_length]) for d in documents
        ]

    @classmethod
    def maybe_init_numericalizer(
        cls,
        texts,
        numericalizer: NBOWNumericalizer,
        tokenizer: Tokenizer,
        min_freq: int,
    ):
        if numericalizer is None:
            return NBOWNumericalizer.build_from_texts(
                texts, tokenizer, min_freq=min_freq
            )
        else:
            return numericalizer

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return self.queries[i], self.documents[i]

    def get_pair_data_loader(
        self,
        collate_fn,
        batch_size: int = 128,
        shuffle: bool = True,
        shuffle_tokens: bool = False,
    ):
        return torch.utils.data.DataLoader(
            self,
            collate_fn=collate_fn,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    @classmethod
    def prepare_from_dataframe(
        cls,
        df,
        query_cols,
        doc_col,
        document_tokenize=python_tokens.tokenize_python_code,
        query_numericalizer=None,
        document_numericalizer=None,
        min_query_token_freq=1,
        min_document_token_freq=5,
    ):
        queries = pd.concat([df[query_col] for query_col in query_cols]).to_list()
        docs = pd.concat([df[doc_col]] * len(query_cols)).to_list()
        return QueryDocumentDataset(
            queries,
            docs,
        )
