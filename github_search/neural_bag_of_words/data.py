from dataclasses import dataclass
from functools import partial
from typing import Callable, List

import numpy as np
import torch
import torch.utils
from github_search import python_tokens
import torchtext

Tokenizer = Callable[[List[str]], List[List[str]]]


def prepare_dependency_texts(dependency_records_df):
    dependency_records_df = dependency_records_df[
        dependency_records_df["source"] != "<ROOT>"
    ]
    return dependency_records_df.groupby("repo")["destination"].agg(" ".join)


def tokenize_python_code(code_text):
    """tokenize each word in code_text as python token"""
    toks = code_text.split()
    return [
        tok
        for raw_tok in code_text.replace("/", " ").replace("-", "_").split()
        for tok in python_tokens.tokenize_python(raw_tok)
    ]


def split_tokenizer(s):
    return s.split()


def pack_sequences_as_tensors(sequences, padding_value):
    return torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(s) for s in sequences],
        batch_first=True,
        padding_value=padding_value,
    )


def collate_fn(batch, padding_value_query, padding_value_document, device="cuda"):
    queries, documents = zip(*batch)
    return pack_sequences_as_tensors(queries, padding_value_query).to(
        device
    ), pack_sequences_as_tensors(documents, padding_value_document).to(device)


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
        self,
        queries,
        documents,
        document_tokenize=python_tokens.tokenize_python_code,
        query_numericalizer=None,
        document_numericalizer=None,
        min_query_token_freq=1,
        min_document_token_freq=5,
    ):
        assert len(queries) == len(documents)

        self.query_numericalizer = self.maybe_init_numericalizer(
            queries,
            query_numericalizer,
            tokenizer=split_tokenizer,
            min_freq=min_query_token_freq,
        )
        self.document_numericalizer = self.maybe_init_numericalizer(
            documents,
            document_numericalizer,
            tokenizer=document_tokenize,
            min_freq=min_document_token_freq,
        )
        self.numericalized_queries = self.query_numericalizer.numericalize_texts(
            queries
        )
        self.numericalized_documents = self.document_numericalizer.numericalize_texts(
            documents
        )

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
        return len(self.numericalized_documents)

    def __getitem__(self, i):
        return self.numericalized_queries[i], self.numericalized_documents[i]

    def get_query_vocab(self):
        return self.query_numericalizer.vocab

    def get_document_vocab(self):
        return self.document_numericalizer.vocab

    def get_document_frequency_weights(self, numericalized_texts, vocab):
        counts = np.zeros(len(vocab.vocab.itos_), dtype=np.int32)
        for nums in numericalized_texts:
            counts[nums] += 1
        return counts

    def get_pair_data_loader(self, batch_size: int = 64, shuffle: bool = True):
        return torch.utils.data.DataLoader(
            self,
            collate_fn=partial(
                collate_fn,
                padding_value_query=self.query_numericalizer.get_padding_idx(),
                padding_value_document=self.document_numericalizer.get_padding_idx(),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )