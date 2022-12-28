from dataclasses import dataclass, field
from typing import List, Callable, Optional, Union

import pickle
import tqdm
from collections import Counter
import sentence_transformers
import fasttext
import pandas as pd
import numpy as np
import torch.utils
from mlutil import sentence_transformers_utils
from findkit import feature_extractor, index
from github_search.ir import InformationRetrievalColumnConfig, models
from github_search import python_tokens, utils
from github_search.neural_bag_of_words import utils as nbow_utils, layers, embedders
from github_search.neural_bag_of_words.data import (
    NBOWNumericalizer,
    QueryDocumentDataset,
)
from mlutil.text import code_tokenization
from nltk import tokenize


@dataclass
class TrainValColumnConfig:
    train_query_cols: List[str]
    val_query_col: str
    doc_cols: List[str]
    list_cols: List[str]
    doc_col = "document"

    def get_information_retrieval_column_config(self):
        return InformationRetrievalColumnConfig(
            self.doc_cols, self.val_query_col, self.list_cols
        )


@dataclass
class NBOWTrainValData:

    train_dset: QueryDocumentDataset
    val_dset: QueryDocumentDataset
    _train_df: pd.DataFrame
    val_df: pd.DataFrame
    column_config: TrainValColumnConfig

    @classmethod
    def _prepare_df(cls, df, config):
        return utils.concatenate_flattened_list_cols(
            df,
            concat_cols=config.doc_cols,
            target_col=config.doc_col,
            str_list_cols=config.list_cols + [config.val_query_col],
        )

    @classmethod
    def build(
        cls,
        query_corpus,
        train_df,
        val_df,
        config,
        document_tokenizer=code_tokenization.tokenize_python_code,
    ):

        query_num = NBOWNumericalizer.build_from_texts(
            query_corpus.dropna(), tokenizer=tokenize.wordpunct_tokenize
        )

        train_df = cls._prepare_df(train_df, config)
        val_df = cls._prepare_df(val_df, config)
        document_num = NBOWNumericalizer.build_from_texts(
            train_df[config.doc_col],
            tokenizer=code_tokenization.tokenize_python_code,
        )

        train_dset = QueryDocumentDataset.prepare_from_dataframe(
            train_df,
            config.train_query_cols,
            config.doc_col,
            query_numericalizer=query_num,
            document_numericalizer=document_num,
        )
        val_dset = QueryDocumentDataset(
            val_df[config.val_query_col].to_list(),
            val_df[config.doc_col].to_list(),
        )
        return NBOWTrainValData(train_dset, val_dset, train_df, val_df, config)

    def get_train_dl(self, batch_size=128, shuffle=True, shuffle_tokens=True, **kwargs):
        return self.train_dset.get_pair_data_loader(
            shuffle=shuffle,
            batch_size=batch_size,
            shuffle_tokens=shuffle_tokens,
            **kwargs
        )

    def get_val_dl(self, batch_size=256, shuffle=False, **kwargs):
        return self.val_dset.get_pair_data_loader(
            shuffle=shuffle, batch_size=batch_size, shuffle_tokens=False
        )

    def get_document_padding_idx(self):
        return self.train_dset.document_numericalizer.get_padding_idx()

    def get_query_padding_idx(self):
        return self.train_dset.query_numericalizer.get_padding_idx()

@dataclass
class TokenizerWithWeights:

    tokenizer: sentence_transformers_utils.CustomTokenizer
    frequency_weights: np.ndarray
    max_seq_length: int

    @classmethod
    def make_from_data(
        cls,
        tokenize_fn: Callable[[str], List[str]],
        min_freq: int,
        data: List[str],
        max_seq_length: int,
    ):
        token_counter = cls.get_token_counter(
            data, tokenize_fn, max_seq_length
        )
        vocab = cls.get_vocab_from_counter(token_counter, min_freq)
        tokenizer = cls.get_tokenizer(vocab, tokenize_fn)
        frequency_weights = cls.get_frequency_weights(token_counter, vocab)
        return TokenizerWithWeights(
            tokenizer=tokenizer,
            frequency_weights=frequency_weights,
            max_seq_length=max_seq_length,
        )

    @classmethod
    def get_token_counter(cls, data, tokenize_fn, max_seq_length):
        all_tokens = (
            tok.lower()
            for doc in tqdm.auto.tqdm(data)
            for tok in tokenize_fn(doc)[:max_seq_length]
        )
        return Counter(all_tokens)

    @classmethod
    def get_vocab_from_counter(cls, counter, min_freq):
        return [item for (item, cnt) in counter.items() if cnt >= min_freq]

    @classmethod
    def get_tokenizer(cls, vocab, tokenize_fn):
        return sentence_transformers_utils.CustomTokenizer(
            vocab=vocab, tokenize_fn=tokenize_fn, do_lower_case=True
        )

    @classmethod
    def get_frequency_weights(cls, counter, vocab):
        freqs = np.array([counter.get(t, 0) for t in vocab])
        return freqs

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

