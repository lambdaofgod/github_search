from dataclasses import dataclass, field, asdict
from typing import List, Callable, Optional, Union, Dict

import yaml
import os
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
class TrainValConfig:
    loss_function_name: str
    document_cols: List[str]
    query_embedder: Optional[str]
    max_seq_length: int
    query_cols: List[str]
    val_query_col: str

    @classmethod
    def load(cls, name, conf_path="conf"):
        path = os.path.join(conf_path, name)
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return TrainValConfig(**config)

    def get_information_retrieval_column_config(self):
        return InformationRetrievalColumnConfig(
            self.document_cols, self.val_query_col, self.get_list_cols()
        )

    def get_list_cols(self):
        return ["titles"] if "titles" in self.document_cols else []

    def save(self, name, conf_path="conf"):
        path = os.path.join(conf_path, name)
        with open(path, "r") as f:
            yaml.dump(f, asdict(self))


@dataclass
class QueryDocumentCollator:

    query_tokenize_fn: Callable[[List[str]], Dict[str, torch.Tensor]]
    document_tokenize_fn: Callable[[List[str]], Dict[str, torch.Tensor]]

    @classmethod
    def from_embedder_pair(cls, embedder_pair):
        return QueryDocumentCollator(
            query_tokenize_fn=embedder_pair.query_embedder.tokenize,
            document_tokenize_fn=embedder_pair.document_embedder.tokenize,
        )

    def collate_fn(self, batch):
        queries, documents = zip(*batch)
        return self.query_tokenize_fn(queries), self.document_tokenize_fn(documents)


@dataclass
class NBOWTrainValData:

    train_dset: QueryDocumentDataset
    val_dset: QueryDocumentDataset
    _train_df: pd.DataFrame
    val_df: pd.DataFrame
    column_config: TrainValConfig

    @classmethod
    def _prepare_df(cls, df, config):
        return utils.concatenate_flattened_list_cols(
            df,
            concat_cols=config.document_cols,
            target_col=cls.get_doc_col(),
            str_list_cols=config.get_list_cols() + [config.val_query_col],
        )

    @classmethod
    def get_doc_col(cls):
        return "document"

    @classmethod
    def build(
        cls,
        query_corpus,
        train_df,
        val_df,
        config,
    ):

        query_num = NBOWNumericalizer.build_from_texts(
            query_corpus.dropna(), tokenizer=tokenize.wordpunct_tokenize
        )

        train_df = cls._prepare_df(train_df, config)
        val_df = cls._prepare_df(val_df, config)
        document_num = NBOWNumericalizer.build_from_texts(
            train_df[cls.get_doc_col()],
            tokenizer=code_tokenization.tokenize_python_code,
        )

        train_dset = QueryDocumentDataset.prepare_from_dataframe(
            train_df,
            config.query_cols,
            cls.get_doc_col(),
            query_numericalizer=query_num,
            document_numericalizer=document_num,
        )
        val_dset = QueryDocumentDataset(
            val_df[config.val_query_col].to_list(),
            val_df[cls.get_doc_col()].to_list(),
        )
        return NBOWTrainValData(train_dset, val_dset, train_df, val_df, config)

    def get_train_dl(
        self,
        collator: QueryDocumentCollator,
        batch_size=128,
        shuffle=True,
        shuffle_tokens=True,
        **kwargs
    ):
        return self.train_dset.get_pair_data_loader(
            collate_fn=collator.collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            shuffle_tokens=shuffle_tokens,
            **kwargs
        )

    def get_val_dl(self, collator, batch_size=256, shuffle=False, **kwargs):
        return self.val_dset.get_pair_data_loader(
            collate_fn=collator.collate_fn,
            shuffle=shuffle,
            batch_size=batch_size,
            shuffle_tokens=False,
        )

    def get_document_padding_idx(self):
        return self.train_dset.document_numericalizer.get_padding_idx()

    def get_query_padding_idx(self):
        return self.train_dset.query_numericalizer.get_padding_idx()
