from dataclasses import dataclass, field
from typing import List, Callable

import pandas as pd
import numpy as np
import torch.utils
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
            query_numericalizer=query_num,
            document_numericalizer=document_num,
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



@utils.kwargs_only
@dataclass
class NBOWPair:
    query_nbow: layers.NBOWLayer
    document_nbow: layers.NBOWLayer


@dataclass
class EmbedderConfigurator:

    encoding_fn: Callable[[List[str]], np.ndarray]
    train_val_data: NBOWTrainValData
    max_seq_length: int

    def get_embedder_pair(self):
        nbow_configurator = NBOWLayerConfigurator(self.encoding_fn, self.train_val_data)
        nbow_pair = nbow_configurator.get_nbow_pair()
        dset = self.train_val_data.train_dset
        query_embedder = embedders.make_sentence_transformer_nbow_model(
            nbow_pair.query_nbow,
            vocab=dset.query_numericalizer.vocab.vocab.itos_,
            tokenize_fn=dset.query_numericalizer.tokenizer,
            token_weights=nbow_pair.query_nbow.token_weights.cpu().numpy().tolist(),
            max_seq_length=self.max_seq_length,
        )
        document_embedder = embedders.make_sentence_transformer_nbow_model(
            nbow_pair.document_nbow,
            vocab=dset.document_numericalizer.vocab.vocab.itos_,
            tokenize_fn=dset.document_numericalizer.tokenizer,
            token_weights=nbow_pair.document_nbow.token_weights.cpu().numpy().tolist(),
            max_seq_length=self.max_seq_length,
        )
        return models.EmbedderPair(
            query_embedder=query_embedder, document_embedder=document_embedder
        )


@dataclass
class NBOWLayerConfigurator:

    encoding_fn: Callable[[List[str]], np.ndarray]
    train_val_data: NBOWTrainValData

    def get_nbow_pair(self):
        return NBOWPair(
            query_nbow=self.get_query_nbow_layer(),
            document_nbow=self.get_document_nbow_layer(),
        )

    def get_document_nbow_layer(self):
        return layers.NBOWLayer.make_from_encoding_fn(
            vocab=self.train_val_data.train_dset.get_document_vocab(),
            df_weights=self.train_val_data.train_dset.get_document_frequency_weights(
                self.train_val_data.train_dset.numericalized_documents,
                self.train_val_data.train_dset.get_document_vocab(),
            ),
            encoding_fn=self.encoding_fn,
        )

    def get_query_nbow_layer(self):
        return layers.NBOWLayer.make_from_encoding_fn(
            vocab=self.train_val_data.train_dset.get_query_vocab(),
            df_weights=self.train_val_data.train_dset.get_document_frequency_weights(
                self.train_val_data.train_dset.numericalized_queries,
                self.train_val_data.train_dset.get_query_vocab(),
            ),
            encoding_fn=self.encoding_fn,
        )
