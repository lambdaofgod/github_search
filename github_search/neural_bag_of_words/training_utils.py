from dataclasses import dataclass

import torch.utils
from findkit import feature_extractor, index
from github_search import python_tokens, utils
from github_search.neural_bag_of_words.data import *
from mlutil.text import code_tokenization
from nltk import tokenize


@dataclass
class NBOWTrainValData:

    train_dset: QueryDocumentDataset
    val_dset: QueryDocumentDataset

    @staticmethod
    def build(
        query_corpus,
        train_df,
        val_df,
        doc_col="dependencies",
        train_query_cols=["tasks", "titles"],
        val_query_col="tasks",
        document_tokenizer=code_tokenization.tokenize_python_code,
    ):

        query_num = NBOWNumericalizer.build_from_texts(
            query_corpus.dropna(), tokenizer=tokenize.wordpunct_tokenize
        )

        document_num = NBOWNumericalizer.build_from_texts(
            train_df[doc_col],
            tokenizer=code_tokenization.tokenize_python_code,
        )

        train_dset = QueryDocumentDataset.prepare_from_dataframe(
            train_df,
            train_query_cols,
            doc_col,
            query_numericalizer=query_num,
            document_numericalizer=document_num,
        )
        val_dset = QueryDocumentDataset(
            val_df[val_query_col].apply(" ".join).to_list(),
            val_df[doc_col].to_list(),
            query_numericalizer=query_num,
            document_numericalizer=document_num,
        )
        return NBOWTrainValData(train_dset, val_dset)

    def get_train_dl(self, batch_size=128, shuffle=True, shuffle_tokens=True, **kwargs):
        return self.train_dset.get_pair_data_loader(
            shuffle=shuffle, batch_size=batch_size, shuffle_tokens=shuffle_tokens, **kwargs
        )

    def get_val_dl(self, batch_size=256, shuffle=False, **kwargs):
        return self.val_dset.get_pair_data_loader(
            shuffle=shuffle, batch_size=batch_size, shuffle_tokens=False
        )
