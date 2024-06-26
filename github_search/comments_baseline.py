# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/Comment_Baseline.ipynb (unless otherwise specified).

__all__ = [
    "get_comment_contents",
    "bm25_tokenizer",
    "get_tokenized_comments",
    "get_valid_comments_and_files",
    "run_dask_series_apply",
    "RankBM25AggregateSearcher",
]

# Cell

import ast
import io
import itertools
import os
import pickle
import re
import string
import tokenize
from operator import itemgetter

import astunparse
import csrgraph
import csrgraph as cg
import dask.dataframe as ddf
import docstring_parser
import gensim
import igraph
import mlutil
import nodevectors
import numpy as np
import pandas as pd
import rank_bm25
import stop_words
import toolz
import tqdm
from dask.diagnostics import ProgressBar as DaskProgressBar
from mlutil import prototype_selection
from mlutil.feature_extraction import embeddings
from sklearn import metrics

from github_search import paperswithcode_tasks, python_tokens

from .python_call_graph import try_run

# Cell


def get_comment_contents(file_contents):
    docstrings = re.findall('""".*?"""', file_contents.replace("\n", ""))
    docstring_contents = [d.replace('"""', "") for d in docstrings]
    comments = [
        tok.replace("#", "")
        for (toktype, tok, _, _, _) in tokenize.generate_tokens(
            io.StringIO(file_contents).readline
        )
        if toktype == tokenize.COMMENT
    ]
    return "\n".join(comments + docstring_contents)


# Cell


def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in stop_words.get_stop_words("english"):
            tokenized_doc.append(token)
    return tokenized_doc


def get_tokenized_comments(comments):
    tokenized_comments = []
    for passage in tqdm.tqdm(comments.values):
        tokenized_comments.append(bm25_tokenizer(passage))
    return tokenized_comments


def get_valid_comments_and_files(python_files_df, comment_contents):
    is_comment_valid = ~comment_contents.isna()
    nonempty_file_repos = python_files_df[is_comment_valid]
    tokenized_comments = run_dask_series_apply(
        comment_contents.dropna(), bm25_tokenizer
    )
    valid_comments = tokenized_comments[tokenized_comments.apply(len) > 0]
    valid_comment_files_df = nonempty_file_repos[tokenized_comments.apply(len) > 0]
    return valid_comment_files_df, valid_comments


def run_dask_series_apply(values, function, chunksize=10000):
    ddf_values = ddf.from_array(values, chunksize=chunksize)
    with DaskProgressBar():
        computed_values = ddf_values.apply(function).compute()
    return computed_values


# Cell


class RankBM25AggregateSearcher:
    def __init__(self, df, corpus):
        self.bm25 = rank_bm25.BM25Okapi(corpus)
        self.df = df
        self.corpus = corpus

    def show_query_results(self, query, topn=10, verbose=False):
        scores = self.bm25.get_scores(query)
        top_document_indices = ((-scores).argsort()[:10],)
        top_scores = scores[top_document_indices]
        top_documents = list(np.array(self.corpus)[top_document_indices])
        if verbose:
            for doc in top_documents:
                res = ""
                for term in tokenized_query:
                    if term in doc:
                        res += term + " in doc "
                print(res)
        return top_document_indices, top_scores

    def get_aggregate_scores(
        self, df, indices, top_scores, aggregated_column="repo_name"
    ):
        result_df = df.iloc[indices].copy()
        result_df["score"] = top_scores
        return result_df.groupby(aggregated_column)["score"].agg("mean")

    def search(self, query, topn=10):
        tokenized_query = query.split()
        indices, scores = self.show_query_results(tokenized_query, topn)
        return self.get_aggregate_scores(self.df, indices, scores)
