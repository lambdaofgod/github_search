import ast
import logging

import numpy as np
import pandas as pd
from github_search import github_readmes, python_tokens, utils
from github_search.neural_bag_of_words import embedders, tokenization
from github_search.neural_bag_of_words.data import prepare_dependency_texts
from github_search.python_code import signatures
from mlutil.text import code_tokenization
from nltk import tokenize


def tokenize_python_code(code_text):
    """tokenize each word in code_text as python token"""
    toks = code_text.split()
    return [
        tok
        for raw_tok in code_text.split()
        for tok in python_tokens.tokenize_python(raw_tok)
    ]


def get_dependency_texts(dependency_records_df):
    dependency_records_df = dependency_records_df[
        dependency_records_df["source"] != "<ROOT>"
    ]
    return (
        dependency_records_df.groupby("repo")["destination"].agg(" ".join).reset_index()
    )


def truncate_readme(readme, n_lines):
    return " ".join([l for l in readme.split("\n") if l != ""][:n_lines])


def truncate_and_impute_readmes(readmes, imputing_col, n_lines):
    return readmes.fillna(imputing_col).apply(lambda r: truncate_readme(r, n_lines))


def add_imputed_readmes(df, n_readme_lines):
    df["readme"] = truncate_and_impute_readmes(
        df["readme"],
        df["repo"],
        n_readme_lines,
    )
    return df


def add_repo_name_to_dependencies(df):
    df["dependencies"] = df["repo"] + " " + df["dependencies"]
    return df


def get_nbow_dataset(
    paperswithcode_df,
    df_dependency_corpus,
    df_signatures_corpus,
    additional_columns,
    n_readme_lines,
):
    additional_paperswithcode_columns = [
        col for col in additional_columns if col in paperswithcode_df
    ]
    df_signatures_corpus = df_signatures_corpus.rename({"repo_name": "repo"}, axis=1)
    df_nbow_data = (
        paperswithcode_df[["repo", "tasks"] + additional_paperswithcode_columns]
        .merge(df_dependency_corpus, on="repo")
        .merge(df_signatures_corpus, on="repo")
    )
    if "readme" in additional_columns:
        dep_texts_with_tasks_df = add_imputed_readmes(df_nbow_data, n_readme_lines)
    if "dependencies" in additional_columns:
        add_repo_name_to_dependencies(df_nbow_data)
    return dep_texts_with_tasks_df
