import numpy as np
import pandas as pd
from github_search import python_tokens, utils
from github_search.neural_bag_of_words.data import prepare_dependency_texts


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
    return dependency_records_df.groupby("repo")["destination"].agg(" ".join).reset_index()


def get_dependency_nbow_dataset(paperswithcode_df, df_dependency_corpus):
    dep_texts_with_tasks_df = (
        paperswithcode_df[["repo", "tasks"]]
        .merge(df_dependency_corpus, on="repo")
        .dropna()
    )
    return dep_texts_with_tasks_df


def prepare_dependency_data_corpus(upstream, product):
    dependency_records_path = str(upstream["dependency_graph.prepare_records"])
    dependency_records_df = pd.read_feather(dependency_records_path)
    dep_texts = get_dependency_texts(dependency_records_df)
    dep_texts = dep_texts.dropna().rename({"destination": "dependencies"}, axis=1)
    pd.DataFrame(dep_texts).reset_index().to_parquet(product["text"])
    pd.DataFrame(dep_texts).reset_index(drop=True).to_csv(
        product["raw_text"], index=False, header=False
    )


def prepare_nbow_dataset(upstream, product):
    df_dependency_corpus = pd.read_parquet(
        str(upstream["nbow.prepare_dependency_data_corpus"]["text"])
    )
    for split_name in ["train", "test"]:
        df_paperswithcode = pd.read_csv(
            str(upstream["prepare_repo_train_test_split"][split_name])
        )
        df_corpus = get_dependency_nbow_dataset(df_paperswithcode, df_dependency_corpus)
        df_corpus.to_parquet(str(product[split_name]))
