import ast
import numpy as np
import pandas as pd
from github_search import python_tokens, utils, github_readmes
from github_search.neural_bag_of_words.data import prepare_dependency_texts
from github_search.neural_bag_of_words import embedders, tokenization

from mlutil.text import code_tokenization
from nltk import tokenize
import logging


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


def get_dependency_nbow_dataset(
    paperswithcode_df, df_dependency_corpus, additional_columns, n_readme_lines
):
    additional_paperswithcode_columns = [
        col for col in additional_columns if col in paperswithcode_df
    ]
    dep_texts_with_tasks_df = paperswithcode_df[
        ["repo", "tasks"] + additional_paperswithcode_columns
    ].merge(df_dependency_corpus, on="repo")
    if "readme" in additional_columns:
        dep_texts_with_tasks_df["readme"] = truncate_and_impute_readmes(
            dep_texts_with_tasks_df["readme"],
            dep_texts_with_tasks_df["repo"],
            n_readme_lines,
        )
    return dep_texts_with_tasks_df


def prepare_raw_dependency_data_corpus(upstream, product):
    dependency_records_path = str(upstream["dependency_graph.prepare_records"])
    dependency_records_df = pd.read_feather(dependency_records_path)
    dep_texts = get_dependency_texts(dependency_records_df)
    dep_texts = dep_texts.dropna().rename({"destination": "dependencies"}, axis=1)
    pd.DataFrame(dep_texts).reset_index().to_parquet(product["text"])
    pd.DataFrame(dep_texts).reset_index(drop=True).to_csv(
        product["raw_text"], index=False, header=False
    )


def prepare_readmes(upstream, product, max_workers):
    dep_texts_df = pd.read_parquet(
        str(upstream["nbow.prepare_raw_dependency_data_corpus"]["text"])
    )
    readmes = github_readmes.get_readmes(dep_texts_df, max_workers)
    dep_texts_df["readme"] = readmes
    dep_texts_df[["repo", "readme"]].to_csv(product)


def prepare_dependency_data_corpus(upstream, product):
    dep_texts = pd.read_csv(
        str(upstream["nbow.prepare_raw_dependency_data_corpus"]["raw_text"])
    )
    dep_texts_df = pd.read_parquet(
        str(upstream["nbow.prepare_raw_dependency_data_corpus"]["text"])
    )
    dep_texts_df["readme"] = pd.read_csv(str(upstream["nbow.prepare_readmes"]))[
        "readme"
    ]
    dep_texts_df.to_csv(product["text"])
    dep_texts.to_csv(product["raw_text"])


def prepare_nbow_dataset(upstream, product, additional_columns, n_readme_lines):
    df_dependency_corpus = pd.read_csv(
        str(upstream["nbow.prepare_dependency_data_corpus"]["text"])
    )
    for split_name in ["train", "test"]:
        df_paperswithcode = pd.read_csv(
            str(upstream["prepare_repo_train_test_split"][split_name])
        )
        df_corpus = get_dependency_nbow_dataset(
            df_paperswithcode, df_dependency_corpus, additional_columns, n_readme_lines
        )
        df_corpus.to_parquet(str(product[split_name]))


def prepare_tokenizers(upstream, product, min_freq, max_seq_length):
    logging.basicConfig(level="INFO")
    df_corpus = pd.read_parquet(str(upstream["nbow.prepare_dataset"]["train"]))
    titles = df_corpus["titles"].apply(lambda ts: " ".join(ast.literal_eval(ts)))
    tasks = df_corpus["tasks"].apply(lambda ts: " ".join(ast.literal_eval(ts)))
    query_corpus = pd.concat([titles, tasks])
    document_corpus = pd.concat([df_corpus["dependencies"], titles])
    logging.info("preparing query tokenizer")
    query_tokenizer = tokenization.TokenizerWithWeights.make_from_data(
        tokenize_fn=tokenize.wordpunct_tokenize,
        min_freq=min_freq,
        max_seq_length=max_seq_length,
        data=query_corpus,
    )
    logging.info("preparing document tokenizer")
    document_tokenizer = tokenization.TokenizerWithWeights.make_from_data(
        tokenize_fn=code_tokenization.tokenize_python_code,
        min_freq=min_freq,
        max_seq_length=max_seq_length,
        data=document_corpus,
    )
    query_tokenizer.save(str(product["query_tokenizer"]))
    document_tokenizer.save(str(product["document_tokenizer"]))
