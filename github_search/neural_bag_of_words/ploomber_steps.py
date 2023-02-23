import ast
import logging

import numpy as np
import pandas as pd
from github_search import github_readmes, python_tokens, utils
from github_search.neural_bag_of_words import embedders, tokenization
from github_search.neural_bag_of_words.data import prepare_dependency_texts
from github_search.neural_bag_of_words.prepare_data import *
from github_search.python_code import signatures
from mlutil.text import code_tokenization
from nltk import tokenize
from mlutil_rust import code_tokenization


NATURAL_LANGUAGE_COLS = ["titles", "readme"]

DEPENDENCY_RECORDS_PATH = "output/dependency_records.feather"
SIGNATURES_PATH = 'output/python_signatures.parquet'
TRAIN_TEST_SPLIT_PATHS = {'test': 'output/repos_test.csv', 'train': 'output/repos_train.csv'}


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


def prepare_rarest_signatures_corpus(product, n_rarest, upstream):
    rarest_signatures_corpus_pldf = (
        signatures.SignatureSelector.prepare_rarest_signatures_corpus_pldf(
            str(upstream["prepare_function_signatures_df"]), n_rarest
        )
    )
    rarest_signatures_corpus_pldf.write_parquet(str(product["texts"]))
    rarest_signatures_corpus_pldf.to_pandas()["function_signature"].to_csv(
        str(product["raw_text"])
    )


def prepare_nbow_dataset(upstream, product, additional_columns, n_readme_lines):
    df_dependency_corpus = pd.read_csv(
        str(upstream["nbow.prepare_dependency_data_corpus"]["text"])
    )
    df_signatures_corpus = pd.read_parquet(
        str(upstream["nbow.prepare_signature_corpus"]["texts"])
    )
    for split_name in ["train", "test"]:
        df_paperswithcode = pd.read_csv(
            str(upstream["prepare_repo_train_test_split"][split_name])
        )
        df_corpus = get_nbow_dataset(
            df_paperswithcode,
            df_dependency_corpus,
            df_signatures_corpus,
            additional_columns,
            n_readme_lines,
        )
        df_corpus.to_parquet(str(product[split_name]))


def prepare_tokenizers(upstream, product, document_col, min_freq, max_seq_length):
    logging.basicConfig(level="INFO")
    df_corpus = pd.read_parquet(str(upstream["nbow.prepare_dataset"]["train"]))
    titles = df_corpus["titles"].apply(lambda ts: " ".join(ast.literal_eval(ts)))
    tasks = df_corpus["tasks"].apply(lambda ts: " ".join(ast.literal_eval(ts)))
    query_corpus = pd.concat([titles, tasks])
    document_corpus = pd.concat([df_corpus[document_col], titles])
    logging.info("preparing query tokenizer")
    query_tokenizer = tokenization.TokenizerWithWeights.make_from_data(
        text_type="text",
        min_freq=min_freq,
        max_seq_length=max_seq_length,
        data=query_corpus,
    )
    logging.info("preparing document tokenizer")
    document_tokenizer = tokenization.TokenizerWithWeights.make_from_data(
        text_type="text" if document_col in NATURAL_LANGUAGE_COLS else "code",
        min_freq=min_freq,
        max_seq_length=max_seq_length,
        data=document_corpus,
    )
    query_tokenizer.save(str(product["query_tokenizer"]))
    document_tokenizer.save(str(product["document_tokenizer"]))
    
