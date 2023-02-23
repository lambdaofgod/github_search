import io
import json
import logging
import re
import string
import tokenize
from operator import itemgetter

import dask
import dask.dataframe as ddf
import nltk
import numpy as np
import pandas as pd
import rank_bm25
import stop_words
import tqdm
from haystack import document_stores
from haystack.nodes import retriever as haystack_retriever

from github_search import paperswithcode_tasks
from github_search.ir import ir_utils
from github_search.python_call_graph import try_run


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


def extract_python_tokens(
    product,
    python_files_path="data/all_crawled_python_files.feather",
):
    tqdm.tqdm.pandas()
    logging.info("loading files")
    python_files_df = pd.read_feather(python_files_path)
    big_files_repos = python_files_df["repo_name"].value_counts().iloc[:10].index

    python_files_df = python_files_df[
        ~python_files_df["repo_name"].isin(big_files_repos)
    ].reset_index(drop=True)
    logging.info("extracting tokens")
    files_tokens = python_files_df["content"].progress_apply(
        get_joined_parsed_python_tokens
    )
    python_files_df["tokens_space_sep"] = files_tokens
    python_files_df.drop(columns=["content"]).to_feather(str(product))


def parse_camelcase(token):
    return re.split("(?<=[a-z])(?=[A-Z])", token)


def parse_snakecase(token):
    return token.split("_")


def parse_token(token):
    token_camelcase_results = parse_camelcase(token)
    if token.startswith("#"):
        return (
            token.split()
        )  # python source tokenizer treats comments as single strings
    elif len(token_camelcase_results) > 1:
        return token_camelcase_results
    else:
        return parse_snakecase(token)


def get_python_tokens_naively(contents):
    return nltk.tokenize.wordpunct_tokenize(contents)


def get_python_tokens(contents):
    try:
        buf = io.BytesIO(contents.encode("utf-8"))
        return [tinfo.string for tinfo in tokenize.tokenize(buf.readline)]
    except (tokenize.TokenError, IndentationError):
        return []


def get_parsed_python_tokens(contents):
    return [
        part
        for tok in get_python_tokens_naively(contents)
        for part in parse_token(tok.strip(string.punctuation))
        if part.strip() != ""
    ]


def get_joined_parsed_python_tokens(contents):
    try:
        return " ".join(get_parsed_python_tokens(contents))
    except TypeError:
        return ""


def make_query_results_list(searcher, queries, queries_ids, topn=10):
    return [
        [
            {"corpus_id": str(corpus_id), "score": round(score, 3)}
            for corpus_id, score in zip(*searcher.search(query, topn=topn))
        ]
        for (query, query_id) in zip(queries, queries_ids)
    ]


def prepare_bow_retrieval_evaluation_results(upstream, index, product):
    logging.getLogger().setLevel(logging.ERROR)
    document_store = document_stores.ElasticsearchDocumentStore(index=index)
    retriever = haystack_retriever.sparse.ElasticsearchRetriever(document_store)
    test_tasks = pd.read_csv(
        str(upstream["prepare_task_train_test_split"]["test"])
    ).iloc[:, 0]
    metric_results_df = pd.DataFrame(
        [
            {
                "query": query,
                **ir_utils.evaluate_query_results(retriever, query, [0.5], topk=10),
            }
            for query in tqdm.tqdm(test_tasks.dropna())
        ]
    )
    metric_results_df.to_csv(str(product), index=False)
