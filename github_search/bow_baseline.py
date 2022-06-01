import io
import tokenize

import nltk

import json
import re
import string
import tokenize
from operator import itemgetter

import dask
import dask.dataframe as ddf
import numpy as np
import pandas as pd
import rank_bm25
import stop_words
import tqdm
from dask.diagnostics import ProgressBar as DaskProgressBar

from github_search import paperswithcode_tasks
from github_search.python_call_graph import try_run
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

papers_with_repo_df = paperswithcode_tasks.get_papers_with_biggest_tasks_df(10000)


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
    product, python_files_path="data/all_crawled_python_files.feather"
):
    logging.info("loading files")
    python_files_df = pd.read_feather(python_files_path)
    big_files_repos = python_files_df["repo_name"].value_counts().iloc[:10].index

    python_files_df = python_files_df[
        ~python_files_df["repo_name"].isin(big_files_repos)
    ]
    logging.info("extracting tokens")
    files_tokens = run_dask_series_apply(
        get_parsed_python_tokens, python_files_df["content"]
    )
    python_files_df["tokens_space_sep"] = files_tokens
    python_files_df["tokens_space_sep"] = python_files_df["tokens_space_sep"].apply(
        " ".join
    )
    python_files_df.drop(columns=["content"]).reset_index(drop=True).to_feather(
        str(product)
    )


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


def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.split():
        token = token.strip(string.punctuation)
        token_parts = parse_token(token)
        for token_part in token_parts:
            if len(token_part) > 0 and token_part not in stop_words.get_stop_words(
                "english"
            ):
                tokenized_doc.append(token_part)
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


def run_dask_series_apply(function, values, chunksize=100):
    ddf_values = ddf.from_array(values, chunksize=chunksize)
    with dask.config.set(scheduler="threads", n_workers=10):
        with DaskProgressBar():
            computed_values = ddf_values.apply(function).compute()
    return computed_values


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


class RankBM25AggregateSearcher:
    def __init__(self, df, corpus):
        self.bm25 = rank_bm25.BM25Okapi(corpus)
        self.df = df.reset_index(drop=True)
        self.corpus = corpus

    def show_query_results(self, query, topn=10, verbose=False):
        query = query.lower().split()
        scores = self.bm25.get_scores(query)
        top_document_indices = (-scores).argsort()[:10]
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

    def search_aggregated(self, query, topn=10):
        indices, scores = self.show_query_results(query, topn * 5)
        return (
            self.get_aggregate_scores(self.df, indices, scores)
            .sort_values(ascending=False)
            .iloc[:topn]
        )

    def search(self, query, topn=10):
        indices, scores = self.show_query_results(query, topn)
        return indices, scores
        # return pd.Series(data=scores, index=indices[0])


def make_query_results_list(searcher, queries, queries_ids, topn=10):
    return [
        [
            {"corpus_id": str(corpus_id), "score": round(score, 3)}
            for corpus_id, score in zip(*searcher.search(query, topn=topn))
        ]
        for (query, query_id) in zip(queries, queries_ids)
    ]


from github_search.ir_utils import get_ir_evaluator


def get_bm25_evaluation_results(papers_with_repo_df, files_df, files_tokens):
    searcher = RankBM25AggregateSearcher(files_df, files_tokens)
    selected_files_df = files_df.merge(
        papers_with_repo_df[["repo", "tasks"]].drop_duplicates(subset=["repo"]),
        left_on="repo_name",
        right_on="repo",
    ).drop(columns="repo_name")
    evaluator = get_ir_evaluator(
        selected_files_df.reset_index(drop=True), doc_col="tokens_space_sep"
    )
    return evaluator.compute_metrics(
        make_query_results_list(searcher, evaluator.queries, evaluator.queries_ids)
    )

def prepare_bm25_evaluation_results(upstream, product):
    files_with_tokens_df = pd.read_feather(upstream['extract_python_tokens'])
    files_tokens = files_with_tokens_df['tokens_space_sep'].str.split()
    results = get_bm25_evaluation_results(papers_with_repo_df, files_with_tokens_df, files_tokens)
    with open(str(product), "w") as f:
        json.dump(results, f)
    #
# searcher = RankBM25AggregateSearcher(selected_files_df, files_tokens)
# searcher.df.iloc[searcher.search("metric learning")[0]]
# searcher.search("metric learning")[1]
# searcher.search_aggregated("reinforcement learning")
#
# # evaluator.relevant_docs
# papers_with_repo_df.iloc[:25][["repo", "tasks"]]
# searcher.search("contour detection")
# valid_readme_files_df = readme_files_df.dropna(subset=["content"])
# valid_readme_files_df = valid_readme_files_df[
#     valid_readme_files_df["content"].str.split().apply(len) > 0
# ]
# readme_searcher = RankBM25AggregateSearcher(
#     valid_readme_files_df, valid_readme_files_df["content"].str.split()
# )
# valid_readme_files_df["content"].str.split()
# valid_readme_files_df[
#     valid_readme_files_df["repo_name"] == "IGITUGraz/spore-nest-module"
# ]
# readme_searcher.search("segmentation")
# from github_search import information_retrieval_baseline
#
# task_grouped_repos = information_retrieval_baseline.get_task_grouped_rows(
#     papers_with_repo_df, col="repo"
# )
# task_grouped_repos.index = pd.Index(
#     pd.Series(task_grouped_repos.index).str.replace("-", " ")
# )
# task_grouped_repos
# searcher.search("semantic segmentation", 10)
# searcher.search("transfer learning", 10)
#
#
# def get_recalled_repos(searcher, task_grouped_repos):
#     return [
#         [
#             repo
#             for repo in searcher.search(task, topn=20).index
#             if repo in task_grouped_repos[task]
#         ]
#         for task in tqdm.tqdm(task_grouped_repos.index)
#     ]
#
#
# set(searcher.search("semantic segmentation").index).intersection(
#     set(task_grouped_repos["semantic segmentation"])
# )
# searcher.search("semantic segmentation")
# task_grouped_repos["semantic segmentation"]
# task_grouped_repos["semantic segmentation"]
# comment_recalled_repos = get_recalled_repos(searcher, task_grouped_repos)
# readme_recalled_repos = get_recalled_repos(readme_searcher, task_grouped_repos)
#
#
# def get_accuracies(recalled_repos, task_grouped_repos):
#     return [
#         len(set(repos).intersection(set(task_grouped_repos[task]))) / 20
#         for (repos, task) in tqdm.tqdm(zip(recalled_repos, task_grouped_repos.index))
#     ]
#
#
# "semantic segmentation" in task_grouped_repos.index
# np.mean(get_accuracies(readme_recalled_repos, task_grouped_repos))
# np.mean(get_accuracies(comment_recalled_repos, task_grouped_repos))
# np.mean(accuracies)
# get_ipython().run_cell_magic(
#     "time", "", "searcher.search(task_grouped_repos.index[4])\n"
# )
# task
