__all__ = [
    "get_modules_string",
    "get_module_corpus",
    "prepare_module_corpus",
    "store_word_vectors",
]

import pickle
import json
import pandas as pd
import tqdm
import numpy as np
import gensim
import ast

from github_search import (
    parsing_imports,
    paperswithcode_tasks,
    python_call_graph,
    node_embedding_evaluation,
)
from sklearn import feature_extraction, decomposition
import fasttext

import csv

csv.field_size_limit(1000000)


def get_modules_string(modules):
    public_modules = [mod for mod in modules if not mod[0] == "_"]
    return " ".join(public_modules)


def get_module_corpus(files_df):
    module_lists = []
    repos = []

    for __, row in tqdm.tqdm(files_df.iterrows(), total=len(files_df)):
        try:
            maybe_imports = parsing_imports.get_modules(row["content"])
            module_lists.append(list(set(maybe_imports)))
            repos.append(row["repo_name"])
        except SyntaxError as e:
            print(row["repo_name"], e)
    df = pd.DataFrame({"repo": repos, "imports": module_lists})
    return df


def get_paperswithcode_with_imports_df(papers_with_repo_df, per_repo_imports):
    return papers_with_repo_df.merge(
        per_repo_imports, left_on="repo", right_index=True
    ).drop_duplicates("repo")


def prepare_paperswithcode_with_imports_df(product, upstream, python_file_paths):
    """
    add import data to python files csv
    """
    print("PYTHON FILE PATHS", python_file_paths)
    python_files_df = pd.concat([pd.read_csv(path, encoding="utf-8") for path in python_file_paths])
    print(python_files_df.shape)
    repo_names = python_files_df["repo_name"]
    paperswithcode_df, all_papers_df = paperswithcode_tasks.get_paperswithcode_dfs()
    papers_with_repo_df = paperswithcode_tasks.get_papers_with_repo_df(
        all_papers_df, paperswithcode_df, repo_names
    )
    papers_with_repo_df = paperswithcode_tasks.get_papers_with_biggest_tasks(
        papers_with_repo_df, 500
    )
    import_corpus_df = pd.read_csv(upstream["prepare_module_corpus"])
    import_corpus_df["imports"] = import_corpus_df["imports"].apply(ast.literal_eval)
    per_repo_imports = import_corpus_df.groupby("repo")["imports"].agg(sum).apply(set)
    paperswithcode_with_imports_df = get_paperswithcode_with_imports_df(
        papers_with_repo_df, per_repo_imports
    )
    paperswithcode_with_imports_df.to_csv(str(product))


def prepare_module_corpus(python_file_paths, product):
    """
    prepare csv file with modules for import2vec
    """
    python_files_df = pd.concat(
        [
            pd.read_csv(
                path,
                encoding="utf-8",
            )
            for path in python_file_paths
        ]
    ).dropna(subset=["repo_name", "content"])
    get_module_corpus(python_files_df).to_csv(str(product))


def prepare_dependency_records(python_file_paths, product):
    """
    prepare python dependency graph records (function calls, files in repo) csv
    """
    python_files_df = pd.concat(
        [
            pd.read_csv(
                path,
                encoding="utf-8"
            )
            for path in python_file_paths
        ]
    ).dropna(subset=["repo_name", "content"])
    repo_dependency_fetcher = python_call_graph.RepoDependencyGraphFetcher()
    sample_files_df = python_call_graph.get_sample_files_df(
        python_files_df, n_files=1000
    )
    repo_records_df = repo_dependency_fetcher.get_dependency_df(
        sample_files_df, "repo", clean_content=True
    )
    function_records_df = repo_dependency_fetcher.get_dependency_df(
        sample_files_df, "function", clean_content=True
    )
    dependency_records_df = pd.concat([repo_records_df, function_records_df])
    dependency_records_df["source"] = dependency_records_df["source"].apply(
        lambda s: s if type(s) is str else s.decode("utf-8")
    )
    dependency_records_df["destination"] = dependency_records_df["destination"].apply(
        lambda s: s if type(s) is str else s.decode("utf-8")
    )
    dependency_records_df.dropna().to_csv(str(product), index=False)


def train_python_token_fasttext(python_file_path, epoch, dim, product):
    python_files_df = pd.read_csv(python_file_path, encoding="utf-8")
    fasttext_corpus_path = "/tmp/python_files.csv"
    python_files_df["content"].dropna().to_csv(
        fasttext_corpus_path, index=False, header=False
    )
    model = fasttext.train_unsupervised(fasttext_corpus_path, dim=int(dim), epoch=epoch)
    model.save_model(str(product))


def make_igraph(upstream, product):
    python_files_df = pd.read_csv(upstream["prepare_dependency_records"]).dropna()
    graph = node_embedding_evaluation.make_igraph(python_files_df)
    pickle.dump(graph, open(str(product), "wb"))


def _word_vectors_to_word2vec_format_generator(vocabulary, word_vectors):
    for (word, vector) in zip(vocabulary, word_vectors):
        yield word + " " + " ".join([str("{:.5f}".format(f)) for f in vector])


def store_word_vectors(words, word_vectors, file_name):
    with open(file_name, "w") as f:
        f.write(str(len(words)) + " " + str(word_vectors.shape[1]) + "\n")
        for line in _word_vectors_to_word2vec_format_generator(words, module_vectors):
            f.write(line + "\n")
