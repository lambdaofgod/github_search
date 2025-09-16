import os
import polars as pl

__all__ = [
    "get_modules_string",
    "get_module_corpus",
    "prepare_module_corpus",
    "store_word_vectors",
]

import ast
import csv
import logging

import pandas as pd
import tqdm

from github_search import (
    paperswithcode_tasks,
    parsing_imports,
    python_call_graph,
    python_function_code,
    utils,
)
from github_search.papers_with_code import paperswithcode_tasks

csv.field_size_limit(1000000)


logging.basicConfig(level="INFO")


def get_task_counts(paperswithcode_path, product):
    papers_with_tasks_df = utils.load_paperswithcode_df(
        paperswithcode_path, drop_na_cols=["tasks"]
    )
    return (
        papers_with_tasks_df["tasks"]
        .explode()
        .dropna()
        .value_counts()
        .rename("task_count")
        .reset_index()
        .rename({"index": "task"}, axis=1)
        .to_csv(str(product), index=False)
    )


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
    python_files_df = pd.concat([pd.read_feather(path) for path in python_file_paths])
    print(python_files_df.shape)
    repo_names = python_files_df["repo_name"]
    paperswithcode_df, all_papers_df = paperswithcode_tasks.get_paperswithcode_dfs()
    papers_with_repo_df = paperswithcode_tasks.get_papers_with_repo_df(
        all_papers_df, paperswithcode_df, repo_names
    )
    papers_with_repo_df = paperswithcode_tasks.get_papers_with_biggest_tasks(
        papers_with_repo_df, 2000
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
        [pd.read_feather(path) for path in python_file_paths]
    ).dropna(subset=["repo_name", "content"])
    get_module_corpus(python_files_df).to_csv(str(product))


def prepare_dependency_records(
    python_file_path,
    sample_files_per_repo,
    add_repo_col,
    use_basename,
    excluded_prefix,
    product,
):
    """
    prepare python dependency graph records (function calls, files in repo) csv
    """
    python_files_df = pd.read_parquet(python_file_path).dropna(
        subset=["repo_name", "content"]
    )
    contains_excluded_prefix = python_files_df["path"].str.startswith(excluded_prefix)
    python_files_df = python_files_df[~contains_excluded_prefix]
    logging.info(f"found {len(python_files_df)}")
    dependency_records_df = python_call_graph.get_dependency_records_df(
        python_files_df,
        sample_files_per_repo,
        add_repo_col,
        use_basename,
    )
    dependency_records_df.reset_index().dropna().to_feather(str(product))


def postprocess_dependency_records(
    product, use_additional_records, upstream, description_mode
):
    """
    filter out ROOT records, add
    """
    dependency_records_df = pd.read_feather(upstream["prepare_dependency_records"])
    non_root_dependency_records_df = dependency_records_df[
        dependency_records_df["source"] != "<ROOT>"
    ]
    if description_mode:
        non_root_dependency_records_df = python_call_graph.get_description_records_df(
            non_root_dependency_records_df.dropna()
        )

    if use_additional_records:
        papers_data_df = pd.read_csv(upstream["prepare_paperswithcode_with_imports_df"])
        papers_data_df["tasks"] = papers_data_df["tasks"].apply(ast.literal_eval)
        paperswithcode_single_task_df = papers_data_df.explode("tasks")
        shared_task_pairs_df = paperswithcode_single_task_df.merge(
            paperswithcode_single_task_df, on="tasks"
        )
        shared_task_pairs_df = shared_task_pairs_df[
            shared_task_pairs_df["repo_x"] != shared_task_pairs_df["repo_y"]
        ]

        shared_task_repo_pairs = list(
            set(
                [
                    tuple(set(t))
                    for t in shared_task_pairs_df[["repo_x", "repo_y"]].itertuples(
                        index=False
                    )
                ]
            )
        )
        shared_task_records_df = pd.DataFrame.from_records(
            {"source": r1, "destination": r2, "edge_type": "repo-repo"}
            for (r1, r2) in shared_task_repo_pairs
        )
        non_root_dependency_records_df = pd.concat(
            [shared_task_records_df, non_root_dependency_records_df]
        )
    non_root_dependency_records_df.reset_index(drop=True).dropna().to_feather(
        str(product)
    )


# TODO: WYTRENUJ WORD2VECA Z PYTHON CODE TOKENIZEREMM


def train_python_token_fasttext(python_file_path, epoch, dim, n_cores, product):
    import fasttext

    python_files_df = (
        pl.scan_parquet(python_file_path)
        .select([pl.col("content")])
        .collect()
        .to_pandas()
    )
    fasttext_corpus_path = "/tmp/python_files.csv"
    if not os.path.exists(fasttext_corpus_path):
        python_files_df["content"].dropna().to_csv(
            fasttext_corpus_path,
            index=False,
            header=False,
        )
    model = fasttext.train_unsupervised(
        fasttext_corpus_path, dim=int(dim), epoch=epoch, thread=n_cores
    )
    model.save_model(str(product))


def prepare_function_code_df(product, max_depth, n_cores, python_file_path):
    python_files_df = pd.read_parquet(python_file_path).dropna()
    print("#" * 50)
    print("PYTHON FILES")
    print(python_files_df.shape)
    functions_df = python_function_code.get_function_data_df(python_files_df, max_depth)
    functions_df.to_parquet(product)


def prepare_function_signatures_df(product, n_cores, python_file_path):
    python_files_df = pd.read_parquet(python_file_path).dropna()
    print("#" * 50)
    print("PYTHON FILES")
    print(python_files_df.shape)
    functions_df = python_function_code.get_function_signatures_df(
        python_files_df, n_cores=n_cores
    )
    functions_df.to_parquet(product)


def _word_vectors_to_word2vec_format_generator(vocabulary, word_vectors):
    for word, vector in zip(vocabulary, word_vectors):
        yield word + " " + " ".join([str("{:.5f}".format(f)) for f in vector])
