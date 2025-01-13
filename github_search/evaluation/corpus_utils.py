from pathlib import Path

import pandas as pd
from beir.retrieval.search.lexical import BM25Search as BM25
from github_search.evaluation.beir_evaluation import (
    EvaluateRetrievalCustom as CorpusDataLoader,
)
import logging


class ExperimentParams:
    sampled_repos_per_task = 20
    min_repos_per_task = 10


# sampled_repos_df, sampled_generated_readmes_df, sample_python_code_df = small_sample_loader.load_corpus_dfs(librarian_signatures_df["repo"])


def filter_dfs_by_cols_in(dfs, col_values, colnames=["repo", "repo_name"]):
    out_dfs = []
    for df in dfs:
        df_cols = [c for c in colnames if c in df.columns]
        col = df_cols[0]
        filtered_df = df[df[col].isin(col_values)]
        out_dfs.append(filtered_df)
    return out_dfs


def align_dfs(dfs, colname="repo"):
    df0 = dfs[0].reset_index()
    df_index = df0[colname]
    new_dfs = [df.set_index(colname).loc[df_index].reset_index() for df in dfs[1:]]
    return [df0] + new_dfs


def get_repos_for_query(query, repos_df):
    return repos_df[repos_df["tasks"].apply(lambda ts: query in ts)]


def get_queries(repos_df, min_query_count):
    all_queries = repos_df["query_tasks"].explode()
    qcounts = all_queries.value_counts()
    return qcounts[qcounts >= min_query_count].index.to_list()


def prepare_query_data(repos_df, min_query_count=5):
    task_queries = {
        str(i): query
        for (i, query) in enumerate(
            get_queries(repos_df, min_query_count=min_query_count)
        )
    }

    task_qrels = {
        qid: {
            str(corpus_id): 1
            for corpus_id in get_repos_for_query(task_queries[qid], repos_df).index
        }
        for qid in task_queries.keys()
    }
    return task_queries, task_qrels


def prepare_readme_corpus(repos_df):
    return {
        str(i): {"text": row["readme"], "title": row["repo"], "tasks": row["tasks"]}
        for (i, row) in repos_df.iterrows()
    }


def prepare_generated_readme_corpus(repos_df, generated_readmes_df, columns=["answer"]):
    return {
        str(i): {"text": "\n".join(row[columns]), "title": row["repo_name"]}
        for (i, row) in generated_readmes_df.iterrows()
    }


def prepare_code_corpus(repos_df, selected_python_code_df):
    per_repo_code_df = selected_python_code_df.groupby("repo_name").apply(
        lambda df: "\n\n".join(df["selected_code"].fillna(""))
    )
    per_repo_code_df = per_repo_code_df.loc[repos_df["repo"]].reset_index()
    return {
        str(i): {"text": row[0], "title": row["repo_name"]}
        for (i, row) in per_repo_code_df.iterrows()
    }


# THIS IS FOR ONE GENERATION ONLY NOW
def prepare_librarian_corpora(repos_df, sampled_librarian_signatures_df):
    columns = ["dependency_signature", "repository_signature", "generated_tasks"]
    sampled_librarian_signatures_df = (
        sampled_librarian_signatures_df.set_index("repo")
        .loc[repos_df["repo"]]
        .reset_index()
    )
    if "generation" in sampled_librarian_signatures_df.columns:
        return {
            (column, g): {
                str(i): {"text": row[column], "title": row["repo"]}
                for (i, row) in sampled_librarian_signatures_df[
                    sampled_librarian_signatures_df["generation"] == g
                ]
                .reset_index(drop=True)[["repo", column]]
                .iterrows()
            }
            for column in columns
            for g in sampled_librarian_signatures_df["generation"].unique()
        }
    else:
        return {
            column: {
                str(i): {"text": row[column], "title": row["repo"]}
                for (i, row) in sampled_librarian_signatures_df.reset_index(drop=True)[
                    ["repo", column]
                ].iterrows()
            }
            for column in columns
        }


def prepare_basic_corpora(repos_df, selected_python_code_df):
    readme_corpus = prepare_readme_corpus(repos_df)
    selected_python_code_corpus = prepare_code_corpus(repos_df, selected_python_code_df)
    return {"readme": readme_corpus, "selected_code": selected_python_code_corpus}


def prepare_corpora(repos_df, generated_readmes_df, selected_python_code_df):

    # aligning
    # this is done in case the dfs come from unsynced dagster runs
    repos = set(repos_df["repo"]).intersection(generated_readmes_df["repo_name"])
    repos_df = (
        repos_df[repos_df["repo"].isin(repos)]
        .sort_values("repo")
        .reset_index(drop=True)
    )
    generated_readmes_df = (
        generated_readmes_df[generated_readmes_df["repo_name"].isin(repos)]
        .sort_values("repo_name")
        .reset_index(drop=True)
    )
    selected_python_code_df = selected_python_code_df[
        selected_python_code_df["repo_name"].isin(repos)
    ].sort_values("repo_name")

    basic_corpora = prepare_basic_corpora(repos_df, selected_python_code_df)
    readme_corpus = basic_corpora["readme"]
    selected_python_code_corpus = basic_corpora["selected_code"]
    generated_readme_corpus = prepare_generated_readme_corpus(
        repos_df, generated_readmes_df
    )
    generated_rationale_corpus = prepare_generated_readme_corpus(
        repos_df, generated_readmes_df, columns=["rationale"]
    )
    generated_readme_context_corpus = prepare_generated_readme_corpus(
        repos_df, generated_readmes_df, columns=["context_history"]
    )

    # assert len(readme_corpus) == len(generated_readme_corpus)
    # assert len(selected_python_code_corpus) == len(readme_corpus)

    for k in readme_corpus.keys():
        assert readme_corpus[k]["title"] == generated_readme_corpus[k]["title"], str(
            (readme_corpus[k]["title"], generated_readme_corpus[k]["title"])
        )
        assert readme_corpus[k]["title"] == selected_python_code_corpus[k]["title"]
    return {
        "readme": readme_corpus,
        "generated_readme": generated_readme_corpus,
        "selected_code": selected_python_code_corpus,
        "generated_rationale": generated_rationale_corpus,
        "generation_context": generated_readme_context_corpus,
    }
