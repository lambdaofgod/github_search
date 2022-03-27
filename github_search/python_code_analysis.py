import ast
import pickle

import numpy as np
import pandas as pd
import tqdm
from sklearn import feature_extraction, metrics

from github_search import python_function_code


def get_per_repo_similarities(
    paperswithcode_df: pd.DataFrame,
    repo_grouped_contents: dict,
    similar_col: str,
    bow_vectorizer_class: str,
):
    bow_vectorizer = getattr(feature_extraction.text, bow_vectorizer_class)

    bow_vectorizer.fit(paperswithcode_df[similar_col])
    return {
        repo: metrics.pairwise.cosine_similarity(
            bow_vectorizer.transform([abstract]),
            bow_vectorizer.transform(repo_grouped_contents[repo]),
        )[0]
        for (repo, abstract) in paperswithcode_df[["repo", similar_col]].itertuples(
            index=False
        )
        if len(repo_grouped_contents[repo]) > 0
    }


def get_top_similar_files_dict(
    python_files_df: pd.DataFrame,
    paperswithcode_df: pd.DataFrame,
    similar_col: str,
    files_per_repo: int,
    bow_vectorizer_class: str,
):
    repo_grouped_contents = {
        repo: python_files_df[python_files_df["repo_name"] == repo]["content"]
        for repo in tqdm.tqdm(paperswithcode_df["repo"])
    }
    per_repo_file_similarities = get_per_repo_similarities(
        paperswithcode_df, repo_grouped_contents, similar_col, bow_vectorizer_class
    )
    top_similar_files = {
        key: list(
            repo_grouped_contents[key].iloc[
                np.argsort(-per_repo_file_similarities[key])[:files_per_repo]
            ]
        )
        for key in per_repo_file_similarities.keys()
    }
    return top_similar_files


def select_repo_files(
    python_files_path: str,
    paperswithcode_path: str,
    similar_col: str,
    files_per_repo: int,
    product: str,
    bow_vectorizer_class: str,
):
    python_files_df = pd.read_feather(python_files_path).dropna()
    paperswithcode_with_tasks_df = pd.read_csv(paperswithcode_path).dropna(
        subset=["least_common_task", similar_col]
    )
    paperswithcode_with_tasks_df["tasks"] = paperswithcode_with_tasks_df["tasks"].apply(
        ast.literal_eval
    )
    top_similar_files = get_top_similar_files_dict(
        python_files_df,
        paperswithcode_with_tasks_df,
        similar_col=similar_col,
        files_per_repo=files_per_repo,
        bow_vectorizer_class=bow_vectorizer_class,
    )
    pickle.dump(top_similar_files, open(str(product), "wb"))
