import ast
import pickle

import numpy as np
import pandas as pd
import tqdm
from sklearn import feature_extraction, metrics

from github_search import python_function_code


def get_per_repo_similarities(paperswithcode_df, repo_grouped_contents, similar_col):
    bow_vectorizer = feature_extraction.text.CountVectorizer()

    bow_vectorizer.fit(paperswithcode_df["abstract"])
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


def get_top_similar_files_dict(python_files_df, paperswithcode_df, similar_col):
    repo_grouped_contents = {
        repo: python_files_df[python_files_df["repo_name"] == repo]["content"]
        for repo in tqdm.tqdm(paperswithcode_df["repo"])
    }
    per_repo_file_similarities = get_per_repo_similarities(
        paperswithcode_df, repo_grouped_contents, similar_col
    )
    top_similar_files = {
        key: list(
            repo_grouped_contents[key].iloc[
                np.argsort(-per_repo_file_similarities[key])
            ]
        )
        for key in per_repo_file_similarities.keys()
    }
    return top_similar_files


def select_repo_files(python_files_path, paperswithcode_path, similar_col, product):
    python_files_df = pd.read_feather(python_files_path).dropna()
    paperswithcode_with_tasks_df = pd.read_csv(paperswithcode_path).dropna(
        subset=["least_common_task", similar_col]
    )
    paperswithcode_with_tasks_df["tasks"] = paperswithcode_with_tasks_df["tasks"].apply(
        ast.literal_eval
    )
    top_similar_files = get_top_similar_files_dict(
        python_files_df, paperswithcode_with_tasks_df, similar_col=similar_col
    )
    pickle.dump(top_similar_files, open(str(product), "wb"))
