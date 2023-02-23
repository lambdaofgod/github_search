import pandas as pd


def get_python_files_df():
    return pd.read_feather("../output/python_functions.feather")


def filter_out_big_repos(python_files_df, max_files=1000):
    repo_function_counts = python_files_df["repo_name"].value_counts()
    small_repos = repo_function_counts[repo_function_counts <= max_files].index
    return python_files_df[python_files_df["repo_name"].isin(small_repos)]


def get_repos_with_readmes():
    return pd.read_csv("../output/papers_with_readmes.csv")
