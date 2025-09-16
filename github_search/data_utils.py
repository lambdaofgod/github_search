
import numpy as np
import attr
import pandas as pd
from sklearn import model_selection

from github_search import paperswithcode_tasks


def get_repo_records(repos: pd.Series, dependency_records_df: pd.DataFrame):
    repo_dependency_records_df = dependency_records_df[
        dependency_records_df["edge_type"] == "repo-file"
    ]
    repo_dependencies = repo_dependency_records_df.groupby("source").apply(
        lambda df: " ".join(df["destination"]).replace(df["source"].iloc[0] + ":", "")
    )

    return repo_dependencies[repo_dependencies.index.isin(repos)]


def make_records_df(repos: pd.Series, repo_dependent_nodes: pd.Series) -> pd.DataFrame:
    """
    repos: series of repositories, for example "trangvu/ape-npi"
    repo_dependent_nodes: series of repos (index) with list of connected nodes (values)
    """
    return pd.DataFrame.from_records(
        [
            {
                "source": src,
                "destination": dst.replace(src + ":", ""),
                "edge_type": "repo-file",
            }
            for (src, destinations) in zip(repos, repo_dependent_nodes)
            for dst in destinations.split()
        ]
    )


@attr.s
class RepoTaskData:
    tasks = attr.ib()
    repos = attr.ib()
    X = attr.ib()
    all_tasks = attr.ib()
    y = attr.ib()

    def split_tasks(area_grouped_tasks, task_counts, test_size=0.2):
        stratify_df = area_grouped_tasks.merge(task_counts, on="task")
        stratify_df["task_count"] = np.clip(
            np.log10(stratify_df["task_count"]).astype(int), 0, 2
        )
        stratify_df = stratify_df.sort_values("task_count")
        stratify_df = stratify_df.drop_duplicates(subset="task")
        tasks_train, tasks_test = model_selection.train_test_split(
            stratify_df["task"],
            stratify=stratify_df[["area", "task_count"]],
            test_size=test_size,
            random_state=0,
        )
        return tasks_train, tasks_test

    def create_split(
        tasks_test,
        all_tasks,
        paperswithcode_with_features_df,
        X_repr,
        y_col="least_common_task",
    ):
        train_indicator = paperswithcode_with_features_df["tasks"].apply(
            lambda ts: not (any([t in list(tasks_test) for t in ts]))
        )
        repos_train = paperswithcode_with_features_df["repo"][train_indicator]
        repos_test = paperswithcode_with_features_df["repo"][~train_indicator]
        X_repr = X_repr.apply(lambda x: " ".join(x))
        X_train = X_repr[train_indicator]
        X_test = X_repr[~train_indicator]
        all_tasks_train = all_tasks[train_indicator]
        all_tasks_test = all_tasks[~train_indicator]
        y_train = (
            paperswithcode_with_features_df[train_indicator][y_col]
            .str.lower()
            .apply(paperswithcode_tasks.clean_task_name)
        )
        y_test = (
            paperswithcode_with_features_df[~train_indicator][y_col]
            .str.lower()
            .apply(paperswithcode_tasks.clean_task_name)
        )

        tasks_train = [task for task in all_tasks if not task in tasks_test]
        return (
            RepoTaskData(tasks_train, repos_train, X_train, all_tasks_train, y_train),
            RepoTaskData(tasks_test, repos_test, X_test, all_tasks_test, y_test),
        )
