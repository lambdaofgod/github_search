import abc
import pandas as pd
from pydantic import BaseModel
from typing import Optional


class TaskSampler:
    @classmethod
    def get_task_counts(cls, task_lists: pd.Series):
        return task_lists.explode().value_counts()

    @classmethod
    def sample_tasks_from_lists(cls, task_lists, sample_size: int, min_size: int, max_size: int = 10000):
        task_counts = cls.get_task_counts(task_lists)
        valid_tasks = task_counts[(task_counts >= min_size) & (
            task_counts <= max_size)]
        return valid_tasks.sample(sample_size)


class RepoSampler(BaseModel):
    repo_col: str = "repo"
    tasks_col: str = "tasks"

    @ classmethod
    def sample_repos(cls, repos_df, sample_tasks, sample_size_per_task: int) -> pd.DataFrame:
        if type(sample_tasks) is pd.Series:
            sample_tasks = sample_tasks.index
        repo_names = []
        for task in sample_tasks:
            task_repos = repos_df[repos_df["tasks"].apply(
                lambda ts: task in ts)]["repo"]
            repo_names.extend(task_repos.sample(sample_size_per_task))
        repo_names = list(set(repo_names))
        return repos_df[repos_df["repo"].isin(repo_names)]
