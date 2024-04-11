import abc
import pandas as pd
from pydantic import BaseModel
from typing import Optional
import logging
import ast


class DeterministicSampler:
    """
    this class is used to make sampling deterministic
    we do this by specifying a 'random' function
    which will be used for sorting samples
    """

    @classmethod
    def string_pseudohash(cls, s):
        return sum([ord(c) for c in s])

    @classmethod
    def sample(cls, strings, sample_size):
        if type(strings) == pd.Series:
            if strings.dtype == "object":
                strings = strings.values
            else:
                strings = strings.index
        pseudohashes = [cls.string_pseudohash(s) for s in strings]
        strings_by_pseudohashes = sorted(
            zip(strings, pseudohashes), key=lambda x: x[1])
        if len(strings_by_pseudohashes) < sample_size:
            logging.warning(
                "Sample size is larger than the number of strings. Returning all strings.")
        return [s for s, _ in strings_by_pseudohashes[:sample_size]]


class TaskSampler:
    @classmethod
    def get_task_counts(cls, task_lists: pd.Series):
        return task_lists.explode().value_counts()

    @classmethod
    def sample_tasks_from_lists(cls, task_lists, sample_size: int, min_size: int, max_size: int = 10000, sample_fn=DeterministicSampler.sample):
        task_counts = cls.get_task_counts(task_lists)
        valid_tasks = task_counts[(task_counts >= min_size) & (
            task_counts <= max_size)]
        return sample_fn(valid_tasks, sample_size)


class RepoSampler(BaseModel):
    repo_col: str = "repo"
    tasks_col: str = "tasks"

    @classmethod
    def sample_repos(cls, repos_df, sample_tasks, sample_size_per_task: int, min_repo_tasks: int = 1, sample_fn=DeterministicSampler.sample) -> pd.DataFrame:
        if type(sample_tasks) is pd.Series:
            sample_tasks = sample_tasks.index
        repo_names = []
        for task in sample_tasks:
            task_repos_mask = repos_df["tasks"].apply(
                lambda ts: task in ts)
            min_repo_tasks_mask = repos_df["tasks"].apply(
                lambda ts: len(ts) >= min_repo_tasks)
            task_repos = repos_df[task_repos_mask &
                                  min_repo_tasks_mask]["repo"]
            if len(task_repos) < sample_size_per_task:
                logging.warning(
                    f"Task {task} has less than {sample_size_per_task} repos. Sampling all.")
            repo_names.extend(sample_fn(task_repos, sample_size_per_task))
        repo_names = list(set(repo_names))
        return repos_df[repos_df["repo"].isin(repo_names)]
