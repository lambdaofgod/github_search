import abc
import pandas as pd
from pydantic import BaseModel
from typing import Optional
import logging
import ast
import hashlib
from pydantic import BaseModel
import itertools
import scipy
from github_search.utils import get_random_retrieval_baseline
import tqdm


class TaskSizeRepoSampler(BaseModel):
    min_task_count: int = 5
    n_repos_per_task: int = 10
    max_repos_per_task: int = 500

    def add_valid_tasks_from_sample(self, repos_df, sampled_task_repos):
        seq_sampled_repos = list(
            set(itertools.chain.from_iterable(sampled_task_repos.values()))
        )
        seq_sampled_repos_df = repos_df[repos_df["repo"].isin(seq_sampled_repos)]
        sampled_task_counts = seq_sampled_repos_df["tasks"].explode().value_counts()
        for task in sampled_task_counts[
            sampled_task_counts >= self.min_task_count
        ].index:
            if sampled_task_repos.get(task) is None:
                sampled_task_repos[task] = seq_sampled_repos_df[
                    seq_sampled_repos_df["tasks"].apply(
                        lambda ts: any(t == task for t in ts)
                    )
                ]
        return sampled_task_repos

    def _get_sampled_task_repos(self, repos_df):
        task_counts = repos_df["tasks"].explode().value_counts()
        sampled_task_repos = self.sample_repos_sequentially(
            repos_df,
            task_counts.sort_values().index,
            min_task_count=self.min_task_count,
            n_repos_per_task=self.n_repos_per_task,
            max_repos_per_task=self.max_repos_per_task,
        )
        sampled_task_repos = self.add_valid_tasks_from_sample(
            repos_df, sampled_task_repos
        )
        sampled_repos = list(
            set(itertools.chain.from_iterable(sampled_task_repos.values()))
        )
        sampled_repos_df = repos_df[repos_df["repo"].isin(sampled_repos)].copy()
        sampled_repos_df["query_tasks"] = sampled_repos_df["tasks"].apply(
            lambda ts: [t for t in ts if t in sampled_task_repos.keys()]
        )
        return sampled_task_repos, sampled_repos_df

    def get_sampled_task_repos_df(self, repos_df, max_baseline_score=None):
        sampled_task_repos, sampled_repos_df = self._get_sampled_task_repos(repos_df)
        if max_baseline_score is not None:
            sampled_task_repos = self.filter_max_random_score_tasks(
                sampled_task_repos, sampled_repos_df.shape[0], max_baseline_score
            )
            sampled_repos_df["query_tasks"] = sampled_repos_df["query_tasks"].apply(
                lambda ts: [t for t in ts if t in sampled_task_repos.keys()]
            )
        return sampled_repos_df

    def estimate_task_random_accuracy(
        self, sampled_repos_df=None, task_counts=None, n_repos=None
    ):
        if sampled_repos_df is not None:
            task_counts = sampled_repos_df["query_tasks"].explode().value_counts()
            n_repos = sampled_repos_df.shape[0]
        return task_counts.apply(lambda tc: get_random_retrieval_baseline(n_repos, tc))

    def filter_max_random_score_tasks(
        self, sampled_task_repos, n_repos, max_baseline_score
    ):
        sampled_task_repos_srs = pd.Series(sampled_task_repos)
        task_counts = sampled_task_repos_srs.apply(len)
        random_scores = self.estimate_task_random_accuracy(
            task_counts=task_counts, n_repos=n_repos
        )
        valid_random_score_tasks = random_scores[
            random_scores < max_baseline_score
        ].index
        return sampled_task_repos_srs[valid_random_score_tasks].to_dict()

    @classmethod
    def sample_repos_sequentially(
        cls,
        repos_df,
        tasks,
        min_task_count=10,
        n_repos_per_task=25,
        max_repos_per_task=500,
    ):
        task_repos = dict()
        task_counts_so_far = dict()

        for t in tqdm.tqdm(tasks):
            t_repos_df = repos_df[repos_df["tasks"].apply(lambda ts: t in ts)]
            t_repos_df = t_repos_df[
                t_repos_df["tasks"].apply(
                    lambda ts: all(
                        task_counts_so_far.get(t, 0) <= max_repos_per_task for t in ts
                    )
                )
            ]
            if t_repos_df.shape[0] >= min_task_count:
                t_repos = t_repos_df["repo"]
                if len(t_repos) >= n_repos_per_task:
                    # weights = t_repos_df[t_repos_df["repo"].isin(t_repos)]["tasks"].apply(len)
                    t_repos = t_repos.iloc[:n_repos_per_task]
                for repo_tasks in repos_df[repos_df["repo"].isin(t_repos)]["tasks"]:
                    for task in repo_tasks:
                        if task_counts_so_far.get(task) is None:
                            task_counts_so_far[task] = 1
                        else:
                            task_counts_so_far[task] += 1
                task_repos[t] = t_repos.to_list()

        return task_repos


class DeterministicSampler:
    """
    this class is used to make sampling deterministic
    we do this by specifying a 'random' function
    which will be used for sorting samples
    """

    @classmethod
    def string_pseudohash(cls, s):
        return hashlib.shake_256(s.encode("utf-8")).hexdigest(16)

    @classmethod
    def sample(cls, strings, sample_size):
        if type(strings) == pd.Series:
            if strings.dtype == "object":
                strings = strings.values
            else:
                strings = strings.index
        pseudohashes = [cls.string_pseudohash(s) for s in strings]
        strings_by_pseudohashes = sorted(zip(strings, pseudohashes), key=lambda x: x[1])
        if len(strings_by_pseudohashes) < sample_size:
            logging.warning(
                "Sample size is larger than the number of strings. Returning all strings."
            )
        return [s for s, _ in strings_by_pseudohashes[:sample_size]]


class TaskSampler:
    @classmethod
    def get_task_counts(cls, task_lists: pd.Series):
        return task_lists.explode().value_counts()

    @classmethod
    def sample_tasks_from_lists(
        cls,
        task_lists,
        sample_size: int,
        min_size: int,
        max_size: int = 10000,
        sample_fn=DeterministicSampler.sample,
    ):
        task_counts = cls.get_task_counts(task_lists)
        valid_tasks = task_counts[(task_counts >= min_size) & (task_counts <= max_size)]
        return sample_fn(valid_tasks, sample_size)


class RepoSampler(BaseModel):
    repo_col: str = "repo"
    tasks_col: str = "tasks"

    @classmethod
    def sample_repos(
        cls,
        repos_df,
        sample_tasks,
        sample_size_per_task: int,
        min_repo_tasks: int = 1,
        sample_fn=DeterministicSampler.sample,
    ) -> pd.DataFrame:
        if type(sample_tasks) is pd.Series:
            sample_tasks = sample_tasks.index
        repo_names = []
        for task in sample_tasks:
            task_repos_mask = repos_df["tasks"].apply(lambda ts: task in ts)
            min_repo_tasks_mask = repos_df["tasks"].apply(
                lambda ts: len(ts) >= min_repo_tasks
            )
            task_repos = repos_df[task_repos_mask & min_repo_tasks_mask]["repo"]
            if len(task_repos) < sample_size_per_task:
                logging.warning(
                    f"Task {task} has less than {sample_size_per_task} repos. Sampling all."
                )
            repo_names.extend(sample_fn(task_repos, sample_size_per_task))
        repo_names = list(set(repo_names))
        return repos_df[repos_df["repo"].isin(repo_names)]
