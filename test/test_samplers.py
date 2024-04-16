from github_search.samplers import TaskSampler, RepoSampler
import pandas as pd
import ast


pwc_df = pd.read_json("output/paperswithcode_with_readmes.json.gz")
pwc_df["tasks"] = pwc_df["tasks"].apply(ast.literal_eval)


def test_task_sampler_min():
    task_lists = pwc_df["tasks"]
    sample_size = 4
    min_size = 5000
    tasks = TaskSampler.sample_tasks_from_lists(
        task_lists, sample_size, min_size)
    assert len(tasks) == 4


def test_task_sampler_min_max():
    task_lists = pwc_df["tasks"]
    sample_size = 5
    min_size = 4000
    max_size = 7000
    tasks = TaskSampler.sample_tasks_from_lists(
        task_lists, sample_size, min_size)
    assert len(tasks) == 5


def test_repo_sampler():
    task_lists = pwc_df["tasks"]
    tasks_sample_size = 4
    min_task_size = 5000
    tasks = TaskSampler.sample_tasks_from_lists(
        task_lists, tasks_sample_size, min_task_size)

    n_repos_per_task = 10
    repos_sample = RepoSampler.sample_repos(
        pwc_df, tasks, sample_size_per_task=n_repos_per_task)
    assert len(repos_sample) == 33
