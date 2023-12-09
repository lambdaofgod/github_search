from github_search import data_utils, utils
import pandas as pd


def prepare_task_train_test_split(upstream, test_size, product):
    area_grouped_tasks = pd.read_csv(
        str(upstream["prepare_area_grouped_tasks"]))
    task_counts = pd.read_csv(
        str(upstream["pwc_data.prepare_final_paperswithcode_df"]
            ["task_counts_path"])
    )

    if test_size == 1:
        tasks_train = pd.Series([], name="task")
        tasks_test = area_grouped_tasks["task"]
    else:
        tasks_train, tasks_test = data_utils.RepoTaskData.split_tasks(
            area_grouped_tasks, task_counts, test_size=test_size
        )
    tasks_train.to_csv(product["train"], index=None)
    tasks_test.to_csv(product["test"], index=None)


def prepare_repo_train_test_split(upstream, product):

    pwc_path = upstream["pwc_data.prepare_paperswithcode_with_readmes"]
    papers_with_tasks_df = utils.load_paperswithcode_df(
        str(pwc_path), drop_na_cols=["tasks"]
    )
    train_tasks = set(
        pd.read_csv(
            str(upstream["prepare_task_train_test_split"]["train"])).iloc[:, 0]
    )
    contains_train_tasks = papers_with_tasks_df["tasks"].apply(
        lambda tasks: all(task in train_tasks for task in tasks)
    )
    test_repos = papers_with_tasks_df[~contains_train_tasks]
    train_repos = papers_with_tasks_df[contains_train_tasks]
    test_repos.to_json(product["test"], index=False)
    train_repos.to_json(product["train"], index=False)
