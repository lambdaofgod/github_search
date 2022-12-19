from github_search import data_utils, utils
import pandas as pd


def prepare_task_train_test_split(upstream, test_size, product):
    area_grouped_tasks = pd.read_csv(str(upstream["prepare_area_grouped_tasks"]))
    task_counts = pd.read_csv(
        str(upstream["pwc_data.prepare_final_paperswithcode_df"]["task_counts_path"])
    )
    tasks_train, tasks_test = data_utils.RepoTaskData.split_tasks(
        area_grouped_tasks, task_counts, test_size=test_size
    )
    tasks_train.to_csv(product["train"], index=None)
    tasks_test.to_csv(product["test"], index=None)


def prepare_repo_train_test_split(upstream, paperswithcode_with_tasks_path, product):
    papers_with_tasks_df = utils.load_paperswithcode_df(
        paperswithcode_with_tasks_path, drop_na_cols=["tasks"]
    )
    test_tasks = set(
        pd.read_csv(str(upstream["prepare_task_train_test_split"]["test"])).iloc[:, 0]
    )
    contains_test_tasks = papers_with_tasks_df["tasks"].apply(
        lambda tasks: all(task in test_tasks for task in tasks)
    )

    test_repos = papers_with_tasks_df[contains_test_tasks]
    train_repos = papers_with_tasks_df[~contains_test_tasks]
    test_repos.to_csv(product["test"], index=False)
    train_repos.to_csv(product["train"], index=False)
