import paperswithcode
import pandas as pd
import logging
import tea_client
from github_search import paperswithcode_tasks

logging.basicConfig(level="INFO")


def get_area_grouped_tasks():
    client = paperswithcode.PapersWithCodeClient()
    areas = client.area_list().results
    s = 0

    area_grouped_tasks = {}

    for a in areas:
        logging.info("downloading tasks from {} area".format(a))
        task_exhausted = False
        i = 1
        area_tasks = []
        while not task_exhausted:
            try:
                area_tasks += [
                    t.id
                    for t in client.area_task_list(
                        a.id, page=i, items_per_page=1000
                    ).results
                ]
                i += 1
            except tea_client.errors.HttpClientError:
                task_exhausted = True
                i = 1
        area_grouped_tasks[a.id] = area_tasks
        n_tasks_per_area = len(area_tasks)
        logging.info(a.id + ":" + str(n_tasks_per_area))
        s += n_tasks_per_area
    logging.info("total tasks:" + str(s))
    return area_grouped_tasks


def preprocess_area_grouped_tasks(area_grouped_tasks):
    area_grouped_tasks["task"] = area_grouped_tasks["task"].apply(
        paperswithcode_tasks.clean_task_name
    )
    area_counts = area_grouped_tasks["area"].value_counts()
    area_grouped_tasks = area_grouped_tasks[
        area_grouped_tasks["area"].isin(area_counts.index[area_counts > 1])
    ]
    return area_grouped_tasks


def prepare_area_grouped_tasks(product):
    area_grouped_tasks_dict = get_area_grouped_tasks()
    area_tasks_df = pd.DataFrame(
        {
            "area": area_grouped_tasks_dict.keys(),
            "task": area_grouped_tasks_dict.values(),
        }
    ).explode("task")
    preprocess_area_grouped_tasks(area_tasks_df).to_csv(str(product), index=None)
