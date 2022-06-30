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
        area_task_descriptions = []
        while not task_exhausted:
            try:
                area_tasks_client_result = client.area_task_list(
                    a.id, page=i, items_per_page=1000).results
                area_tasks += [t.id for t in area_tasks_client_result]
                area_task_descriptions += [
                    t.description for t in area_tasks_client_result
                ]
                i += 1
            except tea_client.errors.HttpClientError:
                task_exhausted = True
                i = 1
        area_grouped_tasks[a.id] = {
            "task_names": area_tasks,
            "task_descriptions": area_task_descriptions
        }
        n_tasks_per_area = len(area_tasks)
        logging.info(a.id + ":" + str(n_tasks_per_area))
        s += n_tasks_per_area
    logging.info("total tasks:" + str(s))
    return area_grouped_tasks


def preprocess_area_grouped_tasks(area_grouped_tasks):
    area_grouped_tasks["task"] = area_grouped_tasks["task"].apply(
        paperswithcode_tasks.clean_task_name)
    area_counts = area_grouped_tasks["area"].value_counts()
    area_grouped_tasks = area_grouped_tasks[area_grouped_tasks["area"].isin(
        area_counts.index[area_counts > 1])]
    return area_grouped_tasks


def get_area_grouped_tasks_df():
    area_grouped_tasks_dict = get_area_grouped_tasks()
    area_tasks_df = pd.DataFrame.from_records(
        [{
            "area": area,
            "task": task,
            "task_description": description
        } for area in area_grouped_tasks_dict.keys()
         for (task, description) in zip(*area_grouped_tasks_dict[area].values())])
    return area_tasks_df



def prepare_area_grouped_tasks(product):
    area_tasks_df = get_area_grouped_tasks_df()
    preprocess_area_grouped_tasks(area_tasks_df).to_csv(str(product),
                                                        index=None)

def prepare_paperswithcode_with_areas_df(paperswithcode_df, area_tasks_df):
    return paperswithcode_df.merge(area_tasks_df, left_on="least_common_task", right_on="task")
