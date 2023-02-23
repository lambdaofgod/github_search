from pathlib import Path as P
import yaml
import re


def get_config_path(task_name_or_path):
    task_name_or_path = P(task_name_or_path)
    task_name = task_name_or_path if task_name_or_path.name == task_name_or_path else task_name_or_path.name
    return P("conf") / (re.sub("nbow_|-\d+", "", task_name))

def get_config(task_name):
    with open(get_config_path(task_name)) as f:
        config = yaml.safe_load(f)
    return config

def get_document_cols(task_name):
    return tuple(get_config(task_name)["document_cols"])
