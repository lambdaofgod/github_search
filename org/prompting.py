from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
import sys
import re
from pathlib import Path
import ast
from typing import List
from pathlib import Path as P
from pydantic import BaseModel, Field


def preprocess_dep(dep):
    return P(dep).name


def select_deps(deps, n_deps):
    return [preprocess_dep(dep) for dep in deps if not "__init__" in dep][:n_deps]


def get_repo_records_by_index(
    data_df, indices, fields=["repo", "dependencies", "tasks"], n_deps=10
):
    records_df = data_df.iloc[indices].copy()
    raw_deps = records_df["dependencies"].str.split()
    records_df["dependencies"] = raw_deps.apply(lambda dep: select_deps(dep, n_deps))
    return records_df[fields].to_dict(orient="records")


# sys.path.insert(0, str(Path("/home/kuba/Projects/forks/GPTQ-for-LLaMa")))
# import llama


def get_n_tokens(text):
    ids = tokenizer(text)["input_ids"]
    return len(ids)


import pandas as pd

train_nbow_df = pd.read_parquet("../output/nbow_data_train.parquet").drop(
    ["count"], axis=1
)
train_nbow_df.head()


def preprocess_dep(dep):
    return P(dep).name


def select_deps(deps, n_deps):
    return [preprocess_dep(dep) for dep in deps if not "__init__" in dep][:n_deps]


def get_repo_records_by_index(
    data_df, indices, fields=["repo", "dependencies", "tasks"], n_deps=10
):
    records_df = data_df.iloc[indices].copy()
    raw_deps = records_df["dependencies"].str.split()
    records_df["dependencies"] = raw_deps.apply(lambda dep: select_deps(dep, n_deps))
    return records_df[fields].to_dict(orient="records")


from typing import List
from pathlib import Path as P


def get_prompt_template(repo_prompt, prefix="", n_repos=2):
    return "\n\n".join([prefix] + [repo_prompt.strip()] * (n_repos + 1)).strip()


class PromptInfo(BaseModel):
    """
    information about sample repositories passed to prompt
    """

    repo_records: List[dict]
    predicted_repo_record: dict
    repo_text_field: str = Field(default="dependencies")

    def format_prompt(self, prompt_template):
        n_placeholders = len(re.findall(r"\{\}", prompt_template))
        expected_n_placeholders = 3 * (len(self.repo_records) + 1)
        assert (
            n_placeholders == expected_n_placeholders
        ), f"unexpected placeholders: {n_placeholders}"
        return prompt_template.format(*self.get_prompt_args())

    @classmethod
    def from_df(cls, data_df, pos_indices, pred_index, n_deps=10):
        return PromptInfo(
            repo_records=get_repo_records_by_index(data_df, pos_indices, n_deps=n_deps),
            predicted_repo_record=get_repo_records_by_index(
                data_df, [pred_index], n_deps=n_deps
            )[0],
        )

    def get_repo_args(self, record, use_tasks=True):
        record_tasks = record["tasks"]
        if type(record_tasks) is str:
            record_tasks = ast.literal_eval(record_tasks)
        tasks = "[{}]".format(", ".join(record_tasks)) if use_tasks else ""
        return [record["repo"], ", ".join(record[self.repo_text_field]), tasks]

    def get_prompt_args(self):
        repo_with_tags_args = [
            self.get_repo_args(record) for record in self.repo_records
        ]
        predicted_repo_args = self.get_repo_args(
            self.predicted_repo_record, use_tasks=False
        )
        return [
            arg
            for rec_args in repo_with_tags_args + [predicted_repo_args]
            for arg in rec_args
        ]

    def get_promptify_input_dict(self):
        args_dict = {}
        args_dict["input_repo_info"] = [
            (rec["repo"], rec["dependencies"], rec["tasks"])
            for rec in self.repo_records
        ]
        args_dict["predicted_repo"] = self.predicted_repo_record["repo"]
        args_dict["predicted_files"] = self.predicted_repo_record["dependencies"]
        return args_dict
