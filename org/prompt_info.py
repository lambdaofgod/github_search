from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
import sys
import re
from pathlib import Path

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

base_prompt = """
repository {}
contains files {}
its tags are {}
"""


@dataclass
class PromptInfo:
    """
    information about sample repositories passed to prompt
    """

    repo_records: List[dict]
    predicted_repo_record: dict
    repo_text_field: str = field(default="dependencies")

    def get_single_prompt(self, record):
        repo = record["repo"]
        dependencies = ", ".join(record[self.repo_text_field])
        tasks = record["tasks"]
        return base_prompt.format(repo, dependencies, tasks)

    def format_prompt(self, prompt_template):
        n_placeholders = re.findall(r"\{\}", prompt_template)
        assert n_placeholders == len(self.repo_records) + 1
        prefix_prompt = "\n".join(
            self.get_single_prompt(record) for record in self.repo_records
        )
        (
            other_repo_name,
            other_repo_filenames,
            other_repo_tasks,
        ) = self.predicted_repo_record.values()
        other_repo_filenames = [P(fname).name for fname in other_repo_filenames]
        return (
            prefix_prompt
            + f"\nrepository {other_repo_name}\n"
            + f"contains files: {', '.join(other_repo_filenames)}\n"
            + "tags: "
        )

    @classmethod
    def from_df(cls, data_df, pos_indices, pred_index, n_deps=10):
        return PromptInfo(
            get_repo_records_by_index(data_df, pos_indices, n_deps=n_deps),
            get_repo_records_by_index(data_df, [pred_index], n_deps=n_deps)[0],
        )

    def get_repo_args(self, record, use_tasks=True):
        tasks = "[{}]".format(", ".join(record["tasks"])) if use_tasks else ""
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
