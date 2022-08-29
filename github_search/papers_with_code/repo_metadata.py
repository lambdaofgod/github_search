import pandas as pd
from typing import Callable, Dict
from github_search import utils

RepoMetadata = Callable[[str], Dict[str, str]]


class RepoMetadataFromPandas:


    def __init__(self, paperswithcode_df: pd.DataFrame, area_tasks_df: pd.DataFrame):
        self.paperswithcode_df = paperswithcode_df
        self.area_tasks_df = area_tasks_df
        self._validate_dfs()
        self.mapping = self._prepare_mapping()

    def _validate_dfs(self):
        for col in ["least_common_task", "tasks", "repo"]:
            assert col in self.paperswithcode_df.columns
        for col in ["area", "task"]:
            assert col in self.area_tasks_df.columns

    def _prepare_mapping(self):
        return (
            self.paperswithcode_df[["least_common_task", "tasks", "repo"]]
            .merge(self.area_tasks_df, left_on="least_common_task", right_on="task")
            .drop_duplicates(subset="repo")
            .set_index("repo")[["least_common_task", "tasks", "area"]]
            .to_dict(orient="index")
        )

    @classmethod
    def load_from_files(cls, paperswithcode_path, area_tasks_path):
        paperswithcode_df = utils.load_paperswithcode_df(paperswithcode_path) 
        return cls(paperswithcode_df=paperswithcode_df, area_tasks_df=pd.read_csv(area_tasks_path))

    def __call__(self, repo):
        return self.mapping[repo]

    def repo_exists(self, repo):
        return self.mapping.get(repo) is not None
