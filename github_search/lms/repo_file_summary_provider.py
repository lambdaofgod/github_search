import abc
from typing import List, Dict
from pydantic import BaseModel


class RepoFileSummaryProvider(abc.ABC):
    def extract_summary(self, repo_name) -> str:
        pass

    def get_filenames(self, repo_name) -> List[str]:
        pass


class DataFrameRepoFileSummaryProvider(RepoFileSummaryProvider):
    def __init__(self, df, files_per_repo, code_col):
        self.df = df
        self.files_per_repo = files_per_repo
        self.code_col = code_col

    def get_filenames(self, repo_name):
        return self.df[self.df["repo_name"] == repo_name].iloc[: self.files_per_repo][
            "path"
        ]

    def extract_summary(self, repo_name):
        selected_python_files = self.df[self.df["repo_name"] == repo_name].iloc[
            : self.files_per_repo
        ]
        return "\n\n".join(
            [
                f"file {path}\n```\n{code}\n```"
                for path, code in zip(
                    selected_python_files["path"],
                    selected_python_files[self.code_col],
                )
            ]
        )


class RepoMapProvider(RepoFileSummaryProvider, BaseModel):
    repo_maps: Dict[str, str]

    def get_filenames(self, repo_name):
        return self.repo_maps[repo_name]

    def extract_summary(self, repo_name):
        return self.repo_maps[repo_name]
