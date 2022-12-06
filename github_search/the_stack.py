import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

import huggingface_hub
import pandas as pd
import tqdm

from github_search import utils

logging.basicConfig(level="INFO")


@dataclass
class TheStackPapersWithcodeDownloader:

    repos: Iterable[str]
    save_dir: str = field(default="data/the_stack_paperswithcode_repos")
    recreate_existing: bool = field(default=False)
    delete_temporary_files: bool = field(default=True)
    max_idx: int = 206

    def prepare_dfs(self, max_idx=None, min_idx=None):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        min_idx = 0 if min_idx is None else min_idx
        max_idx = self.max_idx if max_idx is None else max_idx
        idx_range = range(min_idx, max_idx)
        for idx in tqdm.tqdm(idx_range):
            self.prepare_file(idx)

    def filter_df(self, repos, ith_stack_df):
        repos_in_pwc = ith_stack_df["max_stars_repo_name"].isin(repos)
        return ith_stack_df[repos_in_pwc]

    def prepare_file(self, file_idx):
        filename = self.get_filename(file_idx)
        out_filepath = os.path.join(
            self.save_dir, "filtered_" + os.path.basename(filename)
        )
        if os.path.exists(out_filepath):
            logging.info(f"file {out_filepath} already exists, skipping")
            return
        logging.info(f"not found {out_filepath}")
        filepath = huggingface_hub.hf_hub_download(
            repo_id="bigcode/the-stack", filename=filename, repo_type="dataset"
        )
        ith_stack_df = pd.read_parquet(filepath)
        df = self.filter_df(self.repos, ith_stack_df)
        df.to_parquet(out_filepath)
        if self.delete_temporary_files:
            os.remove(filepath)

    @classmethod
    def get_filename(cls, file_idx):
        zero_str = "0" * (5 - len(str(file_idx)))

        return f"data/python/train-{zero_str}{str(file_idx)}-of-00206.parquet"


def prepare_the_stack_files(paperswithcode_path, delete_temporary_files, product):
    paperswithcode_df = utils.load_paperswithcode_df(paperswithcode_path)
    repos = set(paperswithcode_df["repo"])
    downloader = TheStackPapersWithcodeDownloader(repos, save_dir=str(product))
    downloader.prepare_dfs()
