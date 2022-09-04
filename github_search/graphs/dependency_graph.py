from dataclasses import dataclass

import igraph
import pandas as pd
import toolz
import tqdm
from github_search import utils
from github_search.graphs import graph_preprocessor
from github_search.papers_with_code import repo_metadata


def prepare_dependency_gb(dependency_df):
    dependency_df = dependency_df[dependency_df["source"] != "<ROOT>"]
    dependency_without_leaf_functions_df = dependency_df[
        dependency_df["edge_type"] != "function-function"
    ]
    return dependency_without_leaf_functions_df.groupby("repo")


@dataclass
class DependencyDatasetBuilder:

    dependency_gb: pd.core.groupby.DataFrameGroupBy
    metadata: repo_metadata.RepoMetadata

    def get_repo_df(self, repo):
        return self.dependency_gb.get_group(repo)

    def make_graph_record(self, repo, encoding_fn):
        repo_df = self.get_repo_df(repo)
        repo_graph = self.make_repo_df_graph(repo_df)
        return graph_preprocessor.get_graph_data(
            repo_graph, repo, self.metadata(repo), encoding_fn
        )

    def make_repo_df_graph(self, repo_df):
        graph = igraph.Graph()
        vertices = pd.concat([repo_df["source"], repo_df["destination"]]).unique()
        graph.add_vertices(vertices, attributes={"text": vertices})
        graph.add_edges(repo_df[["source", "destination"]].to_records(index=False))
        return graph

    def get_graph_data_generator(self, repos, encoding_fn):
        pbar = tqdm.auto.tqdm(repos)
        for repo in pbar:
            try:
                graph_data = self.make_graph_record(repo, encoding_fn)
                yield graph_data
            except RuntimeError:
                print(f"failed for repo {repo}")
