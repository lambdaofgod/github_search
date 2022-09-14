from dataclasses import dataclass

import torch_geometric
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

    def make_graph_record(self, repo, encoding_fn) -> torch_geometric.data.Data:
        repo_df = self.get_repo_df(repo)
        repo_graph = self.make_repo_df_graph(repo_df)
        return graph_preprocessor.get_graph_data(
            repo_graph, repo, self.metadata(repo), encoding_fn
        )

    def make_repo_df_graph(self, repo_df):
        repo_df = self.clean_repo_df(repo_df.copy())
        graph = igraph.Graph()
        [repo] = list(repo_df["repo"].unique())
        vertices = pd.concat([repo_df["source"], repo_df["destination"]]).unique()
        repo_idx = self._get_element_index(vertices, repo)
        graph.add_vertices(vertices, attributes={"text": vertices})
        graph.add_edges(repo_df[["source", "destination"]].to_records(index=False))
        return self._get_subgraph(graph, repo_idx)

    def get_graph_data_generator(self, repos, encoding_fn):
        pbar = tqdm.auto.tqdm(repos)
        for repo in pbar:
            try:
                graph_data = self.make_graph_record(repo, encoding_fn)
                yield graph_data
            except RuntimeError:
                print(f"failed for repo {repo}")

    def clean_repo_df(self, repo_df):
        repo_df["source"] = repo_df["source"].str.replace("\.py$", "")
        repo_df["repo"] = repo_df["repo"].str.replace("\.py$", "")
        return repo_df

    @classmethod
    def _get_subgraph(cls, g, node_idx):
        components = g.components()
        [component] = [comp for comp in components if node_idx in comp]
        return g.subgraph(component)

    @classmethod
    def _get_element_index(cls, series, elem):
        return list(series).index(elem)
