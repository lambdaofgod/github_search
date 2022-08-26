
import igraph
from typing import Callable, Dict, List, Union
from findkit import feature_extractor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import tqdm
from torch_geometric import data as ptg_data


Label = Union[str, List[str]]


def make_trivial_graph(
    vertex_texts, vertex_encoder: feature_extractor.SentenceEncoderFeatureExtractor
):
    return ptg_data.Data(
        torch.Tensor(
            vertex_encoder.extract_features(vertex_texts, show_progress_bar=False)
        ),
        edge_index=torch.LongTensor([[i, i] for i in range(len(vertex_texts))]).T,
        names=vertex_texts,
    )


def make_igraph(call_records_df):
    vertices = list(
        set(call_records_df["source"].unique()).union(
            call_records_df["destination"].unique()
        )
    )
    edges = [
        (source, destination)
        for (source, destination) in zip(
            call_records_df["source"].values, call_records_df["destination"].values
        )
    ]
    graph = igraph.Graph()
    graph.add_vertices(vertices)
    graph.add_edges(edges)
    return graph


def get_vertex_name(vertex):
    return vertex.attributes()["name"]


def get_induced_subgraph(graph: igraph.Graph, start_vertex_name: str):
    repo_vertex = graph.vs.find(name=repo)
    file_neighbors = set(
        [
            vertex
            for vertex in repo_vertex.neighbors()
            if not vertex.attributes()["name"] == "<ROOT>"
        ]
    )
    function_edges = set(
        [(file, function) for file in file_neighbors for function in file.neighbors()]
    )
    file_edges = [(repo, get_vertex_name(file)) for file in file_neighbors]
    function_edges = set(
        [
            (get_vertex_name(file), get_vertex_name(function))
            for (file, function) in function_edges
        ]
    )
    return pd.DataFrame.from_records(
        list(function_edges.union(file_edges)), columns=["source", "destination"]
    )


def get_repo_vertices(graph, paperswithcode_df):
    vertex_names = (get_vertex_name(v) for v in graph.vs)
    return set(vertex_names).intersection(set(paperswithcode_df["repo"].values))


def get_graph_data(
    graph: igraph.Graph,
    graph_name: str,
    metadata: Dict[str, Label],
    encoding_method: Callable[[List[str]], np.ndarray],
    **kwargs,
):
    texts = graph.vs.get_attribute_values("text")
    names = graph.vs.get_attribute_values("name")
    edge_index = torch.tensor(graph.get_edgelist()).T
    features = torch.tensor(encoding_method(texts, **kwargs))
    return ptg_data.Data(
        features, edge_index, names=names, graph_name=graph_name, **metadata
    )




@dataclass
class GraphDataPreprocessor:
    extractor: feature_extractor.SentenceEncoderFeatureExtractor
    metadata: Dict[str, Dict[str, str]]

    def get_data_list(self, graph: igraph.Graph, show_progress_bar=True, **kwargs):
        subgraphs = (
            graph.induced_subgraph(connected_component_vertices)
            for connected_component_vertices in graph.components(mode="weak")
        )
        subgraphs_with_repo_vertices = list(
            (subgraph, self.get_repo_vertex(subgraph)) for subgraph in subgraphs
        )
        if show_progress_bar:
            subgraphs_with_repo_vertices = tqdm.auto.tqdm(subgraphs_with_repo_vertices)
        return [
            get_graph_data(
                subgraph,
                repo_vertex,
                self.metadata[repo_vertex],
                encoding_method=self.extractor.extract_features,
                show_progress_bar=False,
                **kwargs,
            )
            for (subgraph, repo_vertex) in subgraphs_with_repo_vertices
            if repo_vertex in self.metadata.keys()
        ]

    def get_repo_vertex(self, subgraph):
        names = subgraph.vs.get_attribute_values("name")
        repo_vertices = [n for n in names if len(n.split(":")) == 1]
        assert (
            len(repo_vertices) == 1
        ), f"there should be only one repo vertex per subgraph: {names}"
        return repo_vertices[0]

