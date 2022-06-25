import logging
import pickle
from dataclasses import dataclass

import fasttext
import igraph
import pandas as pd
import tqdm
from mlutil.feature_extraction import embeddings
from sklearn import base, model_selection, preprocessing

from github_search import pytorch_geometric_data

logging.basicConfig(level="INFO")


def get_vertex_name(vertex):
    return vertex.attributes()["name"]


def get_dependency_graph_transitive_closure(graph: igraph.Graph, repo: str):
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


@dataclass
class GraphDataPreprocessor:

    graph: igraph.Graph
    embedder: base.TransformerMixin

    def get_data_list(self, start_vertices, names, labels):
        assert len(start_vertices) == len(labels) and len(names) == len(
            labels
        ), "labels do not match data shape"
        return [
            pytorch_geometric_data.PygGraphWrapper(
                self.embedder,
                get_dependency_graph_transitive_closure(self.graph, vertex),
                label=label,
                name=name,
            ).dataset
            for (vertex, name, label) in tqdm.auto.tqdm(
                zip(start_vertices, names, labels), total=len(start_vertices)
            )
        ]


def get_graph_transitive_closure(graph, repo):
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


def get_repos_with_labels_split(area_tasks_df, paperswithcode_df, graph, test_size):
    repos_in_graph = get_repo_vertices(graph, paperswithcode_df)
    repo_tasks = paperswithcode_df[["repo", "least_common_task"]].set_index("repo")
    repo_areas = repo_tasks["least_common_task"].apply(
        lambda t: area_tasks_df.loc[[t], "area"].iloc[0]
        if t in area_tasks_df.index
        else "other"
    )
    repo_areas_in_graph = repo_areas.loc[repos_in_graph]
    areas_labels = pd.Series(
        index=repo_areas_in_graph.index,
        data=preprocessing.LabelEncoder().fit_transform(repo_areas_in_graph),
    )
    return model_selection.train_test_split(
        repo_areas_in_graph.index, areas_labels, test_size=test_size, stratify=areas_labels
    )


def prepare_datasets(upstream, product, test_size=0.2):
    fasttext_model_path = str(upstream["train_python_token_fasttext"])
    graph_path = str(upstream["make_igraph"])
    train_path = str(product["train"])
    test_path = str(product["test"])
    area_tasks_path = str(upstream["prepare_area_grouped_tasks"])

    logging.info("loading graph object")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    area_tasks_df = pd.read_csv(area_tasks_path).set_index("task")
    paperswithcode_df = pd.read_csv("data/paperswithcode_with_tasks.csv")
    logging.info("preparing graph records")
    (
        repos_train,
        repos_test,
        labels_train,
        labels_test,
    ) = get_repos_with_labels_split(area_tasks_df, paperswithcode_df, graph, test_size)
    logging.info("loading fasttext embedder")
    fasttext_embedder = embeddings.FastTextVectorizer(
        fasttext.load_model(fasttext_model_path)
    )
    graph_preprocessor = GraphDataPreprocessor(graph, fasttext_embedder)

    logging.info("preparing train data")
    train_data_list = graph_preprocessor.get_data_list(
        repos_train, names=repos_train, labels=labels_train
    )
    logging.info("preparing test data")
    test_data_list = graph_preprocessor.get_data_list(
        repos_test, names=repos_test, labels=labels_test
    )
    pickle.dump(train_data_list, open(train_path, "wb"))
    pickle.dump(test_data_list, open(test_path, "wb"))


