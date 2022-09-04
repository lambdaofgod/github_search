import logging
import pickle
from typing import Callable, Dict, List, Union

import h5py
import igraph
import numpy as np
import pandas as pd
import sentence_transformers
import toolz
import torch
from fastai.text import all as text
from findkit import feature_extractor
from findkit.feature_extractor import FastAITextFeatureExtractor
from github_search.graphs import dependency_graph, graph_preprocessor, datasets
from github_search.papers_with_code import repo_metadata
from github_search.papers_with_code.repo_metadata import RepoMetadataFromPandas
from sklearn import model_selection
from torch_geometric import data as ptg_data

Label = Union[str, List[str]]


def get_graphs_train_test_split(graph_data_list, test_size):
    graph_labels = [g.label for g in graph_data_list]
    logging.info(pd.Series(graph_labels).value_counts().sort_values(ascending=False))
    return model_selection.train_test_split(
        graph_data_list,
        graph_labels,
        test_size=test_size,
        stratify=graph_labels,
    )


def prepare_dataset_from_graph_file(
    graph_path,
    paperswithcode_path,
    area_tasks_path,
    sentence_transformer_model_or_path,
    batch_size,
):
    logging.info("loading graph object")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    logging.info("loading embedder")
    extractor = feature_extractor.SentenceEncoderFeatureExtractor(
        sentence_transformers.SentenceTransformer(sentence_transformer_model_or_path)
    )
    get_metadata = repo_metadata.RepoMetadataFromPandas.load_from_files(
        paperswithcode_path, area_tasks_path
    )
    preprocessor = graph_preprocessor.GraphDataPreprocessor(
        extractor, get_metadata=get_metadata
    )
    logging.info("preparing graph records")
    graphs, graph_names = preprocessor.get_subgraphs_with_repo_vertices(graph)
    return preprocessor.get_graph_data_iter(graphs, graph_names, batch_size)


def prepare_dataset_with_transformer(
    upstream, product, sentence_transformer_model_name, paperswithcode_path, batch_size
):
    graph_path = str(upstream["graph.prepare_from_function_code"])
    area_tasks_path = str(upstream["prepare_area_grouped_tasks"])
    data_list = list(
        prepare_dataset_from_graph_file(
            graph_path,
            paperswithcode_path,
            area_tasks_path,
            sentence_transformer_model_name,
            batch_size,
        )
    )
    logging.info(f"loaded {len(data_list)} graphs")
    pickle.dump(data_list, open(str(graph_path), "wb"))


def prepare_dataset_with_word2vec(upstream, product, batch_size, paperswithcode_path):
    graph_path = str(upstream["graph.prepare_from_function_code"])
    area_tasks_path = str(upstream["prepare_area_grouped_tasks"])
    data_list = list(
        prepare_dataset_from_graph_file(
            graph_path,
            paperswithcode_path,
            area_tasks_path,
            str(upstream["sentence_embeddings.prepare_w2v_model"]),
            batch_size,
        )
    )
    logging.info(f"loaded {len(data_list)} graphs")
    pickle.dump(data_list, open(str(graph_path), "wb"))


def prepare_dataset_with_rnn(
    upstream, product, paperswithcode_path, ulmfit_path
):
    dependency_df_path = str(upstream["dependency_graph.prepare_records"])
    dependency_gb = dependency_graph.prepare_dependency_gb(
        pd.read_feather(dependency_df_path)
    )
    area_tasks_path = str(upstream["prepare_area_grouped_tasks"])
    metadata = repo_metadata.RepoMetadataFromPandas.load_from_files(
        paperswithcode_path, area_tasks_path
    )
    fastai_learner = text.load_learner(ulmfit_path)
    fastai_extractor = FastAITextFeatureExtractor.build_from_learner(
        fastai_learner, max_length=48
    )
    encoding_fn = toolz.partial(fastai_extractor.extract_features, show_progbar=False)
    builder = dependency_graph.DependencyDatasetBuilder(dependency_gb, metadata)
    all_repos = dependency_gb.groups.keys()
    repos = [repo for repo in all_repos if metadata.repo_exists(repo)]
    data_iter = builder.get_graph_data_generator(repos, encoding_fn)
    with h5py.File(str(product), "w") as f:
        datasets.write_graph_data_iter_to_h5_file(f, data_iter)


def prepare_dataset_split(upstream, product):
    repos_train = pd.read_csv(str(upstream["prepare_repo_train_test_split"]["train"]))
    with open(str(upstream["gnn.prepare_dataset_with_w2v"]), "rb") as f:
        graph_data_list = pickle.load(f)
    train_graph_list = [
        g for g in graph_data_list if g.graph_name in repos_train["repo"].values
    ]
    test_graph_list = [
        g for g in graph_data_list if g.graph_name not in repos_train["repo"].values
    ]
    pickle.dump(train_graph_list, open(str(product["train"]), "wb"))
    pickle.dump(test_graph_list, open(str(product["test"]), "wb"))
