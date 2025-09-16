import logging

import pandas as pd
from mlutil.feature_extraction import embeddings

from github_search import pytorch_geometric_data

try:
    import fasttext

    FastTextModel = fasttext.FastText._FastText
except ImportError:
    from typing import Any

    FastTextModel = Any


def make_extended_dependency_wrapper(
    repos: pd.Series,
    dependency_records_df: pd.DataFrame,
    fasttext_model: FastTextModel,
) -> pytorch_geometric_data.PygGraphWrapper:
    fasttext_embedder = embeddings.FastTextVectorizer(fasttext_model)

    logging.info("creating dependency nodes")
    repo_dependent_nodes = get_repo_records(repos, dependency_records_df)

    logging.info("loading dependency records")
    non_root_dependency_records_df = dependency_records_df[
        (dependency_records_df["source"] != "<ROOT>")
        & (dependency_records_df["edge_type"] != "repo-repo")
    ]

    logging.info("creating dependency dataframe")
    other_records_df = make_records_df(repos, repo_dependent_nodes.dropna())
    dep_graph_df = pd.concat([non_root_dependency_records_df, other_records_df])

    logging.info("creating dependency graph wrapper")
    return pytorch_geometric_data.PygGraphWrapper(fasttext_embedder, dep_graph_df)
