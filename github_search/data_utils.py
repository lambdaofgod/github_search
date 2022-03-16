import logging

import fasttext
import pandas as pd
from mlutil.feature_extraction import embeddings

from github_search import pytorch_geometric_data


def get_repo_records(repos: pd.Series, dependency_records_df: pd.DataFrame):
    repo_dependency_records_df = dependency_records_df[
        dependency_records_df["edge_type"] == "repo-file"
    ]
    repo_dependencies = repo_dependency_records_df.groupby("source").apply(
        lambda df: " ".join(df["destination"]).replace(df["source"].iloc[0] + ":", "")
    )

    return repo_dependencies[repo_dependencies.index.isin(repos)]


def make_records_df(repos: pd.Series, repo_dependent_nodes: pd.Series) -> pd.DataFrame:
    """
    repos: series of repositories, for example "trangvu/ape-npi"
    repo_dependent_nodes: series of repos (index) with list of connected nodes (values)
    """
    return pd.DataFrame.from_records(
        [
            {
                "source": src,
                "destination": dst.replace(src + ":", ""),
                "edge_type": "repo-file",
            }
            for (src, destinations) in zip(repos, repo_dependent_nodes)
            for dst in destinations
        ]
    )


def make_extended_dependency_wrapper(
    repos: pd.Series,
    dependency_records_df: pd.DataFrame,
    fasttext_model: fasttext.FastText._FastText,
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
