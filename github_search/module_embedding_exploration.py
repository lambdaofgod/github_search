# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/Import2Vec embeddings exploration.ipynb (unless otherwise specified).

__all__ = [
    "make_vizualization_df",
    "default_dimensionality_reducer",
    "reference_modules",
    "show_scatterplot",
]

# Cell

import os

import altair
import gensim
import matplotlib.pyplot as plt
import neptune
import pandas as pd
import umap

from github_search import neptune_util

# Cell


default_dimensionality_reducer = umap.UMAP(n_neighbors=20, metric="cosine")


def make_vizualization_df(
    keyed_vectors, dimensionality_reducer=default_dimensionality_reducer
):
    vectors = import2vec.syn0

    umap_vectors = dimensionality_reducer.fit_transform(vectors)
    viz_df = pd.DataFrame(umap_vectors)
    viz_df.columns = ["x", "y"]
    viz_df["name"] = import2vec.vocab.keys()
    return viz_df


# Cell


reference_modules = [
    "numpy",
    "tensorflow",
    "keras",
    "sklearn",
    "scipy",
    "matplotlib",
    "torch",
    "os",
    "sys",
    "seaborn",
]

# Cell


def show_scatterplot(viz_df, reference_modules=reference_modules):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(viz_df["x"], viz_df["y"], alpha=0.2)

    for module_name in reference_modules:
        module_row = viz_df[viz_df["name"] == module_name].iloc[0]
        ax.scatter([module_row["x"]], [module_row["y"]], c="red")
        ax.annotate(module_name, (module_row["x"], module_row["y"]))
