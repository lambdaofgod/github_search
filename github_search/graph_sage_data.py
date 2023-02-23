import os

import igraph
import numpy as np
import pandas as pd
import sentence_transformers
import torch
from fastai.text import all as fastai_text
from findkit.feature_extractor import (
    fastai_feature_extractor,
    SentenceEncoderFeatureExtractor,
)
from scipy import sparse
from toolz import partial

from github_search import pytorch_geometric_data
from torch_geometric import utils


def make_graphsage_data(upstream, ulmfit_path, product, use_basename=False):

    dep_records_df = pd.read_feather(str(upstream["dependency_graph.prepare_records"]))
    dep_records_df = dep_records_df[dep_records_df["source"] != "<ROOT>"]
    dep_records_df["source"] = dep_records_df["source"].str.replace("\.py$", "")
    dep_records_df["destination"] = dep_records_df["destination"].str.replace(
        "\.py$", ""
    )
    if use_basename:
        dep_records_df["source"] = dep_records_df["source"].apply(os.path.basename)
        dep_records_df["destination"] = dep_records_df["destination"].apply(
            os.path.basename
        )
    fastai_learner = fastai_text.load_learner(ulmfit_path)
    embedder = fastai_feature_extractor.FastAITextFeatureExtractor.build_from_learner(
        fastai_learner, max_length=48
    )

    graph_maker = pytorch_geometric_data.PygGraphWrapper(
        partial(embedder.extract_features, show_progress_bar=True),
        dep_records_df,
        "source",
        "destination",
    )

    graph_maker.dataset.x = graph_maker.dataset.x.float()
    #graph_maker.dataset.edge_index = utils.to_undirected(graph_maker.dataset.edge_index)
    torch.save(graph_maker.dataset, str(product))
