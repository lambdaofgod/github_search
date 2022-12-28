import re
import shutil
from pathlib import Path as P
import sentence_transformers
import yaml
from findkit import index, feature_extractor
from github_search.ir import ir_utils, evaluator
from typing import List, Dict, Set
import tqdm
import ast

from github_search import utils
from github_search.papers_with_code import paperswithcode_tasks
from github_search.neural_bag_of_words import embedders
from github_search.neural_bag_of_words import evaluation
import sentence_transformers
from github_search.neural_bag_of_words.evaluation import *

from collections import Counter

import yaml
import json
import pprint
from github_search.neural_bag_of_words.model_utils import *
import dataclasses


def prepare_display_df(metrics_df):
    """
    sort values by our most important metric
    """
    return metrics_df.sort_values("accuracy@10", ascending=False)


def get_eval_df(upstream_paths):
    """
    get evaluation result yaml files
    """
    return evaluation.get_metrics_df_from_dicts(
        dict(
            evaluation.get_metrics_dict_with_name(
                p / "metrics_final.yaml", get_name=lambda arg: p
            )
            for p in upstream_paths
        )
    )


def get_best_model_config(best_model_directory):
    return EmbedderPairConfig(
        query_embedder_path=best_model_directory / "nbow_query_final",
        document_embedder_path=best_model_directory / "nbow_document_final",
    )


def get_per_query_hits_at_k(predicted_documents, relevant_docs, k=10):
    """
    compare predicted documents to gold standards relevant docs
    """
    return {
        q: int(
            k
            * predicted_results[q]
            .sort_values("score", ascending=False)
            .iloc[:k]["corpus_id"]
            .isin(relevant_docs[q_id])
            .mean()
        )
        for (q_id, q) in queries.items()
    }


def round_float_dict(d, rounding=3):
    if type(d) is dict:
        return {k: round_float_dict(v) for k, v in d.items()}
    else:
        return float(round(d, rounding))


def get_extremal_tasks_with_hits(
    raw_tasks_with_hits_df, get_best=True, n_tasks_per_area=10
):
    """
    get best/worst performing tasks according to hits metric
    """
    tasks_with_hits_df = raw_tasks_with_hits_df.groupby("area").apply(
        lambda df: df.sort_values("hits", ascending=not get_best).iloc[
            :n_tasks_per_area
        ]
    )
    tasks_with_hits_df.index = tasks_with_hits_df.index.get_level_values("area")
    tasks_with_hits_df = tasks_with_hits_df.drop(columns="area")
    return tasks_with_hits_df


def setup_ir_evaluator_from_config(
    searched_dep_texts: pd.DataFrame, config: EmbedderPairConfig
):
    embedder_pair = EmbedderPair.from_config(config)
    ir_evaluator = evaluator.InformationRetrievalEvaluator(embedder_pair)
    ir_evaluator.setup(searched_dep_texts, "tasks", "dependencies", name_col="repo")
    return ir_evaluator
