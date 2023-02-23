#!/usr/bin/env python
# coding: utf-8

# %%


import re
import os
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
from github_search.ir.models import *
from github_search.ir import EmbedderPairConfig, InformationRetrievalColumnConfig

from collections import Counter

import yaml
import json
import pprint
from github_search.neural_bag_of_words.model_utils import *
import dataclasses
import shutil

# %% [markdown]
# ## Helpers

tasks_path = "../../data/paperswithcode_tasks.csv"
product = "../../output/best_nbow/"


# %%
def get_document_cols(model_path):
    model_path_cols = model_path.name.replace("nbow_", "").split("-")[0].split("#")
    col_mapping = {"signatures": "function_signature"}
    return tuple([col_mapping.get(col, col) for col in model_path_cols])


def prepare_display_df(metrics_df):
    """
    sort values by our most important metric
    """
    metrics_df["document_cols"] = (
        pd.Series(metrics_df.index).apply(get_document_cols).values
    )
    return metrics_df.sort_values("accuracy@10", ascending=False)


def get_eval_df(upstream_paths):
    """
    get evaluation result yaml files
    """
    metrics_df = evaluation.get_metrics_df_from_dicts(
        dict(
            evaluation.get_metrics_dict_with_name(
                p / "metrics_final.yaml", get_name=lambda arg: p
            )
            for p in upstream_paths
        )
    )
    return metrics_df


def get_best_model_config(best_model_directory):
    return EmbedderPairConfig(
        query_embedder_path=best_model_directory / "nbow_query_final",
        document_embedder_path=best_model_directory / "nbow_document_final",
    )


def round_float_dict(d, rounding=3):
    if type(d) is dict:
        return {k: round_float_dict(v) for k, v in d.items()}
    else:
        return float(round(d, rounding))


def setup_ir_evaluator_from_config(
    searched_dep_texts: pd.DataFrame, config: EmbedderPairConfig, embedded_cols: str
):
    embedder_pair = EmbedderPair.from_config(config)
    ir_evaluator = evaluator.InformationRetrievalEvaluator(embedder_pair)
    ir_evaluator.setup(searched_dep_texts, "tasks", embedded_col, name_col="repo")
    return ir_evaluator


def get_grouped_model_results(eval_df):
    grouped_best_model_names = eval_df.groupby("document_cols").apply(
        lambda df: df.reset_index().sort_values("accuracy@10").iloc[-1]
    )["name"]
    grouped_results_dict = {}
    results_lists_dict = {}
    for (document_cols, best_model_directory) in zip(
        grouped_best_model_names.index, grouped_best_model_names.values
    ):
        best_model_config = get_best_model_config(best_model_directory)
        ir_config = evaluator.InformationRetrievalEvaluatorConfig(
            search_data_path,
            best_model_config,
            InformationRetrievalColumnConfig(
                document_cols=list(get_document_cols(best_model_directory)),
                query_col="tasks",
                list_cols=["titles"],
            ),
        )
        ir_evaluator = evaluator.InformationRetrievalEvaluator.setup_from_config(
            ir_config
        )
        ir_results_list = ir_evaluator.get_results_list()
        ir_metrics = ir_evaluator.evaluate()
        ir_metrics_yaml = yaml.dump(ir_metrics["cos_sim"])
        print(document_cols)
        print(best_model_directory.name)
        print(ir_metrics_yaml)
        grouped_results_dict[(document_cols, best_model_directory)] = ir_metrics
        results_lists_dict[(document_cols, best_model_directory)] = ir_results_list
    return grouped_results_dict, results_lists_dict


# %% tags=["parameters"]

product = None
upstream = dict()
document_cols = ["dependencies"]

# %%
tasks_path = str(upstream["prepare_area_grouped_tasks"])
upstream_paths = [P(v["model_dir"]) for v in upstream["nbow.train-*-*-*"].values()]
search_data_path = str(upstream["nbow.prepare_dataset"]["test"])
out_dir = P(product["best_model_dir"])

# %% [markdown]

# ## Combining evaluation results

# %%

eval_df = prepare_display_df(get_eval_df(upstream_paths))
eval_df.style.highlight_max()

# %%

grouped_model_results, results_lists_dict = get_grouped_model_results(eval_df)

# %% [markdown]
# ## Results

# %%

# %%
with open(out_dir / "nbow_results.yaml", "w") as f:
    f.write(yaml.dump(grouped_model_results))

for k, val in results_lists_dict.items():
    with open(pathlib.Path("/tmp") / "_".join(k) + ".pkl", "wb") as f:
        pickle.dump(val, f)

# %% [markdown]

# ## Best/worst performing test queries

# %%


task_df = pd.read_csv(tasks_path)
predictor = evaluator.InformationRetrievalPredictor(ir_evaluator, "repo")
predicted_results = predictor.get_predicted_documents()
best_worst_results = predictor.get_best_worst_results_df(task_df)

# %%

best_worst_results["best_tasks"].head()

# %%

best_worst_results["worst_tasks"].head()

# %% [markdown]
# ### Copying files

# %%

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir()

shutil.copytree(best_model_config.query_embedder_path, out_dir / "query_embedder")
shutil.copytree(best_model_config.document_embedder_path, out_dir / "document_embedder")

for (item_name, item) in best_worst_results.items():
    item.to_csv(os.path.join(out_dir, f"{item_name}.csv"))
