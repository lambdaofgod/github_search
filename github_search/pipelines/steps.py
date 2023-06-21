import ast
import json
import logging
from pathlib import Path as P
from typing import Dict, List, Optional

import yaml
from github_search.ir import InformationRetrievalEvaluatorConfig
import clearml
import fire
import jsonlines
import numpy as np
import pandas as pd
import tqdm
from clearml import Dataset, PipelineController, Task
from mlutil.text import rwkv_utils
from pydantic.dataclasses import dataclass
import pandera as pa
from tgutil.configs import (
    PipelineConfig,
    SamplingConfig,
    TextGenerationConfig,
    PromptConfig,
)
from tgutil.prompting_runner import sample_data, expand_documents


def expand_documents_step(
    text_generation_config: dict, prompt_config: dict, prompt_infos: List[dict]
):
    from tgutil.prompting_runner import expand_documents
    from tgutil.configs import load_config_from_dict, PromptConfig
    from tgutil.prompting import PromptInfo

    text_generation_config = load_config_from_dict(text_generation_config)
    prompt_config = PromptConfig(**prompt_config)
    return expand_documents(text_generation_config, prompt_config, prompt_infos)


def sample_data_step(sampling_config: dict):
    from tgutil.prompting_runner import sample_data
    from tgutil.configs import SamplingConfig

    sampling_config = SamplingConfig(**sampling_config)
    return sample_data(sampling_config)


@pa.check_input(
    pa.DataFrameSchema(
        {
            "true_tasks": pa.Column(List[str]),
            "generated_text": pa.Column(str),
            "repo": pa.Column(str),
        }
    )
)
def evaluate_generated_texts(generated_texts_df, paperswithcode_path):
    from github_search.utils import load_paperswithcode_df
    from tgutil.evaluation.evaluators import (
        TextGenerationEvaluator,
    )
    from tgutil.evaluation.preprocessing import EvalDFPreprocessor

    repo_tasks_df = load_paperswithcode_df(paperswithcode_path)
    texts_df = EvalDFPreprocessor(
        id_col="repo", reference_text_col="true_tasks"
    ).get_eval_df_from_raw_generated_text(generated_texts_df, repo_tasks_df)
    eval_df = (
        TextGenerationEvaluator.from_metric_names(
            metric_names=[
                "edit_word",
                "jaccard_lst",
                "bleurt",
                "rouge",
                "wmd",
                "sentence_transformer_similarity",
            ]
        )
        .get_evaluated_df(texts_df=texts_df)
        .sort_values(by="rougeL", ascending=False)
    )
    return eval_df


def evaluate_information_retrieval_step(
    searched_df, ir_config: InformationRetrievalEvaluatorConfig
):
    from github_search.ir.evaluator import InformationRetrievalEvaluator

    ir_evaluator = InformationRetrievalEvaluator.setup_from_df(searched_df, ir_config)
    return ir_evaluator.evaluate()


def evaluate_generated_texts_step(generated_texts_df, paperswithcode_path):
    from github_search.pipelines.steps import evaluate_generated_texts

    generation_evaluation_df = evaluate_generated_texts(
        generated_texts_df, paperswithcode_path
    )
    return generation_evaluation_df
