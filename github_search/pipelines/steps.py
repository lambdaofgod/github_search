import ast
import json
import logging
from pathlib import Path as P
from typing import Dict, List, Optional, Tuple

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
from typing import Annotated
from zenml import pipeline, step
from operator import itemgetter


def _process_generated_records(generated_records):
    prompt_info_dicts = generated_records["prompt_info"].apply(dict)
    return pd.DataFrame(
        dict(
            repo=prompt_info_dicts.apply(itemgetter("name")),
            tasks=generated_records["generated_text"],
            true_tasks=prompt_info_dicts.apply(itemgetter("true_text")),
            generated_text=generated_records["generated_text"].apply(itemgetter(0)),
            prompt_info=prompt_info_dicts,
        )
    )


@step(enable_cache=True)
def expand_documents_step(
    text_generation_config: dict, prompt_config: dict, prompt_infos: List[dict]
) -> Tuple[
    Annotated[pd.DataFrame, "generated_texts_df"], Annotated[List[dict], "failures"]
]:
    from tgutil.prompting import ContextPromptInfo
    from tgutil.prompting_runner import DocumentExpander
    from tgutil.configs import load_config_from_dict, PromptConfig

    text_generation_config = load_config_from_dict(text_generation_config)
    prompt_config = PromptConfig(**prompt_config)
    parsed_prompt_infos = [ContextPromptInfo.parse_obj(pi) for pi in prompt_infos]
    raw_generated_texts_df, failures = DocumentExpander(
        text_generation_config=text_generation_config, prompts_config=prompt_config
    ).expand_documents(parsed_prompt_infos)
    generated_texts_df = _process_generated_records(raw_generated_texts_df)
    return generated_texts_df, failures


@step(enable_cache=True)
def sample_data_step(
    prompt_config: dict, sampling_config: dict
) -> Annotated[List[dict], "prompt_infos"]:
    from tgutil.prompting_runner import DocumentExpander
    from tgutil.configs import SamplingConfig, PromptConfig

    sampling_config = SamplingConfig(**sampling_config)
    prompt_config = PromptConfig(**prompt_config)
    return DocumentExpander.sample_data(prompt_config, sampling_config)


@step(enable_cache=True)
def evaluate_generated_texts(
    generated_texts_df: pd.DataFrame, paperswithcode_path: str
) -> Annotated[pd.DataFrame, "generation_metrics_df"]:
    from github_search.utils import load_paperswithcode_df
    from tgutil.evaluation.evaluators import (
        TextGenerationEvaluator,
    )
    from tgutil.evaluation.preprocessing import EvalDFPreprocessor

    repo_tasks_df = load_paperswithcode_df(paperswithcode_path)
    texts_df = EvalDFPreprocessor(
        id_col="repo", reference_text_col="true_tasks"
    ).get_eval_df_from_raw_generated_text(generated_texts_df, repo_tasks_df)
    return (
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
