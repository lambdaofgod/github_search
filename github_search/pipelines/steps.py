from typing import Dict, List, Optional, Tuple, Any

import ast
from github_search.ir import InformationRetrievalEvaluatorConfig
from github_search.pipelines.configs import EvaluationConfig, SearchDataConfig
import pandas as pd
from typing import Annotated
from zenml import step
from operator import itemgetter
from tgutil.prompting import ContextPromptInfo
from tgutil.prompting_runner import DocumentExpander
from tgutil.configs import load_config_from_dict, PromptConfig
from github_search import utils
from github_search.pipelines.metrics_comparison import *
from github_search.pipelines.postprocessing import GenerationPostprocessor


@step(enable_cache=False)
def expand_documents_step(
    text_generation_config: dict, prompt_config: dict, prompt_infos: List[dict]
) -> Tuple[
    Annotated[pd.DataFrame, "raw_generated_texts_df"], Annotated[List[dict], "failures"]
]:
    logging.info("expanding documents")
    logging.info(f"using text generation config: {text_generation_config}")
    logging.info(f"using prompt config: {prompt_config}")
    text_generation_config = load_config_from_dict(text_generation_config)
    prompt_config = PromptConfig(**prompt_config)
    parsed_prompt_infos = [ContextPromptInfo.parse_obj(pi) for pi in prompt_infos]
    raw_generated_texts_df, failures = DocumentExpander(
        text_generation_config=text_generation_config, prompts_config=prompt_config
    ).expand_documents(parsed_prompt_infos)
    assert (
        len(raw_generated_texts_df)
        == len(parsed_prompt_infos) * text_generation_config.n_generations
    ), "generating failed"
    raw_generated_texts_df = GenerationPostprocessor.convert_cols_to_dict(
        raw_generated_texts_df, ["prompt_info", "context_prompt_infos"]
    )
    return raw_generated_texts_df, failures


@step(enable_cache=False)
def postprocess_generated_texts(
    raw_generated_texts_df: pd.DataFrame,
) -> Annotated[pd.DataFrame, "generated_texts_df"]:
    return GenerationPostprocessor.run(raw_generated_texts_df)


@step(enable_cache=False)
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
    generated_texts_df: pd.DataFrame,
    evaluation_config: EvaluationConfig,
    paperswithcode_path: str,
) -> Annotated[pd.DataFrame, "generation_metrics_df"]:
    from github_search.utils import load_paperswithcode_df
    from tgutil.evaluation.evaluators import (
        TextGenerationEvaluator,
    )
    from tgutil.evaluation.preprocessing import EvalDFPreprocessor

    repo_tasks_df = load_paperswithcode_df(paperswithcode_path)
    texts_df = EvalDFPreprocessor(
        id_col=evaluation_config.id_col,
        reference_text_col=evaluation_config.reference_text_col,
    ).get_eval_df_from_raw_generated_text(generated_texts_df, repo_tasks_df)
    return (
        TextGenerationEvaluator.from_metric_names(
            metric_names=evaluation_config.metric_names
        )
        .get_evaluated_df(texts_df=texts_df, stratify="generation")
        .sort_values(by="rougeL", ascending=False)
    )


@step(enable_cache=True)
def prepare_search_df(
    search_data_config: SearchDataConfig,
) -> Annotated[pd.DataFrame, "search_df"]:
    path = search_data_config.search_df_path
    return utils.pd_read_star(path)


@step(enable_cache=True)
def evaluate_information_retrieval(
    search_df: pd.DataFrame,
    ir_config: InformationRetrievalEvaluatorConfig,
) -> Tuple[
    Annotated[pd.DataFrame, "per_query_metrics"],
    Annotated[pd.DataFrame, "aggregate_metrics"],
]:
    from github_search.ir.evaluator import InformationRetrievalEvaluator

    ir_evaluator = InformationRetrievalEvaluator.setup_from_df(search_df, ir_config)
    ir_metrics_df = ir_evaluator.evaluate()
    return (ir_metrics_df.per_query_metrics, ir_metrics_df.aggregate_metrics)


@step(enable_cache=True)
def prepare_search_df(
    search_data_config: SearchDataConfig, generated_texts_df: pd.DataFrame
) -> Annotated[pd.DataFrame, "search_df"]:
    path = search_data_config.search_df_path
    raw_search_df = utils.pd_read_star(path)
    search_df = raw_search_df.merge(
        generated_texts_df, on="repo", suffixes=["", "_generated"]
    )
    return search_df


@step(enable_cache=True)
def get_ir_experiments_results(
    search_df: pd.DataFrame,
    generation_eval_df: pd.DataFrame,
    column_config_path: str,
    embedder_config_path: str,
) -> Annotated[List[MetricsExperimentResult], "ir_experiments_results"]:
    config = MetricComparisonConfig.load(embedder_config_path, column_config_path)
    run_results = get_run_metrics(config, search_df, generation_eval_df)
    return list(run_results)
