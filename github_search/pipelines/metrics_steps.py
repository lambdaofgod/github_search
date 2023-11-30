from operator import itemgetter
from typing import Annotated, List, Tuple

import pandas as pd
from github_search.ir import InformationRetrievalEvaluatorConfig
from github_search.ir.evaluator import (
    InformationRetrievalEvaluator,
    SearchDataFrameExtractor,
)
from github_search.pipelines.configs import EvaluationConfig
from github_search.pipelines.metrics_comparison import *
from tgutil.evaluation.evaluators import TextGenerationEvaluator
from tgutil.evaluation.preprocessing import EvalDFPreprocessor
from zenml import step


@step(enable_cache=False)
def evaluate_generation(
    generated_texts_df: pd.DataFrame,
    evaluation_config: EvaluationConfig,
) -> Annotated[pd.DataFrame, "generation_metrics_df"]:
    generated_texts_df = generated_texts_df.assign(
        reference_text=generated_texts_df["true_tasks"],
        generated_text=generated_texts_df["tasks"],
    )
    generation_metrics_df = run_evaluation(evaluation_config, generated_texts_df)
    return generation_metrics_df


def preprocess_evaluation_inputs(generated_texts_df, repo_tasks_df, evaluation_config):
    return EvalDFPreprocessor(
        id_col=evaluation_config.id_col,
        reference_text_col=evaluation_config.reference_text_col,
    ).get_eval_df_from_raw_generated_text(generated_texts_df, repo_tasks_df)


def run_evaluation(evaluation_config, texts_df):
    evaluation_df = (
        TextGenerationEvaluator.from_metric_names(
            metric_names=evaluation_config.metric_names
        )
        .get_evaluated_df(texts_df=texts_df, stratify="generation")
        .sort_values(by="rougeL", ascending=False)
    )
    return rename_bertscore_columns(evaluation_df)


def rename_bertscore_columns(df):
    return df.rename(
        columns={
            "f1": "bertscore_f1",
            "recall": "bertscore_recall",
            "precision": "bertscore_precision",
        }
    )


@step(enable_cache=False)
def prepare_search_df(
    generated_texts_df: pd.DataFrame, column_config: dict
) -> Annotated[pd.DataFrame, "search_df"]:
    column_config = InformationRetrievalColumnConfig(**column_config)
    generated_texts_df["dependencies"] = generated_texts_df["prompt_info"].apply(
        itemgetter("content")
    )
    search_df = SearchDataFrameExtractor.prepare_search_df(
        generated_texts_df, column_config
    )
    return search_df


@step(enable_cache=False)
def evaluate_information_retrieval(
    search_df: pd.DataFrame,
    ir_config: InformationRetrievalEvaluatorConfig,
) -> Tuple[
    Annotated[pd.DataFrame, "per_query_metrics"],
    Annotated[pd.DataFrame, "aggregate_metrics"],
]:
    ir_evaluator = InformationRetrievalEvaluator.setup_from_df(search_df, ir_config)
    ir_metrics_df = ir_evaluator.evaluate()
    return (ir_metrics_df.per_query_metrics, ir_metrics_df.aggregate_metrics)


@step(enable_cache=False)
def get_ir_experiments_results(
    search_df: pd.DataFrame,
    generation_eval_df: pd.DataFrame,
    column_config_path: str,
    embedder_config_path: str,
) -> Annotated[List[MetricsExperimentResult], "ir_experiments_results"]:
    config = MetricComparisonConfig.load(embedder_config_path, column_config_path)
    run_results = get_run_metrics(config, search_df, generation_eval_df)
    return list(run_results)
