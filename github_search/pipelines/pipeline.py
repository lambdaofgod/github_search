import zenml
import pandas as pd
from typing import Tuple, Annotated
from github_search.pipelines.configs import EvaluationConfig, SearchDataConfig
from github_search.pipelines.steps import (
    expand_documents_step,
    sample_data_step,
    evaluate_generated_texts,
    evaluate_information_retrieval,
    prepare_search_df,
    get_ir_experiments_results,
)

from github_search.ir import InformationRetrievalEvaluatorConfig
from github_search.utils import load_config_yaml_key
from zenml import pipeline, step
from github_search.pipelines.configs import PipelineConfig


@pipeline
def generation_pipeline(
    sampling,
    generation_method,
    prompting_method,
):
    pipeline_config = PipelineConfig.load_from_paths(
        "generation pipeline",
        "github_search",
        generation_method=generation_method,
        prompting_method=prompting_method,
        sampling=sampling,
        search_config_path="conf/pipeline/search.yaml",
    )
    config = pipeline_config.generation_config
    prompt_infos = sample_data_step(
        dict(config.prompt_config), dict(config.sampling_config)
    )
    generated_texts_df, failures = expand_documents_step(
        dict(config.generation_config), dict(config.prompt_config), prompt_infos
    )


@step(enable_cache=True)
def load_generation_pipeline_df() -> Annotated[pd.DataFrame, "generated_texts_df"]:
    client = zenml.client.Client()
    run = client.get_pipeline("generation_pipeline").runs[0]
    generated_texts_df = (
        run.steps["expand_documents_step"].outputs["generated_texts_df"].load()
    )
    return generated_texts_df


@pipeline
def metrics_pipeline():
    generated_texts_df = load_generation_pipeline_df()
    generation_eval_df = evaluate_generated_texts(
        generated_texts_df,
        EvaluationConfig(reference_text_col="true_tasks"),
    )

    ir_config = load_config_yaml_key(
        InformationRetrievalEvaluatorConfig, "conf/ir_config.yaml", "nbow"
    )
    search_data_config = SearchDataConfig(
        search_df_path="../../output/nbow_data_test.parquet"
    )

    search_df = prepare_search_df(search_data_config, generated_texts_df)
    get_ir_experiments_results(search_df, generation_eval_df)


def run_generation(
    sampling="no_sampling",
    generation_method="api_rwkv",
    prompting_method="few_shot_markdown",
):
    generation_pipeline(sampling, generation_method, prompting_method)

def run_metrics():
    metrics_pipeline()
