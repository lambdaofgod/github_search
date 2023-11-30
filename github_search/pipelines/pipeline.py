from typing import Annotated

import pandas as pd
import zenml
from github_search.ir import InformationRetrievalEvaluatorConfig
from github_search.pipelines.configs import (
    EvaluationConfig,
    PipelineConfig,
    SearchDataConfig,
)
from github_search.pipelines.steps import (
    expand_documents_step,
    postprocess_generated_texts,
    sample_data_step,
)
from github_search.pipelines.metrics_steps import (
    evaluate_generation,
    evaluate_information_retrieval,
    get_ir_experiments_results,
    prepare_search_df,
)
from github_search.utils import load_config_yaml_key
from zenml import pipeline, step


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
    raw_generated_texts_df, failures = expand_documents_step(
        dict(config.generation_config), dict(config.prompt_config), prompt_infos
    )
    generated_texts_df = postprocess_generated_texts(raw_generated_texts_df)


@step(enable_cache=False)
def load_generation_pipeline_df() -> Annotated[pd.DataFrame, "generated_texts_df"]:
    client = zenml.client.Client()
    # runs = client.get_pipeline("generation_pipeline").runs
    # zenml_run = [r for r in runs if r.status == "completed"][0]
    # generated_texts_df = (
    #     zenml_run.steps["postprocess_generated_texts"]
    #     .outputs["generated_texts_df"]
    #     .load()
    # )
    artifact = client.get_artifact("d6f9f42e-a0b7-416e-a82f-fba6b627121a")
    generated_texts_df = artifact.load()
    return generated_texts_df


@pipeline
def metrics_pipeline(
    ir_config_path="conf/pipeline/ir_config.yaml",
    embedder_config_path="conf/pipeline/retrieval.yaml",
    column_config_path="conf/pipeline/column_configs.yaml",
    search_df_path="output/nbow_data_test.parquet",
):
    generated_texts_df = load_generation_pipeline_df()
    generation_eval_df = evaluate_generation(
        generated_texts_df,
        EvaluationConfig(reference_text_col="true_tasks"),
    )

    ir_config = load_config_yaml_key(
        InformationRetrievalEvaluatorConfig, ir_config_path, "nbow"
    )
    search_data_config = SearchDataConfig(search_df_path=search_df_path)

    search_df = prepare_search_df(generated_texts_df, ir_config.column_config.dict())
    get_ir_experiments_results(
        search_df, generation_eval_df, column_config_path, embedder_config_path
    )


def run_generation(
    sampling="no_sampling",
    generation_method="api_lmserver",
    prompting_method="few_shot_markdown",
):
    generation_pipeline(sampling, generation_method, prompting_method)


def run_metrics():
    metrics_pipeline()
