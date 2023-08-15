import zenml
import fire
import pandas as pd
from typing import Tuple
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
def generation_pipeline():
    pipeline_config = PipelineConfig.load_from_paths(
        "generation pipeline",
        "github_search",
        "../../data/paperswithcode_with_tasks.csv",
    )
    config = pipeline_config.generation_config
    prompt_infos = sample_data_step(
        dict(config.prompt_config), dict(config.sampling_config)
    )
    generated_texts_df, failures = expand_documents_step(
        dict(config.generation_config), dict(config.prompt_config), prompt_infos
    )
    generation_eval_df = evaluate_generated_texts(
        generated_texts_df,
        pipeline_config.paperswithcode_path,
        EvaluationConfig(reference_text_col="true_tasks"),
    )


@step
def load_generation_pipeline_dfs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    client = zenml.client.Client()
    run = client.get_pipeline("generation_pipeline").runs[0]
    generated_texts_df = (
        run.steps["expand_documents_step"].outputs["generated_texts_df"].load()
    )
    generation_eval_df = (
        run.steps["evaluate_generated_texts"].outputs["generation_metrics_df"].load()
    )
    return generated_texts_df, generation_eval_df


@pipeline
def metrics_pipeline():
    generated_texts_df, generation_eval_df = load_generation_pipeline_dfs()
    ir_config = load_config_yaml_key(
        InformationRetrievalEvaluatorConfig, "conf/ir_config.yaml", "nbow"
    )
    search_data_config = SearchDataConfig(
        search_df_path="../../output/nbow_data_test.parquet"
    )

    search_df = prepare_search_df(search_data_config, generated_texts_df)
    ir_df = evaluate_information_retrieval(search_df, ir_config)
    get_ir_experiments_results(search_df, generation_eval_df)


class CLI:
    @staticmethod
    def run_generation():
        generation_pipeline()

    @staticmethod
    def run_metrics():
        metrics_pipeline()


if __name__ == "__main__":
    CLI().run_metrics()
