from typing import Annotated

import json
import pandas as pd
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
import pathlib

import logging

logging.basicConfig(level=logging.INFO)


def generation_pipeline(sampling, generation_method, prompting_method, results_dir):
    pipeline_config = PipelineConfig.load_from_paths(
        "generation pipeline",
        "github_search",
        generation_method=generation_method,
        prompting_method=prompting_method,
        sampling=sampling,
        search_config_path="conf/pipeline/search.yaml",
    )
    save_dir = pathlib.Path(results_dir) / "_".join(
        [generation_method, prompting_method, sampling]
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    config = pipeline_config.generation_config
    logging.info(f"Sampling")
    prompt_infos = sample_data_step(
        dict(config.prompt_config), dict(config.sampling_config)
    )
    with open(save_dir / "prompt_infos.json", "w") as f:
        f.write(json.dumps(prompt_infos))
    logging.info(f"Expanding documents")
    if not (save_dir / "raw_generated_texts.csv").exists():
        raw_generated_texts_df, failures = expand_documents_step(
            dict(config.generation_config), dict(
                config.prompt_config), prompt_infos
        )
        raw_generated_texts_df.to_csv(
            save_dir / "raw_generated_texts.csv", index=False)
    else:
        raw_generated_texts_df = pd.read_csv(
            save_dir / "raw_generated_texts.csv", index=False
        )
    logging.info(f"Postprocessing")
    generated_texts_df = postprocess_generated_texts(raw_generated_texts_df)
    generated_texts_df.to_csv(save_dir / "generated_texts.csv", index=False)


def load_generation_pipeline_df(
    paperswithcode_path,
) -> Annotated[pd.DataFrame, "generated_texts_df"]:
    client = zenml.client.Client()
    # runs = client.get_pipeline("generation_pipeline").runs
    # zenml_run = [r for r in runs if r.status == "completed"][0]
    # generated_texts_df = (
    #     zenml_run.steps["postprocess_generated_texts"]
    #     .outputs["generated_texts_df"]
    #     .load()
    # )

    artifact = client.get_artifact("36c1d840-5347-40ee-85a8-103a544aaed9")
    generated_texts_df = artifact.load()
    paperswithcode_df = load_filtered_paperswithcode_df(
        paperswithcode_path, generated_texts_df.columns
    )

    return generated_texts_df.merge(paperswithcode_df, on="repo")


def load_filtered_paperswithcode_df(paperswithcode_path, ignore_cols):
    paperswithcode_df = pd.read_json(paperswithcode_path)
    paperswithcode_df = paperswithcode_df[~paperswithcode_df["readme"].isna()]
    selected_cols = [
        col
        for col in paperswithcode_df.columns
        if col not in ignore_cols or col == "repo"
    ]
    return paperswithcode_df[selected_cols]


def metrics_pipeline(
    paperswithcode_path,
    ir_config_path="conf/pipeline/ir_config.yaml",
    embedder_config_path="conf/pipeline/retrieval.yaml",
    column_config_path="conf/pipeline/column_configs.yaml",
):
    generated_texts_df = load_generation_pipeline_df(paperswithcode_path)
    generation_eval_df = evaluate_generation(
        generated_texts_df,
        EvaluationConfig(reference_text_col="true_tasks"),
    )

    ir_config = load_config_yaml_key(
        InformationRetrievalEvaluatorConfig, ir_config_path, "nbow"
    )

    search_df = prepare_search_df(
        generated_texts_df, ir_config.column_config.dict())
    get_ir_experiments_results(
        search_df, generation_eval_df, column_config_path, embedder_config_path
    )


def run_generation(
    sampling="no_sampling",
    generation_method="api_lmserver",
    prompting_method="few_shot_markdown",
    results_dir="output/pipelines",
):
    generation_pipeline(sampling, generation_method,
                        prompting_method, results_dir)


def run_metrics():
    metrics_pipeline()
