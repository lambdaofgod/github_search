from github_search.pipelines.steps import *
from tgutil.configs import PipelineConfig
from clearml import Dataset, PipelineController, Task
import fire
from github_search.utils import load_config_yaml_key
import pandas as pd
from github_search.pipelines.steps import (
    sample_data_step,
    expand_documents_step,
    evaluate_generated_texts_step,
    evaluate_generated_texts,
)
from tgutil.configs import (
    PipelineConfig,
    ConfigPaths,
    APIConfig,
    TextGenerationConfig,
    SamplingConfig,
    PromptConfig,
)
from tgutil.prompting_runner import sample_data, expand_documents
import logging
import yaml
from pathlib import Path


def add_generation_evaluation_step(pipe):
    pipe.add_function_step(
        name="evaluate_generated_texts",
        function=evaluate_generated_texts_step,
        function_kwargs=dict(
            generated_texts_df="${expand_documents.generated_texts_df}"
        ),
        function_return=["generation_evaluation_df"],
        cache_executed_step=True,
    )
    return pipe


def add_information_retrieval_evaluation_step(pipe, ir_config):
    pipe.add_function_step(
        name="evaluate_information_retrieval",
        function=evaluate_information_retrieval_step,
        function_kwargs=dict(
            searched_df="${expand_documents.generated_texts_df}", ir_config=ir_config
        ),
        function_return=["ir_results"],
        cache_executed_step=True,
    )
    return pipe


def make_expansion_pipeline(config: PipelineConfig):
    config_dict = config.dict()
    pipe = PipelineController(
        name=config.name,
        project=config.project,
        version="0.0.1",
        add_pipeline_tags=False,
    )
    pipe.add_parameter(
        name="sampling_config", default=config_dict["sampling_config"], param_type=dict
    )
    pipe.add_parameter(
        name="generation_config",
        default=config_dict["generation_config"],
        param_type=dict,
    )
    pipe.add_parameter(
        name="prompt_config", default=config_dict["prompt_config"], param_type=dict
    )
    params = pipe.get_parameters()
    pipe.add_function_step(
        name="sample_data",
        function=sample_data_step,
        function_kwargs=dict(sampling_config=params["sampling_config"]),
        function_return=["prompt_infos"],
        cache_executed_step=False,
    )
    pipe.add_function_step(
        name="expand_documents",
        function=expand_documents_step,
        function_kwargs=dict(
            text_generation_config=params["generation_config"],
            prompt_config=params["prompt_config"],
            prompt_infos="${sample_data.prompt_infos}",
        ),
        function_return=["generated_texts_df"],
        cache_executed_step=True,
    )
    return pipe


def make_pipeline(
    config: PipelineConfig, ir_config: Optional[InformationRetrievalEvaluatorConfig]
):
    pipe = make_expansion_pipeline(config)
    pipe = add_generation_evaluation_step(pipe)
    if ir_config is not None:
        pipe = add_information_retrieval_evaluation_step(pipe, ir_config)

    return pipe


def run_pipeline(
    sampling="micro",
    generation_method="api_rwkv",
    prompting_method="few_shot_markdown",
    ir_config_path=None,
):

    generation_config = load_config_yaml_key(
        APIConfig, "conf/generation.yaml", generation_method
    )
    sampling_config = load_config_yaml_key(
        SamplingConfig, "conf/sampling.yaml", sampling
    )
    prompt_config = load_config_yaml_key(
        PromptConfig, "conf/prompts.yaml", prompting_method
    )
    cfg = PipelineConfig(
        generation_config=generation_config,
        sampling_config=sampling_config,
        prompt_config=prompt_config,
        project="github_search/document_expansion",
        name=f"{sampling}-sampled document expansion pipeline",
    )
    if ir_config_path is not None:
        with open(ir_config_path) as f:
            ir_config = yaml.safe_load(f)
    else:
        ir_config = None
    pipe = make_pipeline(cfg, ir_config)
    print("running pipeline with config:", cfg)
    # cfg = PipelineConfig.parse_obj(cfg)
    # controller = Main().make_pipeline(cfg.sampling_config, cfg.text_generation_config)
    pipe.set_default_execution_queue("default")
    pipe.start_locally(run_pipeline_steps_locally=True)


if __name__ == "__main__":
    fire.Fire(run_pipeline)
