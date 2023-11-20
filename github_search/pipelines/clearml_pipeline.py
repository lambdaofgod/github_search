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
from github_search.ir import InformationRetrievalEvaluatorConfig
import logging
import yaml
from pathlib import Path

from pydantic_yaml import YamlModel
from pydantic import Field
from pathlib import Path as P


class SamplingConfig(YamlModel):
    n_samples = 100
    pq_data_path = "../../output/nbow_data_test.parquet"
    out_data_path = "../../output/prompt_infos.jsonl"
    dset_kwargs = dict(
        dataset_project="github_search_llms", dataset_name="prompt_info_sample"
    )


class TextGenerationConfig(YamlModel):
    model_name: str = Field("dvruette/oasst-pythia-6.9b-4000-steps")
    prompt_template_name: str = Field("md_prompt.jinja")
    templates_path: str = Field("prompt_templates")
    data_path: str = Field("../output/prompt_infos.jsonl")
    out_dir: str = Field("../output/llms/")


class ConfigPaths(YamlModel):
    sampling: str
    generation: str


class PipelineConfig(YamlModel):
    sampling_config: SamplingConfig
    generation_config: TextGenerationConfig
    ir_config: InformationRetrievalConfig
    evaluate_generation: bool = Field(default=True)

    @staticmethod
    def load_from_paths(config_paths_dir: str):
        root_dir = P(config_paths_dir)
        cfg_paths = ConfigPaths.parse_file(root_dir / "config.yaml")
        sampling_config = SamplingConfig.parse_file(
            f"{str(root_dir)}/sampling/{cfg_paths.sampling}.yaml"
        )
        generation_config = TextGenerationConfig.parse_file(
            f"{str(root_dir)}/generation/{cfg_paths.generation}.yaml"
        )
        with open(f"{str(root_dir)}/generation/{cfg_paths.generation}.yaml") as f:
            ir_config = InformationRetrievalConfig.parse_obj(yaml.safe_load(f))
        return PipelineConfig(
            sampling_config=sampling_config,
            generation_config=generation_config,
            ir_config=ir_config,
        )


def add_generation_evaluation_step(pipe):
    paperswithcode_path = pipe.get_parameters()["paperswithcode_path"]
    pipe.add_function_step(
        name="evaluate_generated_texts",
        function=evaluate_generated_texts_step,
        function_kwargs=dict(
            generated_texts_df="${expand_documents.generated_texts_df}",
            paperswithcode_path=paperswithcode_path,
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
        name="paperswithcode_path", default=config.paperswithcode_path, param_type=str
    )
    pipe.add_parameter(
        name="sampling_config", default=dict(config.sampling_config), param_type=dict
    )
    pipe.add_parameter(
        name="generation_config",
        default=dict(config.generation_config),
        param_type=dict,
    )
    pipe.add_parameter(
        name="prompt_config", default=dict(config.prompt_config), param_type=dict
    )
    params = pipe.get_parameters()
    pipe.add_function_step(
        name="sample_data",
        function=sample_data_step,
        function_kwargs=dict(
            sampling_config=params["sampling_config"],
            prompt_config=params["prompt_config"],
        ),
        function_return=["prompt_infos"],
        cache_executed_step=True,
    )
    pipe.add_function_step(
        name="expand_documents",
        function=expand_documents_step,
        function_kwargs=dict(
            text_generation_config=dict(params["generation_config"]),
            prompt_config=dict(params["prompt_config"]),
            prompt_infos="${sample_data.prompt_infos}",
        ),
        function_return=["generated_texts_df", "failures"],
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
    paperswithcode_path="../../data/paperswithcode_with_tasks.csv",
    ir_config_path=None,
):
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
