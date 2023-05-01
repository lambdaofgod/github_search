#!/usr/bin/env python3
# +
import hydra
from clearml import PipelineController
from prompting import *
from mlutil.text import rwkv_utils
from promptify import OpenAI, Prompter
from promptify.models.nlp.model import Model as PromptifyModel
from github_search.text_generation.prompting import PromptInfo
from typing import Dict
import fire
import numpy as np
import tqdm
import json
from pydantic.dataclasses import dataclass
from github_search.text_generation.configs import (
    SamplingConfig,
    TextGenerationConfig,
    PipelineConfig,
)

from pathlib import Path as P

# -

from record_writer import JsonWriterContextManager, RecordWriter
from promptify_utils import PrompterWrapper
import clearml

np.random.seed(seed=0)


import ast

# +
import logging

logging.basicConfig(level="INFO")


# -


# +
from clearml import Dataset, Task
import jsonlines
from github_search.experiment_managers import ClearMLExperimentManager


def get_experiment_manager(project_name, task_name, config=dict()):
    return ClearMLExperimentManager(
        project_name=project_name, task_name=task_name, config=config
    )


from pydantic_yaml import YamlModel


import functools


project_name = "github_search"


def sample_data(
    sampling_config: SamplingConfig,
):
    import pandas as pd
    from github_search.text_generation.context_loader import ContextLoader
    from github_search.text_generation.configs import SamplingConfig
    import logging

    sampling_config = SamplingConfig(**sampling_config)
    prompt_infos = ContextLoader(
        data_path=sampling_config.pq_data_path
    ).load_prompt_infos(n_samples=sampling_config.n_samples)
    sample_df = pd.DataFrame.from_records(map(dict, prompt_infos))
    sample_df.to_json(sampling_config.out_data_path, orient="records", lines=True)
    # with get_experiment_manager(
    #     project_name=project_name, task_name="sample_prompt_infos"
    # ) as mgr:
    #     mgr.add_artifact(
    #         "sample_prompt_infos",
    #         sample_df,
    #         metadata={"n_samples": sampling_config.n_samples},
    #     )
    logging.info("created samples")
    return sample_df


def expand_documents(
    text_generation_config: TextGenerationConfig,
    prompt_infos_df: pd.DataFrame,
    n_samples=None,
    fake=False,
):
    import pandas as pd
    from github_search.text_generation.configs import TextGenerationConfig
    from github_search.text_generation.prompting import PromptInfo
    from github_search.text_generation.promptify_utils import PrompterWrapper
    from pathlib import Path as P
    from github_search.text_generation.record_writer import (
        JsonWriterContextManager,
        RecordWriter,
    )

    text_generation_config = TextGenerationConfig(**text_generation_config)
    print(f"loading data from {text_generation_config.data_path}...")
    prompt_infos = [PromptInfo(**d) for d in prompt_infos_df.to_records()]
    model_nm = P(text_generation_config.model_name.replace("/", "-")).name.parent
    out_path = P(text_generation_config.out_dir) / (model_nm + ".jsonl")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    if out_path.exists():
        out_path.unlink()
    print(f"write to {out_path}...")
    writer_kwargs = {"file_path": out_path}
    prompter_wrapper = PrompterWrapper.create(
        text_generation_config.model_name,
        text_generation_config.templates_path,
        text_generation_config.prompt_template_name,
        use_fake_model=fake,
    )

    json_writer = RecordWriter(JsonWriterContextManager)
    records = list(
        json_writer.map(
            prompter_wrapper.get_dict_with_generated_text,
            prompt_infos,
            **writer_kwargs,
        )
    )
    # with get_experiment_manager(
    #     project_name,
    #     task_name="document_expansion",
    #     config=dict(text_generation_config),
    # ) as mgr:
    #     mgr.add_artifact("generated_texts", out_path, metadata={"total": n_samples})
    generated_texts_df = pd.DataFrame.from_records(records)
    return generated_texts_df


def make_pipeline(config: PipelineConfig):
    sampling_config = config.sampling_config
    generation_config = config.generation_config
    pipe = PipelineController(
        name="sampled document expansion pipeline",
        project="examples",
        version="0.0.1",
        add_pipeline_tags=False,
    )
    pipe.add_function_step(
        name="sample_data",
        function=sample_data,
        function_kwargs=dict(sampling_config=dict(sampling_config)),
        function_return=["sample_df"],
        cache_executed_step=False,
    )
    pipe.add_function_step(
        name="expand_documents",
        function=expand_documents,
        function_kwargs=dict(
            text_generation_config=dict(generation_config),
            prompt_infos_df="${sample_data.sample_df}",
        ),
        function_return=["generated_texts_df"],
    )
    return pipe


def run_pipeline(cfg_path="conf/text_generation"):
    cfg = PipelineConfig.load_from_paths(cfg_path)

    pipe = make_pipeline(cfg)
    print("running pipeline with config:", cfg)
    # cfg = PipelineConfig.parse_obj(cfg)
    # controller = Main().make_pipeline(cfg.sampling_config, cfg.text_generation_config)
    pipe.set_default_execution_queue("default")
    pipe.start_locally(run_pipeline_steps_locally=True)


# -

if __name__ == "__main__":
    fire.Fire(run_pipeline)

# !ls /home/kuba/Projects/forks/text-generation-webui/models/llama-13b-hf/
