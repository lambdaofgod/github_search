#!/usr/bin/env python3
# +
import hydra
from clearml import PipelineController
from prompting import *
from mlutil.text import rwkv_utils
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
import ast
import logging
from clearml import Dataset, Task
import jsonlines
from github_search.experiment_managers import ClearMLExperimentManager
from record_writer import JsonWriterContextManager, RecordWriter
from prompting_utils import (
    PrompterWrapper,
    MinichainRWKVConfig,
    MinichainHFConfig,
    MinichainPrompterWrapper,
)
import clearml

np.random.seed(seed=0)


# +

logging.basicConfig(level="INFO")


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
    # with get_experiment_manager(
    #     project_name=project_name, task_name="sample_prompt_infos"
    # ) as mgr:
    #     mgr.add_artifact(
    #         "sample_prompt_infos",
    #         sample_df,
    #         metadata={"n_samples": sampling_config.n_samples},
    #     )
    logging.info("created samples")
    return list(prompt_infos)


def expand_documents(
    text_generation_config: TextGenerationConfig,
    prompt_infos: List[dict],
    n_samples=None,
    fake=False,
):
    """
    expand documents by generating tasks using PrompterWrapper
    with prompt template et c specified in TextGenerationConfig
    """
    import pandas as pd
    from github_search.text_generation.configs import TextGenerationConfig
    from github_search.text_generation.prompting import PromptInfo
    from github_search.text_generation.prompting_utils import PrompterWrapper
    from pathlib import Path as P
    from github_search.text_generation.record_writer import (
        JsonWriterContextManager,
        RecordWriter,
    )

    text_generation_config = TextGenerationConfig(**text_generation_config)
    print(f"loading data from {text_generation_config.data_path}...")
    model_nm = P(text_generation_config.model_name.replace("/", "-")).parent.name
    out_path = P(text_generation_config.out_dir) / (model_nm + ".jsonl")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    print(f"write to {out_path}...")
    writer_kwargs = {"file_path": out_path}
    config = Mi
    prompter_wrapper = MinichainPrompterWrapper.create(
        text_generation_config.model_name,
        text_generation_config.templates_path,
        text_generation_config.prompt_template_name,
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
        function_return=["prompt_infos"],
        cache_executed_step=False,
    )
    pipe.add_function_step(
        name="expand_documents",
        function=expand_documents,
        function_kwargs=dict(
            text_generation_config=dict(generation_config),
            prompt_infos_df="${sample_data.prompt_infos}",
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


if __name__ == "__main__":
    fire.Fire(run_pipeline)
