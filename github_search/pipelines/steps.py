from typing import Annotated, List, Tuple

import pandas as pd
from github_search.ir.evaluator import (
    InformationRetrievalEvaluator,
    SearchDataFrameExtractor,
)
from github_search.pipelines.metrics_comparison import *
from github_search.pipelines.postprocessing import GenerationPostprocessor
from tgutil.configs import PromptConfig, load_config_from_dict
from tgutil.prompting import ContextPromptInfo
from tgutil.prompting_runner import DocumentExpander
from github_search.samplers import TaskSampler, RepoSampler


def expand_documents_step(
    text_generation_config: dict, prompt_config: dict, prompt_infos: List[dict]
) -> Tuple[
    Annotated[pd.DataFrame,
              "raw_generated_texts_df"], Annotated[List[dict], "failures"]
]:
    logging.info("expanding documents")
    logging.info(f"using text generation config: {text_generation_config}")
    logging.info(f"using prompt config: {prompt_config}")
    text_generation_config = load_config_from_dict(text_generation_config)
    prompt_config = PromptConfig(**prompt_config)
    parsed_prompt_infos = [
        ContextPromptInfo.parse_obj(pi) for pi in prompt_infos]
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


def postprocess_generated_texts(
    raw_generated_texts_df: pd.DataFrame,
) -> Annotated[pd.DataFrame, "generated_texts_df"]:
    return GenerationPostprocessor.run(raw_generated_texts_df)


def sample_data_step(
    prompt_config: dict, sampling_config: dict
) -> Annotated[List[dict], "prompt_infos"]:
    from tgutil.configs import PromptConfig, SamplingConfig
    from tgutil.prompting_runner import DocumentExpander

    sampling_config = SamplingConfig(**sampling_config)
    prompt_config = PromptConfig(**prompt_config)
    return [
        pinfo.dict()
        for pinfo in DocumentExpander.sample_data(prompt_config, sampling_config)
    ]


def select_repos_with_all_data(repos_df_path, python_code_path, output_path):
    repos_df = pd.read_json(repos_df_path)
    python_code_df = pd.read_parquet(python_code_path)
    repos_with_all_data_df = repos_df[repos_df["repo"].isin(
        python_code_df["repo_name"])]
    repos_with_all_data_df.to_json(output_path, orient="records", lines=True)


def create_repos_sample(repos_df_path, output_path, sampled_tasks=100, repos_per_task=20, min_task_size=250, max_task_size=2500, min_repo_tasks=4):
    logging.basicConfig(level=logging.INFO)
    repos_df = pd.read_json(repos_df_path, orient="records", lines=True)
    if type(repos_df["tasks"].iloc[0]) is str:
        repos_df["tasks"] = repos_df["tasks"].apply(ast.literal_eval)
    tasks_sample = TaskSampler.sample_tasks_from_lists(
        repos_df["tasks"], sample_size=sampled_tasks, min_size=min_task_size, max_size=max_task_size)
    logging.info(f"Sampled {len(tasks_sample)} tasks.")
    sampled_repos_df = RepoSampler.sample_repos(
        repos_df, tasks_sample, sample_size_per_task=repos_per_task, min_repo_tasks=min_repo_tasks)
    sampled_repos_df["query_tasks"] = sampled_repos_df["tasks"].apply(
        lambda ts: [t for t in ts if t in tasks_sample])
    logging.info(f"Sampled {len(sampled_repos_df)} repos.")
    sampled_repos_df.to_json(output_path, orient="records", lines=True)
