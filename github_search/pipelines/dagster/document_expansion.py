from typing import Dict, Any
import pandas as pd
import yaml
from dagster import (
    asset,
    AssetExecutionContext,
    Config,
    AssetIn,
)
from github_search.pipelines.steps import ZenMLSteps
import logging


def load_yaml_key(config_path: str, key: str) -> Dict[str, Any]:
    """Load a specific key from a YAML file"""
    with open(config_path) as f:
        conf = yaml.safe_load(f)[key]
    return conf


class DocumentExpansionConfig(Config):
    """Configuration for document expansion pipeline"""

    sampling_config_key: str = "no_sampling"
    generation_config_key: str = "api_vllm"
    prompt_config_key: str = "few_shot_markdown"

    @property
    def sampling_config(self) -> Dict[str, Any]:
        return load_yaml_key("conf/pipeline/sampling.yaml", self.sampling_config_key)

    @property
    def generation_config(self) -> Dict[str, Any]:
        return load_yaml_key(
            "conf/pipeline/generation.yaml", self.generation_config_key
        )

    @property
    def prompt_config(self) -> Dict[str, Any]:
        return load_yaml_key("conf/pipeline/prompts.yaml", self.prompt_config_key)


@asset
def prompt_infos(
    context: AssetExecutionContext, config: DocumentExpansionConfig
) -> pd.DataFrame:
    """Sample data to create prompt infos for document expansion"""
    logging.info("sampling config")
    logging.info(config.sampling_config)
    return pd.DataFrame.from_records(
        ZenMLSteps.sample_data_step(
            prompt_config=config.prompt_config, sampling_config=config.sampling_config
        )
    )


@asset(
    ins={
        "prompt_infos": AssetIn(key="prompt_infos"),
        "sampled_repos": AssetIn(key="sampled_repos"),
    }
)
def expanded_documents(
    context: AssetExecutionContext,
    config: DocumentExpansionConfig,
    prompt_infos: pd.DataFrame,
    sampled_repos: pd.DataFrame,
) -> pd.DataFrame:
    """Expand documents using text generation"""
    prompt_info_repos = prompt_infos["prompt_info"].apply(lambda d: d["name"])
    valid_prompt_infos_mask = prompt_info_repos.isin(sampled_repos["repo"])

    ## SMALL SAMPLE
    prompt_infos = prompt_infos[valid_prompt_infos_mask]

    generated_texts_df, failures = ZenMLSteps.expand_documents_step(
        text_generation_config=config.generation_config,
        prompt_config=config.prompt_config,
        prompt_infos=prompt_infos.to_dict("records"),
    )

    # Log failures for monitoring
    context.log.warning(f"Document expansion had {len(failures)} failures")

    return generated_texts_df


@asset(
    ins={
        "expanded_documents": AssetIn(key="expanded_documents"),
    }
)
def processed_expanded_documents(
    context: AssetExecutionContext, expanded_documents: pd.DataFrame
) -> pd.DataFrame:
    librarian_signatures = expanded_documents["generated_text"]
    dependency_signatures = expanded_documents["prompt_info"].apply(
        lambda pinfo: pinfo["prompt_info"]["content"]
    )
    names = expanded_documents["prompt_info"].apply(
        lambda pinfo: pinfo["prompt_info"]["name"]
    )
    repository_signatures = (
        "repo:\n"
        + names
        + "\ntasks:\n"
        + librarian_signatures
        + "\nimportant nodes:\n"
        + dependency_signatures
    )
    return (
        pd.DataFrame(
            {
                "pagerank_generated_tasks": librarian_signatures,
                "pagerank_dependency_signature": dependency_signatures,
                "repo": names,
                "pagerank_repository_signature": repository_signatures,
            }
        )
        .drop_duplicates(subset=["repo"])
        .sort_values("repo")
    )
