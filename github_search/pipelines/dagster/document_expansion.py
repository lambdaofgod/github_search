from typing import List, Dict, Any
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

    sampling_config: Dict[str, Any] = load_yaml_key(
        "conf/pipeline/sampling.yaml", "no_sampling"
    )
    generation_config: Dict[str, Any] = load_yaml_key(
        "conf/pipeline/generation.yaml", "api_vllm"
    )
    prompt_config: Dict[str, Any] = load_yaml_key(
        "conf/pipeline/prompts.yaml", "few_shot_markdown"
    )


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


# @asset(ins={"prompt_infos": AssetIn(key="prompt_infos")})
# def expanded_documents(
#     context: AssetExecutionContext,
#     config: DocumentExpansionConfig,
#     prompt_infos: List[Dict],
# ) -> pd.DataFrame:
#     """Expand documents using text generation"""
#     raw_generated_texts_df, failures = ZenMLSteps.expand_documents_step(
#         text_generation_config=config.generation_config,
#         prompt_config=config.prompt_config,
#         prompt_infos=prompt_infos,
#     )

#     # Log failures for monitoring
#     context.log.warning(f"Document expansion had {len(failures)} failures")

#     # Post-process the generated texts
#     generated_texts_df = ZenMLSteps.postprocess_generated_texts(raw_generated_texts_df)

#     # context.add_output_metadata(
#     #     {
#     #         "num_generated_texts": len(generated_texts_df),
#     #         "num_prompt_infos": len(prompt_infos),
#     #     }
#     # )

#     return generated_texts_df
