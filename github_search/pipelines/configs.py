from pydantic import BaseModel, Field
from typing import List
from tgutil.configs import (
    PipelineConfig as GenerationPipelineConfig,
    SamplingConfig,
    PromptConfig,
    APIConfig,
    TextGenerationConfig,
    ConfigPaths,
)
from pathlib import Path as P

from github_search.utils import load_config_yaml_key


class EvaluationConfig(BaseModel):
    id_col: str = Field(default="repo")
    reference_text_col: str = Field(default="true_text")
    metric_names: List[str] = Field(
        default_factory=lambda: [
            "edit_word",
            "jaccard_lst",
            "bleurt",
            "rouge",
            # "wmd",
            "sentence_transformer_similarity",
        ]
    )


class SearchDataConfig(BaseModel):
    search_df_path: str = Field(default="data/search_df.csv")


class PipelineConfig(BaseModel):
    generation_config: GenerationPipelineConfig
    name: str
    project: str

    @staticmethod
    def load_from_paths(
        name: str,
        project: str,
        sampling="small",
        generation_method="api_rwkv",
        prompting_method="few_shot_markdown",
        search_config_path="conf/pipeline/search.yaml",
    ):
        generation_config = load_config_yaml_key(
            APIConfig, "conf/pipeline/generation.yaml", generation_method
        )
        sampling_config = load_config_yaml_key(
            SamplingConfig, "conf/pipeline/sampling.yaml", sampling
        )
        prompt_config = load_config_yaml_key(
            PromptConfig, "conf/pipeline/prompts.yaml", prompting_method
        )

        generation_pipeline_config = GenerationPipelineConfig(
            generation_config=generation_config,
            sampling_config=sampling_config,
            prompt_config=prompt_config,
            project="github_search/document_expansion",
            name=f"{sampling}-sampled document expansion pipeline",
        )
        return PipelineConfig(
            generation_config=generation_pipeline_config,
            name=name,
            project=project,
        )
