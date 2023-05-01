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
        return PipelineConfig(
            sampling_config=sampling_config, generation_config=generation_config
        )
