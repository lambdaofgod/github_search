import zenml

from github_search.pipelines.steps import (
    expand_documents_step,
    sample_data_step,
    evaluate_generated_texts,
)
from github_search.utils import load_config_yaml_key
from zenml import pipeline, step
from tgutil.configs import (
    PipelineConfig,
    ConfigPaths,
    APIConfig,
    TextGenerationConfig,
    SamplingConfig,
    PromptConfig,
)


def load_pipeline_config(
    sampling="micro",
    generation_method="api_rwkv",
    prompting_method="few_shot_markdown",
    paperswithcode_path="../../data/paperswithcode_with_tasks.csv",
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

    generation_config.n_generations = 1
    return PipelineConfig(
        generation_config=generation_config,
        sampling_config=sampling_config,
        prompt_config=prompt_config,
        project="github_search/document_expansion",
        name=f"{sampling}-sampled document expansion pipeline",
        paperswithcode_path=paperswithcode_path,
    )


@pipeline()
def generation_pipeline():
    config = load_pipeline_config()
    prompt_infos = sample_data_step(
        dict(config.prompt_config), dict(config.sampling_config)
    )
    generated_texts_df, failures = expand_documents_step(
        dict(config.generation_config), dict(
            config.prompt_config), prompt_infos
    )
    generation_eval_df = evaluate_generated_texts(
        generated_texts_df, config.paperswithcode_path
    )


if __name__ == "__main__":
    generation_pipeline()
