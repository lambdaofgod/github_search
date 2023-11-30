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
from zenml import step


@step(enable_cache=False)
def expand_documents_step(
    text_generation_config: dict, prompt_config: dict, prompt_infos: List[dict]
) -> Tuple[
    Annotated[pd.DataFrame, "raw_generated_texts_df"], Annotated[List[dict], "failures"]
]:
    logging.info("expanding documents")
    logging.info(f"using text generation config: {text_generation_config}")
    logging.info(f"using prompt config: {prompt_config}")
    text_generation_config = load_config_from_dict(text_generation_config)
    prompt_config = PromptConfig(**prompt_config)
    parsed_prompt_infos = [ContextPromptInfo.parse_obj(pi) for pi in prompt_infos]
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


@step(enable_cache=False)
def postprocess_generated_texts(
    raw_generated_texts_df: pd.DataFrame,
) -> Annotated[pd.DataFrame, "generated_texts_df"]:
    return GenerationPostprocessor.run(raw_generated_texts_df)


@step(enable_cache=False)
def sample_data_step(
    prompt_config: dict, sampling_config: dict
) -> Annotated[List[dict], "prompt_infos"]:
    from tgutil.configs import PromptConfig, SamplingConfig
    from tgutil.prompting_runner import DocumentExpander

    sampling_config = SamplingConfig(**sampling_config)
    prompt_config = PromptConfig(**prompt_config)
    return DocumentExpander.sample_data(prompt_config, sampling_config)
