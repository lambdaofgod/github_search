from typing import List, Dict, Any
import pandas as pd
from dagster import (
    asset,
    AssetExecutionContext,
    Config,
)
from github_search.pipelines.steps import ZenMLSteps


class DocumentExpansionConfig(Config):
    """Configuration for document expansion pipeline"""
    sampling_config: Dict[str, Any]
    generation_config: Dict[str, Any] 
    prompt_config: Dict[str, Any]


@asset
def prompt_infos(context: AssetExecutionContext, config: DocumentExpansionConfig) -> List[Dict]:
    """Sample data to create prompt infos for document expansion"""
    return ZenMLSteps.sample_data_step(
        prompt_config=config.prompt_config,
        sampling_config=config.sampling_config
    )


@asset
def expanded_documents(
    context: AssetExecutionContext, 
    config: DocumentExpansionConfig,
    prompt_infos: List[Dict]
) -> pd.DataFrame:
    """Expand documents using text generation"""
    raw_generated_texts_df, failures = ZenMLSteps.expand_documents_step(
        text_generation_config=config.generation_config,
        prompt_config=config.prompt_config,
        prompt_infos=prompt_infos
    )
    
    # Log failures for monitoring
    if failures:
        context.log.warning(f"Document expansion had {len(failures)} failures")
        context.add_output_metadata({"failures_count": len(failures)})
    
    # Post-process the generated texts
    generated_texts_df = ZenMLSteps.postprocess_generated_texts(raw_generated_texts_df)
    
    context.add_output_metadata({
        "num_generated_texts": len(generated_texts_df),
        "num_prompt_infos": len(prompt_infos)
    })
    
    return generated_texts_df
