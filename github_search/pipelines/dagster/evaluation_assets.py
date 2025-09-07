import pandas as pd
from github_search.evaluation.evaluators import TextGenerationEvaluator
from dagster import (
    asset,
    multi_asset,
    AssetOut,
    ConfigurableResource,
    AssetExecutionContext,
    Output,
    AssetIn,
)


@asset
def generation_metrics_df(repos_with_representations_df: pd.DataFrame) -> pd.DataFrame:
    """
    generation metrics evaluated for each repo
    these metrics compare the actual tasks with the generated tasks
    """

    generated_texts_df = repos_with_representations_df.copy().assign(
        reference_text=repos_with_representations_df["tasks"].apply(", ".join),
        generated_text=repos_with_representations_df["generated_tasks"],
    )

    text_generation_evaluator = TextGenerationEvaluator.from_metric_names(
        ["rouge", "sentence_transformer_similarity"]
    )  # , "sentence_transformer_similarity"])

    generation_scores_df = text_generation_evaluator.get_scores(generated_texts_df)

    return pd.concat(
        [
            generated_texts_df[["repo", "tasks"]].rename(columns={"tasks": "task"}),
            generation_scores_df,
        ],
        axis=1,
    )


@asset
def per_query_generation_metrics_df(
    generation_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        generation_metrics_df.drop(columns=["repo"])
        .explode("task")
        .groupby("task")
        .agg("mean")
    )
