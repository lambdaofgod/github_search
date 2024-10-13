from dagster import (
    asset,
    multi_asset,
    Definitions,
    Output,
    AssetOut,
    ConfigurableResource,
    AssetExecutionContext,
)
import yaml
import logging
import pandas as pd
from github_search.pipelines.steps import Code2DocSteps
from tqdm.contrib.logging import tqdm_logging_redirect

logging.basicConfig(level=logging.INFO)


class Code2DocConfig(ConfigurableResource):
    repos_df_path: str = "output/paperswithcode_with_readmes.json.gz"
    python_code_path: str = "output/repo_selected_files.parquet"
    n_repos_per_task: int = 10
    min_task_size: int = 1
    max_task_size: int = 10
    max_random_baseline_score: float = 0.3
    lm_model_name: str = "codellama"
    lm_base_url: str = "http://localhost:11434"
    files_per_repo: int = 10


@multi_asset(outs={"repos_df": AssetOut(), "python_code_df": AssetOut()})
def prepared_data(context: AssetExecutionContext):
    code2doc_config = context.resources.code2doc_config
    with tqdm_logging_redirect():
        repos_df, python_code_df = Code2DocSteps.prepare_data_df(
            code2doc_config.repos_df_path,
            code2doc_config.python_code_path,
        )
    yield Output(repos_df, output_name="repos_df")
    yield Output(python_code_df, output_name="python_code_df")


@asset
def sampled_repos(
    context: AssetExecutionContext, repos_df: pd.DataFrame, python_code_df: pd.DataFrame
) -> pd.DataFrame:
    code2doc_config = context.resources.code2doc_config
    sampled_repos_df = Code2DocSteps.create_repos_sample_df(
        repos_df,
        python_code_df,
        code2doc_config.n_repos_per_task,
        code2doc_config.min_task_size,
        code2doc_config.max_task_size,
        code2doc_config.max_random_baseline_score,
    )
    return sampled_repos_df


@asset
def generated_readmes(
    context: AssetExecutionContext, python_code_df: pd.DataFrame, sampled_repos: pd.DataFrame
) -> pd.DataFrame:
    code2doc_config = context.resources.code2doc_config
    logging.info(
        f"Generating readmes with code2doc using {code2doc_config.lm_model_name}, "
        f"using maximum of {code2doc_config.files_per_repo} files per repo"
    )

    with tqdm_logging_redirect():
        generated_readme_df = Code2DocSteps.generate_code2doc_readmes_df(
            python_code_df,
            sampled_repos,
            files_per_repo=code2doc_config.files_per_repo,
            lm_model_name=code2doc_config.lm_model_name,
            lm_base_url=code2doc_config.lm_base_url,
        )
    return generated_readme_df


@asset
def save_results(generated_readmes: pd.DataFrame):
    generated_readmes.to_json(
        "/tmp/generated_readmes.json", orient="records", lines=True
    )
    logging.info("Code2Doc pipeline completed successfully!")
    logging.info("Generated readmes saved to /tmp/generated_readmes.json")
