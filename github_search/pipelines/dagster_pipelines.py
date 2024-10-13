from dagster import (
    asset,
    multi_asset,
    AssetExecutionContext,
    Config,
    Definitions,
    Output,
    AssetOut,
)
import yaml
import logging
import pandas as pd
from github_search.pipelines.steps import Code2DocSteps
from tqdm.contrib.logging import tqdm_logging_redirect
import logging

logging.basicConfig(level=logging.INFO)


class Code2DocConfig(Config):
    config_path: str = "github_search/pipelines/configs/code2doc_default_config.yaml"


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)["pipeline"]


@asset
def code2doc_config(config: Code2DocConfig) -> dict:
    logging.info(f"Loading config from {config.config_path}")
    return load_config(config.config_path)


@multi_asset(outs={"repos_df": AssetOut(), "python_code_df": AssetOut()})
def prepared_data(code2doc_config: dict):
    with tqdm_logging_redirect():
        repos_df, python_code_df = Code2DocSteps.prepare_data_df(
            code2doc_config["repos_df_path"],
            code2doc_config["python_code_path"],
        )
    yield Output(repos_df, output_name="repos_df")
    yield Output(python_code_df, output_name="python_code_df")


@asset
def sampled_repos(
    code2doc_config: dict, repos_df: pd.DataFrame, python_code_df: pd.DataFrame
) -> pd.DataFrame:
    sampled_repos_df = Code2DocSteps.create_repos_sample_df(
        repos_df,
        python_code_df,
        code2doc_config["n_repos_per_task"],
        code2doc_config["min_task_size"],
        code2doc_config["max_task_size"],
        code2doc_config["max_random_baseline_score"],
    )
    return sampled_repos_df


@asset
def generated_readmes(
    config: dict, python_code_df: pd.DataFrame, sampled_repos: pd.DataFrame
) -> pd.DataFrame:
    logging.info(
        f"Generating readmes with code2doc using {config['lm_model_name']}, "
        f"using maximum of {config['files_per_repo']} files per repo"
    )

    with tqdm_logging_redirect():
        generated_readme_df = Code2DocSteps.generate_code2doc_readmes_df(
            python_code_df,
            sampled_repos,
            files_per_repo=config["files_per_repo"],
        )
    return generated_readme_df


@asset
def save_results(generated_readmes: pd.DataFrame):
    generated_readmes.to_json(
        "/tmp/generated_readmes.json", orient="records", lines=True
    )
    logging.info("Code2Doc pipeline completed successfully!")
    logging.info("Generated readmes saved to /tmp/generated_readmes.json")


defs = Definitions(
    assets=[
        code2doc_config,
        prepared_data,
        sampled_repos,
        generated_readmes,
        save_results,
    ],
    config=Code2DocConfig,
)
