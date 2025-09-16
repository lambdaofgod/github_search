from dagster import (
    asset,
    multi_asset,
    Output,
    AssetOut,
    AssetExecutionContext,
)
import logging
import pandas as pd
from github_search.pipelines.steps import Code2DocSteps
from github_search.pipelines.dagster.resources import InputDataConfig
from tqdm.contrib.logging import tqdm_logging_redirect
import pathlib

logging.basicConfig(level=logging.INFO)


@multi_asset(
    outs={"repos_df": AssetOut(), "python_code_df": AssetOut()},
)
def prepared_data(config: InputDataConfig):
    """
    repos and python code dataframes
    """
    with tqdm_logging_redirect():
        repos_df, python_code_df = Code2DocSteps.prepare_data_df(
            config.repos_df_path,
            config.python_code_path,
        )
    yield Output(repos_df, output_name="repos_df")
    yield Output(python_code_df, output_name="python_code_df")


@asset
def python_files_df(
    context: AssetExecutionContext, config: InputDataConfig
) -> pd.DataFrame:
    """
    Load Python files dataframe from configured path.
    Expected columns: ['path', 'content', 'repo_name']
    """
    python_files_path = pathlib.Path(config.python_files_path)

    context.log.info(f"Loading Python files from {python_files_path}")

    # Load the dataframe - assuming it's saved as parquet/feather/csv
    if python_files_path.suffix == ".parquet":
        df = pd.read_parquet(python_files_path)
    elif python_files_path.suffix == ".feather":
        df = pd.read_feather(python_files_path)
    elif python_files_path.suffix == ".csv":
        df = pd.read_csv(python_files_path)
    else:
        raise ValueError(f"Unsupported file format: {python_files_path.suffix}")

    context.log.info(
        f"Loaded {len(df)} Python files from {df['repo_name'].nunique()} repositories"
    )
    context.add_output_metadata(
        {
            "num_files": len(df),
            "num_repos": df["repo_name"].nunique(),
            "avg_files_per_repo": float(df["repo_name"].value_counts().mean()),
        }
    )

    return df.sort_values("repo_name")
