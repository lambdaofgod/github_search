from dagster import job, op, In, Out, Config
import yaml
import logging
import pandas as pd
from github_search.pipelines.steps import Code2DocSteps
from tqdm.contrib.logging import tqdm_logging_redirect


class Code2DocConfig(Config):
    config_path: str = "github_search/pipelines/configs/code2doc_default_config.yaml"


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)["pipeline"]


@op(config_schema=Code2DocConfig)
def start(context) -> dict:
    config = load_config(context.op_config.config_path)
    return config


@op(ins={"config": In()})
def prepare_data(config: dict) -> tuple:
    with tqdm_logging_redirect():
        repos_df, python_code_df = Code2DocSteps.prepare_data_df(
            config["repos_df_path"],
            config["python_code_path"],
        )
    return repos_df, python_code_df


@op(ins={"config": In(), "data": In()})
def create_repos_sample(config: dict, data: tuple) -> pd.DataFrame:
    repos_df, python_code_df = data
    sampled_repos_df = Code2DocSteps.create_repos_sample_df(
        repos_df,
        python_code_df,
        config["n_repos_per_task"],
        config["min_task_size"],
        config["max_task_size"],
        config["max_random_baseline_score"],
    )
    return sampled_repos_df


@op(ins={"config": In(), "python_code_df": In(), "sampled_repos_df": In()})
def generate_code2doc_readmes(config: dict, python_code_df: pd.DataFrame, sampled_repos_df: pd.DataFrame) -> pd.DataFrame:
    logging.info(
        f"Generating readmes with code2doc using {config['lm_model_name']}, "
        f"using maximum of {config['files_per_repo']} files per repo"
    )

    with tqdm_logging_redirect():
        generated_readme_df = Code2DocSteps.generate_code2doc_readmes_df(
            python_code_df,
            sampled_repos_df,
            files_per_repo=config["files_per_repo"],
        )
    return generated_readme_df


@op(ins={"generated_readme_df": In()})
def save_results(generated_readme_df: pd.DataFrame):
    generated_readme_df.to_json(
        "/tmp/generated_readmes.json", orient="records", lines=True
    )
    logging.info("Code2Doc pipeline completed successfully!")
    logging.info("Generated readmes saved to /tmp/generated_readmes.json")


@job
def code2doc_job():
    config = start()
    repos_df, python_code_df = prepare_data(config)
    sampled_repos_df = create_repos_sample(config, (repos_df, python_code_df))
    generated_readme_df = generate_code2doc_readmes(config, python_code_df, sampled_repos_df)
    save_results(generated_readme_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = code2doc_job.execute_in_process()
