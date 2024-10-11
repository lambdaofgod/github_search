from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import yaml
from pathlib import Path
import logging
from github_search.pipelines.steps import Code2DocSteps
import fire


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


config_path = Path(__file__).parent.parent / "configs" / "code2doc_default_config.yaml"
config = load_config(config_path)


def prepare_data(**kwargs):
    Code2DocSteps.prepare_data(
        config["pipeline"]["repos_df_path"],
        config["pipeline"]["python_code_path"],
        "/tmp/repos_output.json",
        "/tmp/selected_python_code.feather",
    )


def create_repos_sample(**kwargs):
    Code2DocSteps.create_repos_sample(
        "/tmp/repos_output.json",
        "/tmp/selected_python_code.feather",
        "/tmp/sampled_repos.json",
        config["pipeline"]["n_repos_per_task"],
        config["pipeline"]["min_task_size"],
        config["pipeline"]["max_task_size"],
        config["pipeline"]["max_random_baseline_score"],
    )


def generate_code2doc_readmes(**kwargs):
    Code2DocSteps.generate_code2doc_readmes(
        "/tmp/sampled_repos.json",
        "/tmp/selected_python_code.feather",
        "/tmp/generated_readmes.json",
        config["pipeline"]["lm_model_name"],
        config["pipeline"]["lm_base_url"],
        files_per_repo=config["pipeline"]["files_per_repo"],
    )


default_args = {
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "schedule": timedelta(days=1),
    "start_date": datetime(2021, 1, 1),
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "code2doc",
    default_args=default_args,
    description="Code2Doc Pipeline",
    catchup=False,
) as dag:
    prepare_data_task = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
    )

    create_repos_sample_task = PythonOperator(
        task_id="create_repos_sample",
        python_callable=create_repos_sample,
    )

    generate_code2doc_readmes_task = PythonOperator(
        task_id="generate_code2doc_readmes",
        python_callable=generate_code2doc_readmes,
    )

    prepare_data_task >> create_repos_sample_task >> generate_code2doc_readmes_task
