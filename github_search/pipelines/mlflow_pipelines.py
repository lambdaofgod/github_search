import mlflow
from mlflow.pipelines import Pipeline, PipelineStep
import logging
import yaml
from pathlib import Path
from github_search.pipelines.steps import Code2DocSteps

class Code2DocPipeline(Pipeline):
    def __init__(self):
        super().__init__("Code2Doc Pipeline")

    def steps(self):
        return [
            PrepareDataStep(),
            CreateReposSampleStep(),
            GenerateCode2DocReadmesStep()
        ]

class PrepareDataStep(PipelineStep):
    def run(self, repos_df_path, python_code_path):
        repos_output_path = mlflow.artifacts.download_artifacts("repos_output.json")
        selected_python_code_path = mlflow.artifacts.download_artifacts("selected_python_code.feather")
        
        Code2DocSteps.prepare_data(
            repos_df_path,
            python_code_path,
            repos_output_path,
            selected_python_code_path
        )
        
        mlflow.log_artifact(repos_output_path, "repos_output.json")
        mlflow.log_artifact(selected_python_code_path, "selected_python_code.feather")

class CreateReposSampleStep(PipelineStep):
    def run(self, n_repos_per_task, min_task_size, max_task_size, max_random_baseline_score):
        repos_df_path = mlflow.artifacts.download_artifacts("repos_output.json")
        selected_python_code_path = mlflow.artifacts.download_artifacts("selected_python_code.feather")
        sampled_repos_path = mlflow.artifacts.download_artifacts("sampled_repos.json")
        
        Code2DocSteps.create_repos_sample(
            repos_df_path,
            selected_python_code_path,
            sampled_repos_path,
            n_repos_per_task,
            min_task_size,
            max_task_size,
            max_random_baseline_score
        )
        
        mlflow.log_artifact(sampled_repos_path, "sampled_repos.json")

class GenerateCode2DocReadmesStep(PipelineStep):
    def run(self, lm_model_name, lm_base_url, files_per_repo):
        sampled_repos_df_path = mlflow.artifacts.download_artifacts("sampled_repos.json")
        selected_python_code_path = mlflow.artifacts.download_artifacts("selected_python_code.feather")
        generated_readmes_path = mlflow.artifacts.download_artifacts("generated_readmes.json")
        
        logging.info(
            f"Generating readmes with code2doc using {lm_model_name}, using maximum of {files_per_repo} files per repo"
        )
        Code2DocSteps.generate_code2doc_readmes(
            sampled_repos_df_path,
            selected_python_code_path,
            generated_readmes_path,
            lm_model_name,
            lm_base_url,
            files_per_repo=files_per_repo
        )
        
        mlflow.log_artifact(generated_readmes_path, "generated_readmes.json")

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_pipeline(config_path):
    config = load_config(config_path)
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        pipeline = Code2DocPipeline()
        pipeline.run(config['pipeline'])

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Code2Doc MLFlow Pipeline")
    parser.add_argument("--config", type=str, default="github_search/pipelines/configs/code2doc_default_config.yaml",
                        help="Path to the YAML configuration file")
    args = parser.parse_args()
    
    run_pipeline(args.config)
