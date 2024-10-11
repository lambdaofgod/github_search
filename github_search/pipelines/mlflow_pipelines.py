import mlflow
from mlflow.pipelines import Pipeline, PipelineStep
import logging
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

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Code2Doc Pipeline")
    
    with mlflow.start_run():
        pipeline = Code2DocPipeline()
        pipeline.run({
            "repos_df_path": "path/to/repos_df.json",
            "python_code_path": "path/to/python_code.parquet",
            "n_repos_per_task": 10,
            "min_task_size": 5,
            "max_task_size": 500,
            "max_random_baseline_score": 0.5,
            "lm_model_name": "codellama",
            "lm_base_url": "http://localhost:11434",
            "files_per_repo": 10
        })
