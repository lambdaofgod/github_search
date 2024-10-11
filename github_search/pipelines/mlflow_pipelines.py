import mlflow
from mlflow.recipes import Recipe
import logging
import yaml
from pathlib import Path
import fire
from github_search.pipelines.steps import Code2DocSteps

class Code2DocRecipe:
    def __init__(self, config):
        self.config = config

    def prepare_data(self):
        repos_output_path = "repos_output.json"
        selected_python_code_path = "selected_python_code.feather"
        
        Code2DocSteps.prepare_data(
            self.config['repos_df_path'],
            self.config['python_code_path'],
            repos_output_path,
            selected_python_code_path
        )
        
        mlflow.log_artifact(repos_output_path)
        mlflow.log_artifact(selected_python_code_path)

    def create_repos_sample(self):
        sampled_repos_path = "sampled_repos.json"
        
        Code2DocSteps.create_repos_sample(
            "repos_output.json",
            "selected_python_code.feather",
            sampled_repos_path,
            self.config['n_repos_per_task'],
            self.config['min_task_size'],
            self.config['max_task_size'],
            self.config['max_random_baseline_score']
        )
        
        mlflow.log_artifact(sampled_repos_path)

    def generate_code2doc_readmes(self):
        generated_readmes_path = "generated_readmes.json"
        
        logging.info(
            f"Generating readmes with code2doc using {self.config['lm_model_name']}, "
            f"using maximum of {self.config['files_per_repo']} files per repo"
        )
        Code2DocSteps.generate_code2doc_readmes(
            "sampled_repos.json",
            "selected_python_code.feather",
            generated_readmes_path,
            self.config['lm_model_name'],
            self.config['lm_base_url'],
            files_per_repo=self.config['files_per_repo']
        )
        
        mlflow.log_artifact(generated_readmes_path)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_recipe(config_path):
    config = load_config(config_path)
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run():
        recipe = Code2DocRecipe(config['pipeline'])
        recipe.prepare_data()
        recipe.create_repos_sample()
        recipe.generate_code2doc_readmes()

def main(config_path="github_search/pipelines/configs/code2doc_default_config.yaml"):
    """
    Run Code2Doc MLFlow Recipe
    
    Args:
        config_path (str): Path to the YAML configuration file
    """
    run_recipe(config_path)

if __name__ == "__main__":
    fire.Fire(main)
