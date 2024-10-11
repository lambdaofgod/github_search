from metaflow import FlowSpec, step, Parameter
import yaml
from pathlib import Path
import logging
from github_search.pipelines.steps import Code2DocSteps


class Code2DocFlow(FlowSpec):
    config_path = Parameter(
        "config_path",
        default="github_search/pipelines/configs/code2doc_default_config.yaml",
        help="Path to the YAML configuration file",
    )

    @step
    def start(self):
        self.config = self.load_config(self.config_path)
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        self.repos_output_path = "/tmp/repos_output.json"
        self.selected_python_code_path = "/tmp/selected_python_code.feather"

        Code2DocSteps.prepare_data(
            self.config["repos_df_path"],
            self.config["python_code_path"],
            self.repos_output_path,
            self.selected_python_code_path,
        )
        self.next(self.create_repos_sample)

    @step
    def create_repos_sample(self):
        self.sampled_repos_path = "/tmp/sampled_repos.json"

        Code2DocSteps.create_repos_sample(
            self.repos_output_path,
            self.selected_python_code_path,
            self.sampled_repos_path,
            self.config["n_repos_per_task"],
            self.config["min_task_size"],
            self.config["max_task_size"],
            self.config["max_random_baseline_score"],
        )
        self.next(self.generate_code2doc_readmes)

    @step
    def generate_code2doc_readmes(self):
        self.generated_readmes_path = "/tmp/generated_readmes.json"

        logging.info(
            f"Generating readmes with code2doc using {self.config['lm_model_name']}, "
            f"using maximum of {self.config['files_per_repo']} files per repo"
        )
        Code2DocSteps.generate_code2doc_readmes(
            self.sampled_repos_path,
            self.selected_python_code_path,
            self.generated_readmes_path,
            self.config["lm_model_name"],
            self.config["lm_base_url"],
            files_per_repo=self.config["files_per_repo"],
        )
        self.next(self.end)

    @step
    def end(self):
        print("Code2Doc pipeline completed successfully!")

    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)["pipeline"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Code2DocFlow()
