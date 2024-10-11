from metaflow import FlowSpec, step, Parameter
import yaml
import logging
import pandas as pd
from github_search.pipelines.steps import Code2DocSteps
from tqdm.contrib.logging import tqdm_logging_redirect


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
        with tqdm_logging_redirect():
            self.repos_df, self.python_code_df = Code2DocSteps.prepare_data_df(
                self.config["repos_df_path"],
                self.config["python_code_path"],
            )
        self.next(self.create_repos_sample)

    @step
    def create_repos_sample(self):
        self.sampled_repos_df = Code2DocSteps.create_repos_sample_df(
            self.repos_df,
            self.python_code_df,
            self.config["n_repos_per_task"],
            self.config["min_task_size"],
            self.config["max_task_size"],
            self.config["max_random_baseline_score"],
        )
        self.next(self.generate_code2doc_readmes)

    @step
    def generate_code2doc_readmes(self):
        logging.info(
            f"Generating readmes with code2doc using {self.config['lm_model_name']}, "
            f"using maximum of {self.config['files_per_repo']} files per repo"
        )

        with tqdm_logging_redirect():
            self.generated_readme_df = Code2DocSteps.generate_code2doc_readmes_df(
                self.python_code_df,
                self.sampled_repos_df,
                files_per_repo=self.config["files_per_repo"],
            )
        self.next(self.end)

    @step
    def end(self):
        # Save the final results
        self.generated_readme_df.to_json(
            "/tmp/generated_readmes.json", orient="records", lines=True
        )
        print("Code2Doc pipeline completed successfully!")
        print(f"Generated readmes saved to /tmp/generated_readmes.json")

    @staticmethod
    def load_config(config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)["pipeline"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Code2DocFlow()
