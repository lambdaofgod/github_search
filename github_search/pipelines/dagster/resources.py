from dagster import (
    asset,
    multi_asset,
    Definitions,
    Output,
    AssetOut,
    AssetIn,
    Config,
    ConfigurableResource,
    AssetExecutionContext,
)

import phoenix as px


class PhoenixTracker(ConfigurableResource):
    port: int = 6006

    def launch(self):
        px.launch_app(port=self.port)

    def get_traces_df(self):
        return px.Client().get_trace_dataset().dataframe


class Code2DocDataConfig(Config):
    repos_df_path: str = "output/paperswithcode_with_readmes.json.gz"
    python_code_path: str = "output/repo_selected_files.parquet"
    repomaps_path: str = "output/aider/selected_repo_maps_1024.json"
    n_repos_per_task: int = 10
    min_task_size: int = 5
    max_task_size: int = 500
    max_random_baseline_score: float = 0.3


class Code2DocModelConfig(Config):
    lm_model_name: str = "qwen2.5:7b-instruct"
    lm_base_url: str = "http://localhost:11434"
    files_per_repo: int = 10
    is_debug_run: bool = False  # if True it will generate readmes for only a few repos


class CorpusConfig(Config):
    data_path: str = "output"
    librarian_signatures_path: str = "output/dependency_representations.parquet"
    ir_model_name: str = "codellama_repomaps"
    sample_prefix: str = "sample_per_task_5_repos"
    sampled_repos_per_task: int = 10
    min_repos_per_task: int = 5
    python_code_file: str = "python_files_with_selected_code.feather"
    max_repo_tasks: int = 10
