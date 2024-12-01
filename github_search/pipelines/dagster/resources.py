from dagster import (
    asset,
    multi_asset,
    Definitions,
    Output,
    AssetOut,
    AssetIn,
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


class Code2DocConfig(ConfigurableResource):
    repos_df_path: str = "output/paperswithcode_with_readmes.json.gz"
    python_code_path: str = "output/repo_selected_files.parquet"
    n_repos_per_task: int = 10
    min_task_size: int = 1
    max_task_size: int = 10
    max_random_baseline_score: float = 0.3
    lm_model_name: str = "codellama"
    lm_base_url: str = "http://localhost:11434"
    files_per_repo: int = 10


class CorpusConfig(ConfigurableResource):
    data_path: str = "output"
    librarian_signatures_path: str = (
        "/home/kuba/Projects/uhackathons/fastrag_util/data/librarian_signatures.parquet"
    )
    ir_model_name: str = "codellama_repomaps"
    sample_prefix: str = "sample_per_task_5_repos"
    sampled_repos_per_task: int = 20
    min_repos_per_task: int = 10
    python_code_file: str = "python_files_with_selected_code.feather"
