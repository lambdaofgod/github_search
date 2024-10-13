from pathlib import Path
import pandas as pd
from dagster import (
    multi_asset,
    AssetOut,
    ConfigurableResource,
    AssetExecutionContext,
    Output,
)
from github_search.evaluation.beir_evaluation import (
    EvaluateRetrievalCustom as EvaluateRetrieval,
    CorpusDataLoader,
)
from github_search.evaluation.corpus_utils import (
    filter_dfs_by_cols_in,
    align_dfs,
    prepare_query_data,
    prepare_corpora,
    prepare_librarian_corpora,
)


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


@multi_asset(
    outs={
        "task_queries": AssetOut(),
        "task_qrels": AssetOut(),
    },
    required_resource_keys={"corpus_config"},
)
def ir_data(context: AssetExecutionContext):
    config = context.resources.corpus_config
    data_path = Path(config.data_path).expanduser()

    librarian_signatures_df = pd.read_parquet(config.librarian_signatures_path)

    sample_path = (
        data_path
        / f"code2doc/{config.sample_prefix}/sampled_repos{config.sampled_repos_per_task}.jsonl"
    )
    sampled_repos_df = pd.read_json(sample_path, orient="records", lines=True)
    sample_python_code_df = pd.read_feather(
        data_path / "code" / config.python_code_file
    )

    sample_loader = CorpusDataLoader(
        repos_df_path=data_path
        / f"code2doc/{config.sample_prefix}/sampled_repos5.jsonl",
        generated_readmes_df_path=data_path
        / f"code2doc/{config.sample_prefix}/{config.ir_model_name}_generated_readmes5.jsonl",
        code_df_path=data_path / "code" / config.python_code_file,
    )

    sampled_repos_df, sampled_generated_readmes_df, sample_python_code_df = (
        sample_loader.load_corpus_dfs(librarian_signatures_df["repo"])
    )

    repos_with_all_data = set(sampled_repos_df["repo"]).intersection(
        librarian_signatures_df["repo"]
    )

    sampled_repos_df, sample_python_code_df, sampled_librarian_signatures_df = (
        filter_dfs_by_cols_in(
            [sampled_repos_df, sample_python_code_df, librarian_signatures_df],
            repos_with_all_data,
        )
    )
    sampled_repos_df, sampled_librarian_signatures_df = align_dfs(
        [sampled_repos_df, sampled_librarian_signatures_df]
    )

    task_queries, task_qrels = prepare_query_data(
        sampled_repos_df, min_query_count=config.min_repos_per_task
    )

    yield Output(task_queries, "task_queries")
    yield Output(task_qrels, "task_qrels")


@multi_asset(
    outs={
        "corpora": AssetOut(),
        "corpora_keys": AssetOut(),
    },
    required_resource_keys={"corpus_config"},
    ins={
        "sampled_repos": AssetIn(key_prefix=["code2doc"]),
        "generated_readmes": AssetIn(key_prefix=["code2doc"]),
    },
)
def corpus_information(
    context: AssetExecutionContext,
    sampled_repos: pd.DataFrame,
    generated_readmes: pd.DataFrame,
):
    config = context.resources.corpus_config
    data_path = Path(config.data_path).expanduser()

    librarian_signatures_df = pd.read_parquet(config.librarian_signatures_path)
    sample_python_code_df = pd.read_feather(
        data_path / "code" / config.python_code_file
    )

    repos_with_all_data = set(sampled_repos["repo"]).intersection(
        librarian_signatures_df["repo"]
    )

    sampled_repos_df, sample_python_code_df, sampled_librarian_signatures_df = (
        filter_dfs_by_cols_in(
            [sampled_repos, sample_python_code_df, librarian_signatures_df],
            repos_with_all_data,
        )
    )
    sampled_repos_df, sampled_librarian_signatures_df = align_dfs(
        [sampled_repos_df, sampled_librarian_signatures_df]
    )

    corpora = prepare_corpora(
        sampled_repos_df, generated_readmes, sample_python_code_df
    ) | prepare_librarian_corpora(sampled_repos_df, sampled_librarian_signatures_df)

    corpora_keys = list(corpora.keys())

    yield Output(corpora, output_name="corpora")
    yield Output(corpora_keys, output_name="corpora_keys")
