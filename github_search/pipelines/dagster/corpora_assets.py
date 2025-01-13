from typing import Any, Dict, List
from pathlib import Path
import pandas as pd
import json
from dagster import (
    asset,
    multi_asset,
    AssetOut,
    ConfigurableResource,
    AssetExecutionContext,
    Output,
    AssetIn,
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
from github_search.pipelines.dagster.resources import CorpusConfig


@multi_asset(
    outs={
        "task_queries": AssetOut(),
        "task_qrels": AssetOut(),
    },
)
def ir_data(context: AssetExecutionContext):
    """
    information retrieval data: queries and query relevances
    """
    config = context.config.corpus_config
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


@asset(
    ins={
        "sampled_repos": AssetIn(key="sampled_repos"),
        "generated_readmes": AssetIn(key="generated_readmes"),
    },
)
def corpus_information(
    context: AssetExecutionContext,
    config: CorpusConfig,
    sampled_repos: pd.DataFrame,
    generated_readmes: pd.DataFrame,
) -> str:
    """
    texts:
    - READMEs
    - corpora extracted with dependency graph
    - librarian corpora
    - code2doc corpora

    WARNING
    right now it's dumped to string as there are some problems
    with dumping dictionaries...
    """
    data_path = Path(config.data_path).expanduser()

    sampled_repos = sampled_repos[
        sampled_repos["tasks"].apply(len) <= config.max_repo_tasks
    ]

    librarian_signatures_df = pd.read_parquet(config.librarian_signatures_path)
    sample_python_code_df = pd.read_feather(
        data_path / "code" / config.python_code_file
    )

    repos_with_all_data = set(sampled_repos["repo"]).intersection(
        librarian_signatures_df["repo"]
    )
    sampled_repos = sampled_repos[sampled_repos["repo"].isin(repos_with_all_data)]

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

    with open("/tmp/corpora.json", "w") as f:
        json.dump(corpora, f)
    context.add_output_metadata({"corpora_keys": corpora_keys})

    return json.dumps(corpora)
