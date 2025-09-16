from pathlib import Path
import pandas as pd
import json
from dagster import (
    asset,
    multi_asset,
    AssetOut,
    AssetExecutionContext,
    Output,
    AssetIn,
)

from github_search.evaluation.corpus_utils import (
    prepare_query_data,
    create_corpora,
    create_corpora_df,
)
from github_search.pipelines.dagster.resources import CorpusConfig


@asset
def repos_with_representations_df(
    config: CorpusConfig,
    sampled_repos: pd.DataFrame,
    generated_readmes: pd.DataFrame,
    repomap_generated_readmes: pd.DataFrame,
    flat_generated_readmes: pd.DataFrame,
    flat_repomap_generated_readmes: pd.DataFrame,
    librarian_signatures_df: pd.DataFrame,
    selected_python_code_df: pd.DataFrame,
    dependency_signature_generated_readmes: pd.DataFrame,
    flat_dependency_signature_generated_readmes: pd.DataFrame,
):
    valid_repos = sampled_repos[
        sampled_repos["tasks"].apply(len) <= config.max_repo_tasks
    ]
    corpora_df = create_corpora_df(
        valid_repos,
        librarian_signatures_df,
        generated_readmes,
        repomap_generated_readmes,
        flat_generated_readmes,
        flat_repomap_generated_readmes,
        selected_python_code_df,
        dependency_signature_generated_readmes,
        flat_dependency_signature_generated_readmes,
    )
    return corpora_df


@multi_asset(
    outs={
        "task_queries": AssetOut(),
        "task_qrels": AssetOut(),
    },
)
def ir_data(config: CorpusConfig, repos_with_representations_df: pd.DataFrame):
    """
    information retrieval data: queries and query relevances
    """

    task_queries, task_qrels = prepare_query_data(
        repos_with_representations_df, min_query_count=config.min_repos_per_task
    )

    yield Output(task_queries, "task_queries")
    yield Output(task_qrels, "task_qrels")


@asset
def selected_python_code_df(config: CorpusConfig):
    data_path = Path(config.data_path).expanduser()
    python_code_df = pd.read_feather(data_path / "code" / config.python_code_file)
    files_per_repo = 10
    code_col = "selected_code"

    def extract_summary(group):
        """
        Extracts the summary for a single repository group.
        """
        selected_files = group.head(files_per_repo)
        return "\n\n".join(
            [
                f"file {path}\n```\n{code}\n```"
                for path, code in zip(
                    selected_files["path"],
                    selected_files[code_col],
                )
            ]
        )

    return (
        python_code_df.groupby("repo_name")
        .apply(extract_summary)
        .rename("selected_code")
        .reset_index()
    )


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
    repomap_generated_readmes: pd.DataFrame,
    flat_generated_readmes: pd.DataFrame,
    flat_repomap_generated_readmes: pd.DataFrame,
    librarian_signatures_df: pd.DataFrame,
    selected_python_code_df: pd.DataFrame,
    dependency_signature_generated_readmes: pd.DataFrame,
    flat_dependency_signature_generated_readmes: pd.DataFrame,
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

    valid_repos = sampled_repos[
        sampled_repos["tasks"].apply(len) <= config.max_repo_tasks
    ]

    corpora = create_corpora(
        valid_repos,
        librarian_signatures_df,
        generated_readmes,
        repomap_generated_readmes,
        flat_generated_readmes,
        flat_repomap_generated_readmes,
        selected_python_code_df,
        dependency_signature_generated_readmes,
        flat_dependency_signature_generated_readmes,
    )
    context.add_output_metadata({"corpora_keys": list(corpora.keys())})

    return json.dumps(corpora)
