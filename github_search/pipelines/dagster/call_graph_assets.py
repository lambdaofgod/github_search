import pandas as pd
import tqdm
from typing import List
from dagster import (
    asset,
    AssetExecutionContext,
)
from github_search.dependency_records.python_call_graph import GraphExtractor
from github_search.dependency_records.python_call_graph_analysis import (
    GraphCentralityAnalyzer,
    get_dependency_signatures,
)
from github_search.pipelines.dagster.resources import (
    DependencyGraphConfig,
    LibrarianConfig,
)
from github_search.lms.librarian import (
    OllamaTypedPredict,
    create_fewshot_prompts,
    sample_context_repo_idxs,
)
from pydantic import BaseModel


@asset
def graph_dependencies_df(
    context: AssetExecutionContext, python_files_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract repository dependencies using GraphExtractor.
    """
    context.log.info(
        f"Extracting dependencies from {len(python_files_df)} Python files"
    )

    dependencies_df = GraphExtractor.extract_repo_dependencies_df(python_files_df)

    context.log.info(f"Extracted {len(dependencies_df)} dependency relationships")
    context.add_output_metadata(
        {
            "num_dependencies": len(dependencies_df),
            "dependency_types": (
                dependencies_df.get("edge_type", pd.Series()).value_counts().to_dict()
                if "edge_type" in dependencies_df.columns
                else {}
            ),
        }
    )

    return dependencies_df


@asset
def centralities_df(
    context: AssetExecutionContext,
    graph_dependencies_df: pd.DataFrame,
    config: DependencyGraphConfig,
) -> pd.DataFrame:
    """
    Analyze graph centralities using PageRank algorithm.
    """
    # Default edge types and limits from the org file
    edge_type_limits = {
        etype: config.n_nodes_per_type for etype in config.centrality_edge_types
    }

    centrality_method = getattr(config, "centrality_method", "pagerank")

    context.log.info(f"Analyzing centralities using {centrality_method} method")
    context.log.info(f"Edge type limits: {edge_type_limits}")

    analyzer = GraphCentralityAnalyzer(centrality_method=centrality_method)
    centralities_df = analyzer.analyze_centralities(
        graph_dependencies_df, edge_type_limits
    )

    context.log.info(f"Computed centralities for {len(centralities_df)} nodes")
    context.add_output_metadata(
        {
            "num_centrality_nodes": len(centralities_df),
            "centrality_method": centrality_method,
        }
    )

    return centralities_df


@asset
def dependency_signatures(
    context: AssetExecutionContext, centralities_df: pd.DataFrame
) -> pd.Series:
    """
    Extracts dependency signatures from a DataFrame of node centralities.
    """

    signatures = get_dependency_signatures(centralities_df)
    context.log.info(f"Extracted {len(signatures)} dependency signatures")

    context.add_output_metadata(
        {
            "example_signature": signatures.iloc[0],
        }
    )

    return signatures


@asset
def repos_with_dependency_signatures_df(
    context: AssetExecutionContext,
    dependency_signatures: pd.Series,
    sampled_repos: pd.DataFrame,
):

    df = pd.DataFrame({"dependency_signature": dependency_signatures}).merge(
        sampled_repos, left_index=True, right_on="repo", how="inner"
    )

    context.add_output_metadata(
        {
            "n_repos": len(df),
        }
    )
    return df


@asset
def context_repo_idxs(
    context: AssetExecutionContext,
    sampled_repos: pd.DataFrame,
    graph_dependencies_df: pd.DataFrame,
    config: LibrarianConfig,
):
    repos = set(graph_dependencies_df["repo_name"]).intersection(sampled_repos["repo"])
    context.add_output_metadata(
        {
            "n_repos": len(repos),
        }
    )

    return sample_context_repo_idxs(len(repos), config.n_shot)


@asset
def librarian_tasks(
    context: AssetExecutionContext,
    repos_with_dependency_signatures_df: pd.DataFrame,
    context_repo_idxs: List[List[int]],
    config: LibrarianConfig,
):

    class LibrarianTasks(BaseModel):
        tasks: List[str]

    librarian = OllamaTypedPredict(
        model_name="qwen2.5:7b-instruct", output_cls=LibrarianTasks
    )

    fewshot_template = """
    Based on the examples of repos with selected information and their tags, generate tags for the repo without tags

    {% for repo_record in context_repo_records %}
    {{repo_record["dependency_signature"]}}
    tasks:
    {{repo_record["tasks"]}}
    {% endfor %}

    {{target_repo_record["dependency_signature"]}}
    tasks:
    """

    librarian_prompts = create_fewshot_prompts(
        repos_with_dependency_signatures_df, fewshot_template, context_repo_idxs
    )
    librarian_signature_values = [
        librarian(prompt).tasks for prompt in tqdm.tqdm(librarian_prompts.values)
    ]

    return pd.DataFrame(
        {
            "repo": librarian_prompts.index,
            "generated_tasks": [", ".join(s) for s in librarian_signature_values],
            "generated_tasks_raw": librarian_signature_values,
        }
    )


@asset
def librarian_signatures_df(
    librarian_tasks: pd.DataFrame, repos_with_dependency_signatures_df
):
    df = librarian_tasks.merge(repos_with_dependency_signatures_df, on="repo")
    df["repository_signature"] = (
        df["dependency_signature"] + "\ntasks:\n" + df["generated_tasks"]
    )
    return df
