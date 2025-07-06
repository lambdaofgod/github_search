from typing import Any, Dict, List
from pathlib import Path
import pandas as pd
import json
import tqdm
from dagster import (
    asset,
    multi_asset,
    AssetOut,
    ConfigurableResource,
    AssetExecutionContext,
    Output,
    AssetIn,
)
from github_search.python_call_graph import GraphExtractor
from github_search.python_call_graph_analysis import GraphCentralityAnalyzer
from github_search.pipelines.dagster.resources import CorpusConfig


@asset
def python_files_df(context: AssetExecutionContext, config: CorpusConfig) -> pd.DataFrame:
    """
    Load Python files dataframe from configured path.
    Expected columns: ['path', 'content', 'repo_name']
    """
    data_path = Path(config.data_path).expanduser()
    python_files_path = data_path / config.python_files_file
    
    context.log.info(f"Loading Python files from {python_files_path}")
    
    # Load the dataframe - assuming it's saved as parquet/feather/csv
    if python_files_path.suffix == '.parquet':
        df = pd.read_parquet(python_files_path)
    elif python_files_path.suffix == '.feather':
        df = pd.read_feather(python_files_path)
    elif python_files_path.suffix == '.csv':
        df = pd.read_csv(python_files_path)
    else:
        raise ValueError(f"Unsupported file format: {python_files_path.suffix}")
    
    context.log.info(f"Loaded {len(df)} Python files from {df['repo_name'].nunique()} repositories")
    context.add_output_metadata({
        "num_files": len(df),
        "num_repos": df['repo_name'].nunique(),
        "avg_files_per_repo": df['repo_name'].value_counts().mean()
    })
    
    return df.sort_values("repo_name")


@asset
def graph_dependencies_df(
    context: AssetExecutionContext, 
    python_files_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract repository dependencies using GraphExtractor.
    """
    context.log.info(f"Extracting dependencies from {len(python_files_df)} Python files")
    
    dependencies_df = GraphExtractor.extract_repo_dependencies_df(python_files_df)
    
    context.log.info(f"Extracted {len(dependencies_df)} dependency relationships")
    context.add_output_metadata({
        "num_dependencies": len(dependencies_df),
        "dependency_types": dependencies_df.get('edge_type', pd.Series()).value_counts().to_dict() if 'edge_type' in dependencies_df.columns else {}
    })
    
    return dependencies_df


@asset
def centralities_df(
    context: AssetExecutionContext,
    graph_dependencies_df: pd.DataFrame,
    config: CorpusConfig
) -> pd.DataFrame:
    """
    Analyze graph centralities using PageRank algorithm.
    """
    # Default edge types and limits from the org file
    edge_type_limits = {
        "repo-file": 10,
        "file-class": 10, 
        "file-function": 10,
        "file-import": 10
    }
    
    # Override with config if available
    if hasattr(config, 'centrality_edge_types'):
        edge_type_limits.update(config.centrality_edge_types)
    
    centrality_method = getattr(config, 'centrality_method', 'pagerank')
    
    context.log.info(f"Analyzing centralities using {centrality_method} method")
    context.log.info(f"Edge type limits: {edge_type_limits}")
    
    analyzer = GraphCentralityAnalyzer(centrality_method=centrality_method)
    centralities_df = analyzer.analyze_centralities(
        graph_dependencies_df, 
        edge_type_limits
    )
    
    context.log.info(f"Computed centralities for {len(centralities_df)} nodes")
    context.add_output_metadata({
        "num_centrality_nodes": len(centralities_df),
        "centrality_method": centrality_method,
        "edge_type_limits": edge_type_limits
    })
    
    return centralities_df


@asset
def call_graph_analysis_results(
    context: AssetExecutionContext,
    centralities_df: pd.DataFrame,
    graph_dependencies_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Combine call graph analysis results into a summary dictionary.
    """
    results = {
        "centralities_summary": {
            "num_nodes": len(centralities_df),
            "top_nodes": centralities_df.head(10).to_dict('records') if len(centralities_df) > 0 else []
        },
        "dependencies_summary": {
            "num_edges": len(graph_dependencies_df),
            "edge_types": graph_dependencies_df.get('edge_type', pd.Series()).value_counts().to_dict() if 'edge_type' in graph_dependencies_df.columns else {}
        }
    }
    
    context.log.info("Call graph analysis completed")
    context.add_output_metadata(results)
    
    return results
