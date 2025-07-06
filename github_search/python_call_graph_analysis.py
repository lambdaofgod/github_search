from typing import Literal, Dict, Optional, List, Union
from pydantic import BaseModel, Field
import networkx as nx
import pandas as pd
from pandera.typing import DataFrame
from github_search.python_call_graph import DependencyGraphSchema


class GraphCentralityAnalyzer(BaseModel):

    centrality_method: Literal["pagerank", "degree"]

    def analyze_centralities(
        self,
        edge_df: DataFrame[DependencyGraphSchema],
        selected_edge_types: Union[List[str], Dict[str, int]],
        topk: Optional[int] = 10,
    ) -> pd.DataFrame:
        """
        Calculate centrality measures for each repository separately.
        Calculates centrality on the full graph but only returns nodes from selected edge types.

        Args:
            edge_df: DataFrame with dependency edges
            selected_edge_types: Edge types to include in results. Can be:
                - List[str]: list of edge types, all using the same topk
                - Dict[str, int]: mapping of edge types to their specific topk values
            topk: Number of top nodes to return per repository (used when selected_edge_types is a list)

        Returns:
            DataFrame with columns [repo_name, node, centrality_score, edge_type]
        """
        centrality_results = []

        # Group by repository and calculate centralities for each repo
        for repo_name in edge_df["repo_name"].unique():
            # Calculate centrality based on method using full graph
            centralities = self._calculate_centralities(repo_name, edge_df)

            # Skip if no centralities calculated (empty graph)
            if not centralities:
                continue

            # Process centralities for selected edge types
            repo_results = self._process_centralities_for_repo(
                repo_name, edge_df, centralities, selected_edge_types, topk
            )
            centrality_results.extend(repo_results)

        return pd.DataFrame(centrality_results)

    def _calculate_centralities(self, repo_name: str, edge_df: DataFrame[DependencyGraphSchema]) -> Dict[str, float]:
        """
        Calculate centrality measures for a given repository.

        Args:
            repo_name: Name of the repository to analyze
            edge_df: DataFrame with dependency edges

        Returns:
            Dictionary mapping node names to centrality scores
        """
        # Create full graph for this repository (all edge types)
        G = self.load_graph_from_edge_df(repo_name, edge_df)

        # Skip empty graphs
        if G.number_of_nodes() == 0:
            return {}

        if self.centrality_method == "pagerank":
            return nx.pagerank(G)
        elif self.centrality_method == "degree":
            return dict(G.degree())
        else:
            raise ValueError(f"Unknown centrality method: {self.centrality_method}")

    def _process_centralities_for_repo(
        self,
        repo_name: str,
        edge_df: DataFrame[DependencyGraphSchema],
        centralities: Dict[str, float],
        selected_edge_types: Union[List[str], Dict[str, int]],
        topk: Optional[int],
    ) -> List[Dict[str, any]]:
        """
        Process centralities for a repository by filtering to selected edge types and returning top-k results.

        Args:
            repo_name: Name of the repository
            edge_df: DataFrame with dependency edges
            centralities: Dictionary mapping node names to centrality scores
            selected_edge_types: Edge types to include. Can be list or dict with topk values
            topk: Number of top nodes to return (used when selected_edge_types is a list)

        Returns:
            List of dictionaries with repo_name, node, centrality_score, and edge_type
        """
        repo_edges = edge_df[edge_df["repo_name"] == repo_name]

        if isinstance(selected_edge_types, dict):
            return self._process_per_edge_type_topk(
                repo_name, repo_edges, centralities, selected_edge_types
            )
        else:
            return self._process_unified_topk(
                repo_name, repo_edges, centralities, selected_edge_types, topk
            )

    def _process_per_edge_type_topk(
        self,
        repo_name: str,
        repo_edges: pd.DataFrame,
        centralities: Dict[str, float],
        edge_types_with_topk: Dict[str, int],
    ) -> List[Dict[str, any]]:
        """
        Process centralities with per-edge-type topk limits.

        Args:
            repo_name: Name of the repository
            repo_edges: DataFrame with edges for this repository
            centralities: Dictionary mapping node names to centrality scores
            edge_types_with_topk: Dictionary mapping edge types to their topk limits

        Returns:
            List of dictionaries with repo_name, node, centrality_score, and edge_type
        """
        results = []
        
        for edge_type, edge_type_topk in edge_types_with_topk.items():
            edge_type_results = self._get_topk_for_edge_type(
                repo_name, repo_edges, centralities, edge_type, edge_type_topk
            )
            results.extend(edge_type_results)
        
        return results

    def _process_unified_topk(
        self,
        repo_name: str,
        repo_edges: pd.DataFrame,
        centralities: Dict[str, float],
        selected_edge_types: List[str],
        topk: Optional[int],
    ) -> List[Dict[str, any]]:
        """
        Process centralities with unified topk limit across all edge types.

        Args:
            repo_name: Name of the repository
            repo_edges: DataFrame with edges for this repository
            centralities: Dictionary mapping node names to centrality scores
            selected_edge_types: List of edge types to include in results
            topk: Number of top nodes to return

        Returns:
            List of dictionaries with repo_name, node, and centrality_score
        """
        # Get all nodes from selected edge types
        selected_edges = repo_edges[repo_edges["edge_type"].isin(selected_edge_types)]
        selected_nodes = set(
            selected_edges["source"].tolist() + selected_edges["target"].tolist()
        )

        # Filter and sort centralities
        filtered_centralities = {
            node: score
            for node, score in centralities.items()
            if node in selected_nodes
        }

        sorted_centralities = sorted(
            filtered_centralities.items(), key=lambda x: x[1], reverse=True
        )
        if topk is not None:
            sorted_centralities = sorted_centralities[:topk]

        # Convert to list of records (without edge_type for backward compatibility)
        results = []
        for node, score in sorted_centralities:
            results.append({
                "repo_name": repo_name,
                "node": node,
                "centrality_score": score
            })

        return results

    def _get_topk_for_edge_type(
        self,
        repo_name: str,
        repo_edges: pd.DataFrame,
        centralities: Dict[str, float],
        edge_type: str,
        topk: int,
    ) -> List[Dict[str, any]]:
        """
        Get top-k centrality results for a specific edge type.

        Args:
            repo_name: Name of the repository
            repo_edges: DataFrame with edges for this repository
            centralities: Dictionary mapping node names to centrality scores
            edge_type: Specific edge type to process
            topk: Number of top nodes to return for this edge type

        Returns:
            List of dictionaries with repo_name, node, centrality_score, and edge_type
        """
        # Get nodes for this specific edge type
        edge_type_edges = repo_edges[repo_edges["edge_type"] == edge_type]
        edge_type_nodes = set(
            edge_type_edges["source"].tolist() + edge_type_edges["target"].tolist()
        )

        # Filter centralities to nodes from this edge type
        filtered_centralities = {
            node: score
            for node, score in centralities.items()
            if node in edge_type_nodes
        }

        # Sort and take top-k for this edge type
        sorted_centralities = sorted(
            filtered_centralities.items(), key=lambda x: x[1], reverse=True
        )
        if topk is not None:
            sorted_centralities = sorted_centralities[:topk]

        # Add results with edge type information
        results = []
        for node, score in sorted_centralities:
            results.append({
                "repo_name": repo_name,
                "node": node,
                "centrality_score": score,
                "edge_type": edge_type
            })

        return results

    def load_graph_from_edge_df(
        self,
        repo_name: str,
        edge_df: DataFrame[DependencyGraphSchema],
    ) -> nx.DiGraph:
        """
        Create a NetworkX directed graph from the dependency edge DataFrame.
        Uses all edge types for centrality calculation.

        Args:
            repo_name: Name of the repository to filter by
            edge_df: DataFrame with columns [repo_name, target, source, edge_type]

        Returns:
            NetworkX DiGraph with edges and edge attributes
        """
        G = nx.DiGraph()
        repo_edge_df = edge_df[edge_df["repo_name"] == repo_name]

        # Add edges with attributes (all edge types for accurate centrality)
        for _, row in repo_edge_df.iterrows():
            source = row["source"]
            target = row["target"]
            edge_type = row["edge_type"]

            # Add edge with attributes
            G.add_edge(source, target, edge_type=edge_type, repo_name=repo_name)

        return G
