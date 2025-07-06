import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path


def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def compare_centrality_measures(centrality_df, top_k=10):
    """
    Compare different centrality measures for each repository using Jaccard similarity

    Args:
        centrality_df: DataFrame with centrality results
        top_k: Number of top nodes to consider for each measure

    Returns:
        DataFrame with Jaccard similarity scores between measures for each repository
    """
    # Get unique repositories and measures
    repositories = centrality_df["repository"].unique()
    measures = centrality_df["measure"].unique()

    # Create a DataFrame to store results
    results = []

    # For each repository, compare the top nodes for each pair of measures
    for repo in repositories:
        repo_df = centrality_df[centrality_df["repository"] == repo]

        # Compare each pair of measures
        for i, measure1 in enumerate(measures):
            for j, measure2 in enumerate(measures):
                if i >= j:  # Skip duplicate comparisons and self-comparisons
                    continue

                # Get top k nodes for each measure
                top_nodes1 = set(
                    repo_df[repo_df["measure"] == measure1]
                    .sort_values("centrality_score", ascending=False)
                    .head(top_k)["node_name"]
                )

                top_nodes2 = set(
                    repo_df[repo_df["measure"] == measure2]
                    .sort_values("centrality_score", ascending=False)
                    .head(top_k)["node_name"]
                )

                # Calculate Jaccard similarity
                similarity = calculate_jaccard_similarity(top_nodes1, top_nodes2)

                results.append(
                    {
                        "repository": repo,
                        "measure1": measure1,
                        "measure2": measure2,
                        "jaccard_similarity": similarity,
                        "common_nodes": len(top_nodes1.intersection(top_nodes2)),
                        "total_unique_nodes": len(top_nodes1.union(top_nodes2)),
                    }
                )

    return pd.DataFrame(results)


def plot_similarity_heatmap(similarity_df, output_dir):
    """
    Create a heatmap of Jaccard similarities between measures for each repository

    Args:
        similarity_df: DataFrame with similarity results
        output_dir: Directory to save the plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get unique repositories
    repositories = similarity_df["repository"].unique()

    # Get unique measures
    measures = sorted(
        set(similarity_df["measure1"].unique())
        | set(similarity_df["measure2"].unique())
    )

    # Create a figure for the average similarity across all repositories
    plt.figure(figsize=(10, 8))

    # Create a matrix for the average similarity
    avg_matrix = np.zeros((len(measures), len(measures)))

    # Fill the matrix with average similarities
    for i, measure1 in enumerate(measures):
        for j, measure2 in enumerate(measures):
            if i == j:
                avg_matrix[i, j] = 1.0  # Self-similarity is 1
            else:
                # Get the similarity between these measures
                mask1 = (similarity_df["measure1"] == measure1) & (
                    similarity_df["measure2"] == measure2
                )
                mask2 = (similarity_df["measure1"] == measure2) & (
                    similarity_df["measure2"] == measure1
                )

                similarities = similarity_df[mask1 | mask2]["jaccard_similarity"]
                if len(similarities) > 0:
                    avg_matrix[i, j] = similarities.mean()

    # Create the heatmap
    sns.heatmap(
        avg_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=measures,
        yticklabels=measures,
    )
    plt.title("Average Jaccard Similarity Between Centrality Measures")
    plt.tight_layout()
    plt.savefig(output_dir / "average_similarity.png")

    # Create individual heatmaps for each repository
    for repo in repositories:
        plt.figure(figsize=(10, 8))

        # Filter for this repository
        repo_df = similarity_df[similarity_df["repository"] == repo]

        # Create a matrix for this repository
        repo_matrix = np.zeros((len(measures), len(measures)))

        # Fill the matrix
        for i, measure1 in enumerate(measures):
            for j, measure2 in enumerate(measures):
                if i == j:
                    repo_matrix[i, j] = 1.0  # Self-similarity is 1
                else:
                    # Get the similarity between these measures
                    mask1 = (repo_df["measure1"] == measure1) & (
                        repo_df["measure2"] == measure2
                    )
                    mask2 = (repo_df["measure1"] == measure2) & (
                        repo_df["measure2"] == measure1
                    )

                    similarities = repo_df[mask1 | mask2]["jaccard_similarity"]
                    if len(similarities) > 0:
                        repo_matrix[i, j] = similarities.iloc[0]

        # Create the heatmap
        sns.heatmap(
            repo_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=measures,
            yticklabels=measures,
        )
        plt.title(f"Jaccard Similarity Between Centrality Measures for {repo}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{repo.replace('/', '_')}_similarity.png")
        plt.close()

    # Create a bar plot showing average similarity by repository
    plt.figure(figsize=(12, 6))
    repo_avg = (
        similarity_df.groupby("repository")["jaccard_similarity"].mean().reset_index()
    )
    repo_avg = repo_avg.sort_values("jaccard_similarity", ascending=False)

    sns.barplot(x="jaccard_similarity", y="repository", data=repo_avg)
    plt.title("Average Jaccard Similarity by Repository")
    plt.xlabel("Average Jaccard Similarity")
    plt.ylabel("Repository")
    plt.tight_layout()
    plt.savefig(output_dir / "repo_similarity_ranking.png")

    # Create a histogram of similarity values
    plt.figure(figsize=(10, 6))
    sns.histplot(similarity_df["jaccard_similarity"], bins=20, kde=True)
    plt.title("Distribution of Jaccard Similarity Values")
    plt.xlabel("Jaccard Similarity")
    plt.ylabel("Frequency")
    plt.axvline(
        similarity_df["jaccard_similarity"].mean(),
        color="red",
        linestyle="--",
        label=f'Mean: {similarity_df["jaccard_similarity"].mean():.2f}',
    )
    plt.axvline(
        similarity_df["jaccard_similarity"].median(),
        color="green",
        linestyle="--",
        label=f'Median: {similarity_df["jaccard_similarity"].median():.2f}',
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "similarity_distribution.png")


def plot_top_nodes_comparison(centrality_df, output_dir, top_k=10):
    """
    Create visualizations comparing the top nodes across different centrality measures

    Args:
        centrality_df: DataFrame with centrality results
        output_dir: Directory to save the plots
        top_k: Number of top nodes to consider
    """
    output_dir = Path(output_dir)
    repositories = centrality_df["repository"].unique()
    measures = centrality_df["measure"].unique()

    for repo in repositories:
        repo_df = centrality_df[centrality_df["repository"] == repo]

        # Create a plot showing the top nodes for each measure
        plt.figure(figsize=(14, 8))

        # Number of measures to compare
        n_measures = len(measures)

        # Create subplots for each measure
        for i, measure in enumerate(measures):
            measure_df = (
                repo_df[repo_df["measure"] == measure]
                .sort_values("centrality_score", ascending=False)
                .head(top_k)
            )

            plt.subplot(1, n_measures, i + 1)
            sns.barplot(x="centrality_score", y="node_name", data=measure_df)
            plt.title(f"{measure} Centrality")
            plt.xlabel("Centrality Score")
            plt.ylabel("Node Name")
            plt.tight_layout()

        plt.suptitle(f"Top {top_k} Nodes by Centrality Measure for {repo}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        plt.savefig(output_dir / f"{repo.replace('/', '_')}_top_nodes.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare centrality measures using Jaccard similarity"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="output/centrality_results.csv",
        help="Path to the centrality results file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/centrality_comparison",
        help="Directory to save the comparison results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top nodes to consider for each measure",
    )

    args = parser.parse_args()

    # Load centrality results
    print(f"Loading centrality results from {args.input_path}")
    centrality_df = pd.read_csv(args.input_path)

    # Compare centrality measures
    print(f"Comparing top {args.top_k} nodes for each measure")
    similarity_df = compare_centrality_measures(centrality_df, top_k=args.top_k)

    print("\nSummary Statistics:")
    print(
        f"Average Jaccard similarity: {similarity_df['jaccard_similarity'].mean():.4f}"
    )
    print(
        f"Median Jaccard similarity: {similarity_df['jaccard_similarity'].median():.4f}"
    )
    print(f"Min Jaccard similarity: {similarity_df['jaccard_similarity'].min():.4f}")
    print(f"Max Jaccard similarity: {similarity_df['jaccard_similarity'].max():.4f}")

    # Print repositories with highest and lowest similarity
    repo_avg = (
        similarity_df.groupby("repository")["jaccard_similarity"].mean().reset_index()
    )
    highest_repo = repo_avg.loc[repo_avg["jaccard_similarity"].idxmax()]
    lowest_repo = repo_avg.loc[repo_avg["jaccard_similarity"].idxmin()]

    print(
        f"\nRepository with highest average similarity: {highest_repo['repository']} ({highest_repo['jaccard_similarity']:.4f})"
    )
    print(
        f"Repository with lowest average similarity: {lowest_repo['repository']} ({lowest_repo['jaccard_similarity']:.4f})"
    )


if __name__ == "__main__":
    main()
