import gradio as gr
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
from github_search.python_call_graph_analysis import GraphCentralityAnalyzer
import io
from PIL import Image


def init_graphs():
    """Initialize graphs from dependency data on startup"""
    print("Loading dependency data...")
    graph_dependencies_df = pd.read_parquet("output/dependency_records.parquet")

    print("Calculating repo counts...")
    repo_counts = graph_dependencies_df["repo_name"].value_counts()

    print("Selecting repos with 300-305 dependencies...")
    selected_repo_counts = repo_counts[(repo_counts >= 300) & (repo_counts <= 305)]

    print(f"Loading {len(selected_repo_counts)} graphs...")
    selected_graphs = {}
    for repo_name in tqdm.tqdm(selected_repo_counts.index):
        analyzer = GraphCentralityAnalyzer(centrality_method="pagerank")
        graph = analyzer.load_graph_from_edge_df(repo_name, graph_dependencies_df)
        selected_graphs[repo_name] = graph

    print("Graphs loaded successfully!")
    return selected_graphs, selected_repo_counts


def visualize_graph(repo_name, graphs_dict):
    """Visualize the selected repository's graph"""
    if repo_name not in graphs_dict:
        return None, f"Repository '{repo_name}' not found in loaded graphs."
    
    if repo_name is None:
        return None, "Please select a repository."

    graph = graphs_dict[repo_name]

    # Create matplotlib figure
    plt.figure(figsize=(12, 8))

    # Use spring layout for better visualization
    pos = nx.spring_layout(graph, k=1, iterations=50)

    # Draw the graph
    nx.draw(
        graph,
        pos,
        node_size=50,
        node_color="lightblue",
        edge_color="gray",
        alpha=0.7,
        with_labels=False,
    )

    plt.title(f"Dependency Graph for {repo_name}")
    plt.tight_layout()

    # Save plot to bytes and convert to PIL Image
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
    img_buffer.seek(0)
    pil_image = Image.open(img_buffer)
    plt.close()

    # Return the image and stats
    stats = f"""
    Repository: {repo_name}
    Number of nodes: {graph.number_of_nodes()}
    Number of edges: {graph.number_of_edges()}
    """

    return pil_image, stats


def create_app():
    """Create and configure the Gradio app"""
    print("Initializing graphs...")
    graphs_dict, repo_counts = init_graphs()
    repo_names = list(graphs_dict.keys())

    def plot_selected_repo(repo_name):
        img_bytes, stats = visualize_graph(repo_name, graphs_dict)
        return img_bytes, stats

    # Create Gradio interface
    with gr.Blocks(title="Dependency Graph Visualization") as app:
        gr.Markdown("# Dependency Graph Visualization")
        gr.Markdown("Select a repository to visualize its dependency graph.")

        with gr.Row():
            with gr.Column(scale=1):
                repo_dropdown = gr.Dropdown(
                    choices=repo_names,
                    label="Select Repository",
                    value=repo_names[0] if repo_names else None,
                )

                visualize_btn = gr.Button("Visualize Graph", variant="primary")

                stats_text = gr.Textbox(
                    label="Graph Statistics", lines=4, interactive=False
                )

            with gr.Column(scale=2):
                graph_image = gr.Image(label="Dependency Graph", type="pil")

        # Set up event handlers
        visualize_btn.click(
            fn=plot_selected_repo,
            inputs=[repo_dropdown],
            outputs=[graph_image, stats_text],
        )

        # Auto-visualize on dropdown change
        repo_dropdown.change(
            fn=plot_selected_repo,
            inputs=[repo_dropdown],
            outputs=[graph_image, stats_text],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
