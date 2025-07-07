import gradio as gr
import pandas as pd
import networkx as nx
import tqdm
from github_search.python_call_graph_analysis import GraphCentralityAnalyzer
import plotly.graph_objects as go
import plotly.express as px


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


def create_interactive_plotly_graph(repo_name, graph, layout_type="spring"):
    """Create an interactive Plotly graph with node names and edge types"""
    # Get node positions using selected layout
    if layout_type == "spring":
        pos = nx.spring_layout(graph, k=1, iterations=50)
    elif layout_type == "circular":
        pos = nx.circular_layout(graph)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    elif layout_type == "fruchterman_reingold":
        pos = nx.fruchterman_reingold_layout(graph, k=1, iterations=50)
    elif layout_type == "shell":
        pos = nx.shell_layout(graph)
    elif layout_type == "spectral":
        pos = nx.spectral_layout(graph)
    elif layout_type == "planar":
        try:
            pos = nx.planar_layout(graph)
        except nx.NetworkXException:
            # Fallback to spring layout if graph is not planar
            pos = nx.spring_layout(graph, k=1, iterations=50)
    else:
        pos = nx.spring_layout(graph, k=1, iterations=50)
    
    # Extract edges with their data
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Extract edge type from edge data
        edge_type = edge[2].get('edge_type', 'unknown')
        edge_info.append(f"{edge[0]} → {edge[1]}<br>Type: {edge_type}")
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Edges'
    )
    
    # Extract node information
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    node_colors = []
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Truncate long node names for display
        display_name = str(node)
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."
        
        node_text.append(display_name)
        node_info.append(f"Node: {node}<br>Degree: {graph.degree(node)}")
        
        # Color nodes by type (simple heuristic based on node name)
        if '/' in str(node):  # Repository names
            node_colors.append('red')
        elif '.py' in str(node):  # File names
            node_colors.append('lightblue')
        else:  # Functions/classes
            node_colors.append('lightgreen')
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_info,
        text=node_text,
        textposition="middle center",
        textfont=dict(size=8),
        marker=dict(
            size=12,
            color=node_colors,
            line=dict(width=1, color='black'),
            opacity=0.8
        ),
        name='Nodes'
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title=dict(
            text=f'Interactive Dependency Graph: {repo_name}',
            font=dict(size=16)
        ),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Hover over nodes for details. Zoom and pan to explore.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig


def visualize_graph(repo_name, graphs_dict, layout_type="spring"):
    """Visualize the selected repository's graph"""
    if repo_name not in graphs_dict:
        return None, f"Repository '{repo_name}' not found in loaded graphs."
    
    if repo_name is None:
        return None, "Please select a repository."

    graph = graphs_dict[repo_name]
    
    # Create interactive Plotly graph
    fig = create_interactive_plotly_graph(repo_name, graph, layout_type)
    
    # Generate statistics
    edge_types = {}
    for _, _, data in graph.edges(data=True):
        edge_type = data.get('edge_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    edge_type_summary = "\n".join([f"  {edge_type}: {count}" for edge_type, count in edge_types.items()])
    
    stats = f"""Repository: {repo_name}
Number of nodes: {graph.number_of_nodes()}
Number of edges: {graph.number_of_edges()}

Edge types:
{edge_type_summary}
"""

    return fig, stats


def create_app():
    """Create and configure the Gradio app"""
    print("Initializing graphs...")
    graphs_dict, repo_counts = init_graphs()
    repo_names = list(graphs_dict.keys())

    def plot_selected_repo(repo_name, layout_type):
        fig, stats = visualize_graph(repo_name, graphs_dict, layout_type)
        return fig, stats

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

                layout_dropdown = gr.Dropdown(
                    choices=[
                        ("Spring Layout (Force-directed)", "spring"),
                        ("Circular Layout", "circular"),
                        ("Kamada-Kawai Layout", "kamada_kawai"),
                        ("Fruchterman-Reingold Layout", "fruchterman_reingold"),
                        ("Shell Layout", "shell"),
                        ("Spectral Layout", "spectral"),
                        ("Planar Layout", "planar")
                    ],
                    label="Select Layout",
                    value="spring"
                )

                visualize_btn = gr.Button("Visualize Graph", variant="primary")

                stats_text = gr.Textbox(
                    label="Graph Statistics", lines=4, interactive=False
                )

            with gr.Column(scale=2):
                graph_plot = gr.Plot(label="Interactive Dependency Graph")

        # Set up event handlers
        visualize_btn.click(
            fn=plot_selected_repo,
            inputs=[repo_dropdown, layout_dropdown],
            outputs=[graph_plot, stats_text],
        )

        # Auto-visualize on dropdown change
        repo_dropdown.change(
            fn=plot_selected_repo,
            inputs=[repo_dropdown, layout_dropdown],
            outputs=[graph_plot, stats_text],
        )

        # Auto-visualize on layout change
        layout_dropdown.change(
            fn=plot_selected_repo,
            inputs=[repo_dropdown, layout_dropdown],
            outputs=[graph_plot, stats_text],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
