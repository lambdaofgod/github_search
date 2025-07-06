
# Example usage:
println("\nCalculating centrality for repositories:")
repos = ["ai4bharat-indicnlp/indicnlp_corpus", "000Justin000/torchdiffeq", "facebookresearch/online_dialog_eval", "0492wzl/tensorflow_slim_densenet", "huggingface/transformers"]
measures = [("degree", degree_centrality), ("pagerank", pagerank)]

# Call the function with selected nodes
cs = calculate_centrality_for_repos(nodes_df, edges_df, repos, measures, selected_nodes=selected_nodes)
