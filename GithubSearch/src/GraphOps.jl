using Graphs
using CSV
using DataFrames
using ProgressBars
using SortingAlgorithms

# Load the nodes and edges data
nodes_df = CSV.read("../output/dependency_records/nodes.csv", DataFrame)
edges_df = CSV.read("../output/dependency_records/edges.csv", DataFrame)

# Create a graph
g = SimpleGraph(nrow(nodes_df))

# Add edges to the graph
for row in eachrow(edges_df)
    # Add edge between source and destination
    # Note: Julia's Graphs.jl uses 1-based indexing
    add_edge!(g, row.src, row.dst)
end

# Basic graph information
println("Number of nodes: ", nv(g))
println("Number of edges: ", ne(g))

# Function to create a subgraph for a specific repository
function load_repo_subgraph(nodes_df, edges_df, repo_name)
    # Filter edges for the specified repository
    repo_edges_df = filter(row -> row.repo == repo_name, edges_df)

    if nrow(repo_edges_df) == 0
        println("No edges found for repository: ", repo_name)
        return nothing
    end

    # Get unique node indices involved in this repo
    repo_nodes = unique(vcat(repo_edges_df.src, repo_edges_df.dst))

    # Create a mapping from original indices to new consecutive indices
    node_map = Dict(node => i for (i, node) in enumerate(repo_nodes))

    # Create a new graph with the number of nodes in this repo
    subgraph = SimpleGraph(length(repo_nodes))

    # Add edges to the subgraph with remapped indices
    for row in eachrow(repo_edges_df)
        new_src = node_map[row.src]
        new_dst = node_map[row.dst]
        add_edge!(subgraph, new_src, new_dst)
    end

    # Create a reverse mapping to get original node names
    reverse_map = Dict(i => node for (node, i) in node_map)
    node_names = [nodes_df[reverse_map[i], :name] for i in 1:length(repo_nodes)]

    return (
        graph = subgraph,
        node_map = node_map,
        reverse_map = reverse_map,
        node_names = node_names,
        edges = repo_edges_df
    )
end

# Example usage:
repo_name = "000Justin000/torchdiffeq"
repo_subgraph = load_repo_subgraph(nodes_df, edges_df, repo_name)

if !isnothing(repo_subgraph)
    println("\nSubgraph for repository: ", repo_name)
    println("Number of nodes: ", nv(repo_subgraph.graph))
    println("Number of edges: ", ne(repo_subgraph.graph))
end

# Function to calculate centrality scores for nodes in a subgraph
function calculate_node_centrality(subgraph_data, centrality_function)
    # Extract the graph from the subgraph data
    g = subgraph_data.graph
    
    # Calculate centrality scores using the provided function
    centrality_scores = centrality_function(g)
    
    # Create a DataFrame with node names and their centrality scores
    df = DataFrame(
        node_index = 1:length(centrality_scores),
        node_name = subgraph_data.node_names,
        centrality_score = centrality_scores
    )
    
    # Sort the DataFrame by centrality score in descending order
    sort!(df, :centrality_score, rev=true)
    
    return df
end

# Example usage:
# using Graphs.Centrality
# centrality_df = calculate_node_centrality(repo_subgraph, closeness_centrality)
