module GraphOps

using Graphs
using Feather
using DataFrames
using ProgressBars
using SortingAlgorithms
using Statistics

export load_graph_data, load_repo_subgraph, calculate_node_centrality, calculate_repo_centrality, calculate_centrality_for_repos

function load_graph_data(nodes_path, edges_path)
    # Load the nodes and edges data
    nodes_df = Feather.read(nodes_path)
    edges_df = Feather.read(edges_path)

    # Create a graph
    g = SimpleGraph(nrow(nodes_df))

    # Add edges to the graph
    for row in ProgressBar(eachrow(edges_df))
        # Add edge between source and destination
        # Note: Julia's Graphs.jl uses 1-based indexing
        add_edge!(g, row.src, row.dst)
    end

    # Basic graph information
    println("Number of nodes: ", nv(g))
    println("Number of edges: ", ne(g))
    
    # Get node names for repo-file edges - optimized with indexing
    selected_node_indices = filter(row -> row[:edge_type] in ["repo-file", "file-function"], edges_df)[!,:dst]
    # Create a lookup dictionary for faster access
    node_index_to_name = Dict(nodes_df.index .=> nodes_df.name)
    # Filter out indices that don't exist in the dictionary
    valid_indices = filter(idx -> haskey(node_index_to_name, idx), selected_node_indices)
    selected_nodes = [node_index_to_name[idx] for idx in valid_indices] |> unique
    
    return (
        nodes_df = nodes_df,
        edges_df = edges_df,
        graph = g,
        selected_nodes = selected_nodes
    )
end

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
    
    # Filter nodes_df to get only the nodes for this repository
    repo_nodes_df = filter(row -> row.repo == repo_name && row.index in repo_nodes, nodes_df)
    
    # Create a mapping from node index to node name
    node_index_to_name = Dict{Int, String}()
    for row in eachrow(repo_nodes_df)
        node_index_to_name[row.index] = row.name
    end
    
    # Get node names in the correct order
    node_names = [get(node_index_to_name, reverse_map[i], "Unknown") for i in 1:length(repo_nodes)]

    return (
        graph = subgraph,
        node_map = node_map,
        reverse_map = reverse_map,
        node_names = node_names,
        edges = repo_edges_df
    )
end

all_graph = (graph = g, node_names = nodes_df[!,:name], edges_df = edges_df)

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
    sort!(df, :centrality_score, rev=true, alg=QuickSort)
    
    return df
end


# Function to calculate centrality for a single repository with a specific measure
function calculate_repo_centrality(
    nodes_df::DataFrame,
    edges_df::DataFrame,
    repo::String,
    measure_name::String,
    centrality_measure::Function;
    selected_nodes::Union{Vector{String}, Nothing}=nothing,
    top_k::Int=10
)
    println("\nProcessing repository: ", repo)
    repo_subgraph = load_repo_subgraph(nodes_df, edges_df, repo)
    
    if isnothing(repo_subgraph)
        return nothing
    end
    
    centrality_df = calculate_node_centrality(repo_subgraph, centrality_measure)
    
    # Filter by selected nodes if provided
    if !isnothing(selected_nodes)
        centrality_df = filter(row -> row.node_name in selected_nodes, centrality_df)
    end
    
    if nrow(centrality_df) > 0
        println("Top central nodes:")
        display(first(centrality_df, top_k))
        println("Least central nodes:")
        display(last(centrality_df, top_k))
    else
        println("No nodes found with centrality scores.")
    end
    
    return (
        repo = repo,
        measure = measure_name,
        centrality_df = centrality_df
    )
end

# Function to calculate centrality for multiple repositories using different measures
function calculate_centrality_for_repos(
    nodes_df::DataFrame, 
    edges_df::DataFrame, 
    repos::Vector{String}, 
    measures::Vector{Tuple{String, Function}};
    selected_nodes::Union{Vector{String}, Nothing}=nothing,
    top_k::Int=10
)
    results = []
    
    for (measure_name, centrality_measure) in measures
        println("#############")
        println(measure_name)
        println("#############")
        
        for repo in repos
            result = calculate_repo_centrality(
                nodes_df, 
                edges_df, 
                repo, 
                measure_name, 
                centrality_measure;
                selected_nodes=selected_nodes,
                top_k=top_k
            )
            
            if !isnothing(result)
                push!(results, result)
            end
        end
    end
    
    return results
end

end # module

