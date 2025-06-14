using Graphs
using Feather
using DataFrames
using ProgressBars
using SortingAlgorithms

# Load the nodes and edges data
nodes_df = Feather.read("../output/dependency_records/nodes.feather")
edges_df = Feather.read("../output/dependency_records/edges.feather")

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

# Get node names for repo-file edges - optimized with indexing
selected_node_indices = filter(row -> row[:edge_type] == "repo-file", edges_df)[!,:dst]
# Create a lookup dictionary for faster access
node_index_to_name = Dict(nodes_df.index .=> nodes_df.name)
selected_nodes = [node_index_to_name[idx] for idx in selected_node_indices] |> unique

# Example usage:
# using Graphs.Centrality
println("\nCalculating centrality for top 10 repositories:")
repos = ["ai4bharat-indicnlp/indicnlp_corpus", "000Justin000/torchdiffeq", "facebookresearch/online_dialog_eval"] #nodes_df[!,:repo] |> unique
cs = []
for (measure_name, centrality_measure) in zip(["degree", "pagerank"], [degree_centrality, pagerank])
    println("#############")
    println(measure_name)
    println("#############")
    for repo in repos#[1:10]
        println("\nProcessing repository: ", repo)
        repo_subgraph = load_repo_subgraph(nodes_df, edges_df, repo)
        if !isnothing(repo_subgraph)
            centrality_df = calculate_node_centrality(repo_subgraph, centrality_measure)
            # Filter centrality dataframe to only include selected nodes
            centrality_df = filter(row -> row.node_name in selected_nodes, centrality_df)
            if nrow(centrality_df) > 0
                println("Top central nodes:")
                display(first(centrality_df, 10))
                println("Least central nodes:")
                display(last(centrality_df, 10))
            else
                println("No nodes found with centrality scores.")
            end
            push!(cs, centrality_df)
        end
    end
end

# Print repository statistics
println("\n\n=== Repository Statistics ===")
println("Calculating statistics for all repositories...")

# Group edges by repository
repo_stats = Dict{String, Tuple{Int, Int}}()
repo_groups = groupby(edges_df, :repo)

for (repo_name, repo_edges) in pairs(repo_groups)
    # Get unique nodes in this repository
    repo_nodes = unique(vcat(repo_edges.src, repo_edges.dst))
    # Store (node count, edge count) for each repository
    repo_stats[repo_name.repo] = (length(repo_nodes), nrow(repo_edges))
end

# Create a DataFrame for better display
stats_df = DataFrame(
    repository = String[],
    node_count = Int[],
    edge_count = Int[]
)

# Fill the DataFrame with repository statistics
for (repo, (nodes, edges)) in sort(collect(repo_stats), by=x->x[2][1], rev=true)
    push!(stats_df, (repo, nodes, edges))
end

# Display top repositories by node count
println("\nTop 20 repositories by node count:")
display(first(stats_df, 20))

# Calculate and display overall statistics
total_repos = length(repo_stats)
total_nodes = sum(x -> x[1], values(repo_stats))
total_edges = sum(x -> x[2], values(repo_stats))
avg_nodes_per_repo = total_nodes / total_repos
avg_edges_per_repo = total_edges / total_repos

println("\nOverall Statistics:")
println("Total repositories: ", total_repos)
println("Total nodes across all repositories: ", length(unique(vcat(edges_df.src, edges_df.dst))))
println("Total edges across all repositories: ", nrow(edges_df))
println("Average nodes per repository: ", round(avg_nodes_per_repo, digits=2))
println("Average edges per repository: ", round(avg_edges_per_repo, digits=2))
