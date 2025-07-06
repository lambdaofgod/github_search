using Feather
using CSV
using DataFrames
using ProgressBars

dependency_records_path = "../output/dependency_records.feather"
df = Feather.read(dependency_records_path)



# Create a mapping of nodes to their repositories
node_to_repos = Dict{String, Set{String}}()

# Populate the dictionary from the dataframe
for i in 1:size(df)[1]
    source = df[i, :source]
    dest = df[i, :destination]
    repo = df[i, :repo]
    
    # Add repo to source node's set
    if !haskey(node_to_repos, source)
        node_to_repos[source] = Set{String}()
    end
    push!(node_to_repos[source], repo)
    
    # Add repo to destination node's set
    if !haskey(node_to_repos, dest)
        node_to_repos[dest] = Set{String}()
    end
    push!(node_to_repos[dest], repo)
end

# Get unique nodes
source_nodes = (df[!,:source] |> unique)
nodes = vcat(source_nodes, df[!,:destination] |> unique) |> unique

length(nodes)
nodes_dict = Dict([(nodes[i], i) for i in 1:length(nodes)])

# invert the direction
src_idxs = [nodes_dict[df[i,:source]] for i in tqdm(1:size(df)[1])]
dst_idxs = [nodes_dict[df[i,:destination]] for i in tqdm(1:size(df)[1])]

# Create expanded nodes dataframe with one row per node-repo pair
expanded_nodes = []
for i in 1:length(nodes)
    node_name = nodes[i]
    for repo in node_to_repos[node_name]
        push!(expanded_nodes, (i, node_name, repo))
    end
end
nodes_df = DataFrame(expanded_nodes)
nodes_df = rename!(nodes_df, [:index, :name, :repo])

edges_df = DataFrame(index=df[!,:index], src=src_idxs, dst=dst_idxs, edge_type=df[!, :edge_type], repo=df[!, :repo])
CSV.write("../output/dependency_records/nodes.csv", nodes_df)

CSV.write("../output/dependency_records/edges.csv", edges_df)
names(df)
