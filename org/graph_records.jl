using Feather
using CSV
using DataFrames
using ProgressBars

dependency_records_path = "../output/dependency_records.feather"
df = Feather.read(dependency_records_path)
df = filter(row -> row[:edge_type] != "root-repo", df)


# Create a mapping of nodes to their repositories
node_to_repos = Dict{String, Set{String}}()

println("creating dictionary")
# Populate the dictionary from the dataframe
for i in ProgressBar(1:size(df)[1])
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

# Create a nodes dataframe directly from the node_to_repos dictionary
println("creating nodes dataframe")
nodes_df = DataFrame(index = Int[], name = String[], repo = String[])

# Fill the dataframe using the dictionary
println("filling nodes dataframe")
for (node_name, repos) in ProgressBar(node_to_repos)
    node_idx = nodes_dict[node_name]
    for repo in repos
        push!(nodes_df, (node_idx, node_name, repo))
    end
end

edges_df = DataFrame(index=df[!,:index], src=src_idxs, dst=dst_idxs, edge_type=df[!, :edge_type], repo=df[!, :repo])
CSV.write("../output/dependency_records/nodes.csv", nodes_df)

CSV.write("../output/dependency_records/edges.csv", edges_df)
names(df)
