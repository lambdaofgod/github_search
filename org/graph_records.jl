using Feather
using CSV
using DataFrames
using ProgressBars

dependency_records_path = "../output/dependency_records.feather"
df = Feather.read(dependency_records_path)


# Create a dictionary to track node -> repo mappings
node_repo_map = Dict{String, String}()

# Populate the dictionary with source nodes and their repos
for i in 1:size(df)[1]
    node_repo_map[df[i, :source]] = df[i, :repo]
    node_repo_map[df[i, :destination]] = df[i, :repo]
end

# Get unique nodes
source_nodes = (df[!,:source] |> unique)
nodes = vcat(source_nodes, df[!,:destination] |> unique) |> unique

length(nodes)
nodes_dict = Dict([(nodes[i], i) for i in 1:length(nodes)])

# invert the direction
src_idxs = [nodes_dict[df[i,:source]] for i in tqdm(1:size(df)[1])]
dst_idxs = [nodes_dict[df[i,:destination]] for i in tqdm(1:size(df)[1])]

# Create nodes dataframe with index, name, and repo
nodes_df = DataFrame([(i, nodes[i], node_repo_map[nodes[i]]) for i in 1:length(nodes)])
nodes_df = rename!(nodes_df, [:index, :name, :repo])

edges_df = DataFrame(index=df[!,:index], src=src_idxs, dst=dst_idxs, edge_type=df[!, :edge_type], repo=df[!, :repo])
CSV.write("../output/dependency_records/nodes.csv", nodes_df)

CSV.write("../output/dependency_records/edges.csv", edges_df)
names(df)
