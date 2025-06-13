using Feather
using CSV
using DataFrames
using ProgressBars

dependency_records_path = "../output/dependency_records.feather"
df = Feather.read(dependency_records_path)


source_nodes = (df[!,:source] |> unique)
nodes = vcat(source_nodes, df[!,:destination] |> unique) |> unique

length(nodes)
nodes_dict = Dict([(nodes[i], i) for i in 1:length(nodes)])

# invert the direction
src_idxs = [nodes_dict[df[i,:source]] for i in tqdm(1:size(df)[1])]
dst_idxs = [nodes_dict[df[i,:destination]] for i in tqdm(1:size(df)[1])]

# Create nodes dataframe with index and name
nodes_df = DataFrame([(i, nodes[i]) for i in 1:length(nodes)])
nodes_df = rename!(nodes_df, [:index, :name])

# Get repo information for each node by joining with original dataframe
# Create mappings of node names to their repos from both source and destination columns
source_repos = select(df, [:source, :repo]) |> unique
source_repos = rename!(source_repos, :source => :name)

dest_repos = select(df, [:destination, :repo]) |> unique
dest_repos = rename!(dest_repos, :destination => :name)

# Combine both source and destination mappings
node_repos = vcat(source_repos, dest_repos) |> unique

# Add repo information to nodes_df by joining
nodes_df = leftjoin(nodes_df, node_repos, on = :name)

edges_df = DataFrame(index=df[!,:index], src=src_idxs, dst=dst_idxs, edge_type=df[!, :edge_type], repo=df[!, :repo])
CSV.write("../output/dependency_records/nodes.csv", nodes_df)

CSV.write("../output/dependency_records/edges.csv", edges_df)
names(df)
