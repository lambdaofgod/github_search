using Graphs
using DataFrames
using Feather
using CSV
using ArgParse
include("src/GraphOps.jl")
using .GraphOps

function parse_commandline()
    s = ArgParseSettings()
    
    @add_arg_table s begin
        "--nodes-path"
            help = "Path to the nodes feather file"
            default = "../output/dependency_records/nodes.feather"
        "--edges-path"
            help = "Path to the edges feather file"
            default = "../output/dependency_records/edges.feather"
        "--output-path"
            help = "Path to save the centrality results"
            default = "output/centrality_results.feather"
        "--repos-file"
            help = "Path to CSV file containing repositories to analyze"
            default = "data/repos.csv"
    end
    
    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    # Load the nodes and edges data
    println("Loading data from files...")
    nodes_df = Feather.read(args["nodes-path"])
    edges_df = Feather.read(args["edges-path"])
    
    # Build the graph
    println("Building graph...")
    g = build_graph(nodes_df, edges_df)
    
    # Get selected nodes for analysis
    println("Identifying nodes for analysis...")
    selected_nodes = get_selected_nodes(nodes_df, edges_df)
    
    # Create data structure
    data = (
        nodes_df = nodes_df,
        edges_df = edges_df,
        graph = g,
        selected_nodes = selected_nodes
    )
    
    # Load repositories from CSV file
    println("Loading repositories from: ", args["repos-file"])
    repos_df = CSV.read(args["repos-file"], DataFrame)
    repos = Vector{String}(repos_df.repo)
    
    # Define centrality measures
    measures = [("degree", Graphs.degree_centrality), ("pagerank", Graphs.pagerank)]
    
    # Calculate centrality
    println("\nCalculating centrality for repositories:")
    results = calculate_centrality_for_repos(
        data.nodes_df, 
        data.edges_df, 
        repos, 
        measures, 
        selected_nodes=data.selected_nodes
    )
    
    # Combine results into a single DataFrame
    if !isempty(results)
        # Create a new DataFrame with all results
        all_centrality = DataFrame()
        
        for result in results
            if nrow(result.centrality_df) > 0
                # Make a copy of the centrality dataframe
                df_copy = result.centrality_df
                # Add repository and measure columns
                df_copy.repository = fill(result.repo, nrow(df_copy))
                df_copy.measure = fill(result.measure, nrow(df_copy))
                # Append to the combined dataframe
                if isempty(all_centrality)
                    all_centrality = df_copy
                else
                    append!(all_centrality, df_copy)
                end
            end
        end
        
        # Save results
        if !isempty(all_centrality)
            println("\nSaving results to: ", args["output-path"])
            Feather.write(args["output-path"], all_centrality)
            println("Done!")
        else
            println("No results to save.")
        end
    else
        println("No results to save.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
