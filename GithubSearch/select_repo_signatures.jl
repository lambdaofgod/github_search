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
    
    # Write results incrementally to a CSV file
    if !isempty(results)
        # Change output path to CSV if it's a feather file
        output_path = args["output-path"]
        if endswith(output_path, ".feather")
            output_path = replace(output_path, ".feather" => ".csv")
        end
        
        println("\nSaving results to: ", output_path)
        
        # Create directory if it doesn't exist
        mkpath(dirname(output_path))
        
        # Initialize a counter for total rows
        total_rows = 0
        
        # Create a flag to track if this is the first write (to include headers)
        first_write = true
        
        # Process results in batches
        batch_size = 100  # Number of results to process at once
        for batch_start in 1:batch_size:length(results)
            # Get the current batch of results
            batch_end = min(batch_start + batch_size - 1, length(results))
            current_batch = results[batch_start:batch_end]
            
            # Create a DataFrame to hold this batch of data
            batch_df = DataFrame()
            
            # Process each result in the batch
            for result in current_batch
                if !isnothing(result) && nrow(result.centrality_df) > 0
                    # Make a copy of the centrality dataframe
                    df_copy = copy(result.centrality_df)
                    # Add repository and measure columns
                    df_copy.repository = fill(result.repo, nrow(df_copy))
                    df_copy.measure = fill(result.measure, nrow(df_copy))
                    # Append to the batch dataframe
                    if isempty(batch_df)
                        batch_df = df_copy
                    else
                        append!(batch_df, df_copy)
                    end
                    total_rows += nrow(df_copy)
                end
            end
            
            # Write the batch to CSV if it's not empty
            if !isempty(batch_df)
                CSV.write(
                    output_path, 
                    batch_df, 
                    append = !first_write, 
                    header = first_write
                )
                first_write = false
            end
        end
        
        if total_rows > 0
            println("Successfully wrote $total_rows rows to $output_path")
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
