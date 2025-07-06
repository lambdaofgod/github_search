using Graphs
using DataFrames
using Feather
using ArgParse
include("src/GraphOps.jl")
using .GraphOps

function parse_commandline()
    s = ArgParseSettings()
    
    @add_arg_table s begin
        "--nodes-path"
            help = "Path to the nodes feather file"
            default = "output/dependency_records/nodes.feather"
        "--edges-path"
            help = "Path to the edges feather file"
            default = "output/dependency_records/edges.feather"
        "--output-path"
            help = "Path to save the centrality results"
            default = "output/centrality_results.feather"
        "--repos"
            help = "Comma-separated list of repositories to analyze"
            default = "ai4bharat-indicnlp/indicnlp_corpus,000Justin000/torchdiffeq,facebookresearch/online_dialog_eval,0492wzl/tensorflow_slim_densenet,huggingface/transformers"
    end
    
    return parse_args(s)
end

function main()
    args = parse_commandline()
    
    # Load graph data
    println("Loading graph data...")
    data = load_graph_data(args["nodes-path"], args["edges-path"])
    
    # Parse repositories
    repos = split(args["repos"], ",")
    
    # Define centrality measures
    measures = [("degree", degree_centrality), ("pagerank", pagerank)]
    
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
        all_centrality = vcat([result.centrality_df for result in results]...)
        
        # Add repository and measure columns
        all_centrality.repository = repeat([result.repo for result in results], inner=[nrow(result.centrality_df) for result in results])
        all_centrality.measure = repeat([result.measure for result in results], inner=[nrow(result.centrality_df) for result in results])
        
        # Save results
        println("\nSaving results to: ", args["output-path"])
        Feather.write(args["output-path"], all_centrality)
        
        println("Done!")
    else
        println("No results to save.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
