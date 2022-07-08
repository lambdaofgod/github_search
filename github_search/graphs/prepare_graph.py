import pickle

import igraph
import pandas as pd


def make_file_vertices(df):
    file_vertices = df["repo_name"] + ":" + df["path"]
    return file_vertices


def make_function_vertices(df):
    function_vertices = df["repo_name"] + ":" + df["path"] + ":" + df["function_name"]
    return function_vertices


def get_vertex_info(df):
    repo_vertices = df["repo_name"].drop_duplicates().tolist()
    file_vertices = make_file_vertices(df).drop_duplicates().tolist()

    function_vertices = make_function_vertices(df)
    function_vertices_texts = df["function_code"]
    function_records_df = pd.DataFrame(
        {"vertex": function_vertices, "text": function_vertices_texts}
    ).drop_duplicates(subset="vertex")
    non_function_records_df = pd.DataFrame(
        {"vertex": repo_vertices + file_vertices, "text": repo_vertices + file_vertices}
    )
    records_df = pd.concat([function_records_df, non_function_records_df], axis=0)
    records_df = records_df.drop_duplicates(subset="vertex")
    return records_df["vertex"], records_df["text"], function_records_df["vertex"]


def get_edges_from_function_vertex(vname):
    split = vname.split(":")
    repo_name = split[0]
    file_name = ":".join(split[:2])
    return [(repo_name, file_name), (file_name, vname)]


def get_edges(vertices):
    return (
        edge
        for vname in vertices
        if len(vname.split(":")) == 3
        for edge in get_edges_from_function_vertex(vname)
    )


def prepare_from_function_code(upstream, product):
    function_code_df = pd.read_feather(upstream["prepare_function_code_df"])
    paperswithcode_df = pd.read_csv("data/paperswithcode_with_tasks.csv")
    repo_indicator = function_code_df["repo_name"].isin(paperswithcode_df["repo"])
    repo_function_counts = function_code_df["repo_name"].value_counts()
    selected_function_code_df = function_code_df[
        repo_indicator
        & ~function_code_df["repo_name"].isin(repo_function_counts.iloc[:10].index)
    ]
    vertices, vertex_texts, function_vertices = get_vertex_info(
        selected_function_code_df
    )
    edges = list(get_edges(function_vertices))
    graph = igraph.Graph()
    graph.add_vertices(vertices.tolist(), attributes={"text": vertex_texts.tolist()})
    graph.add_edges(edges)
    pickle.dump(graph, open(str(product), "wb"))
