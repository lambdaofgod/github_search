from operator import itemgetter
from typing import List

import h5py
import igraph
import pandas as pd
import sentence_transformers
import toolz
import torch
import torch_geometric
import torch_geometric.data
import torch_geometric.data as ptg_data
import tqdm
from fastai.text import all as text
from findkit.feature_extractor.fastai_feature_extractor import FastAITextFeatureExtractor

from github_search import utils
from github_search.graphs import data_preparation, datasets

dependency_df = pd.read_feather("../../output/dependency_records.feather")
dependency_df = dependency_df[dependency_df["source"] != "<ROOT>"]
paperswithcode_df = utils.load_paperswithcode_df(
    "../../data/paperswithcode_with_tasks.csv"
)


dependency_without_leaf_functions_df = dependency_df[
    dependency_df["edge_type"] != "function-function"
]

dependency_df.shape

dependency_without_leaf_functions_df.shape

repos = dependency_without_leaf_functions_df["repo"].drop_duplicates()


repo = repos.iloc[0]


def get_repo_dependencies(dependency_df, repo):
    repo_df = dependency_df[dependency_df["repo"] == repo]
    return repo_df


dependency_df[dependency_df["repo"] == repo]

repo_df = get_repo_dependencies(dependency_df, repos.iloc[10])


def make_repo_df_graph(repo_df):
    graph = igraph.Graph()
    vertices = pd.concat([repo_df["source"], repo_df["destination"]]).unique()
    graph.add_vertices(vertices, attributes={"text": vertices})
    graph.add_edges(repo_df[["source", "destination"]].to_records(index=False))
    return graph


repo_task_mapping = {
    repo: tasks
    for (repo, tasks) in tqdm.notebook.tqdm(
        paperswithcode_df[paperswithcode_df["repo"].isin(set(repos))][
            ["repo", "tasks"]
        ].to_records(index=False)
    )
}


repo_keyed_dependency_df = dependency_without_leaf_functions_df.groupby("repo")


def get_repo_df(dependency_gb, repo):
    return dependency_gb.get_group(repo)


repo = repos.iloc[10]


make_repo_df_graph(get_repo_df(repo_keyed_dependency_df, repo))

get_repo_df(repo_keyed_dependency_df, repo)

task_areas_df = pd.read_csv("../../data/paperswithcode_tasks.csv")


task_metadata_records_ = (
    paperswithcode_df[["repo", "least_common_task"]]
    .merge(task_areas_df, left_on="least_common_task", right_on="task")
    .drop_duplicates(subset="repo")
    .to_dict(orient="records")
)
task_metadata_records = {}
for rec in task_metadata_records_:
    repo = rec["repo"]
    task_metadata_records[repo] = {k: rec[k] for k in ["area", "least_common_task"]}

len([repo for repo in repo_task_mapping.keys() if repo in task_metadata_records.keys()])

repos_with_metadata = [
    repo for repo in repo_task_mapping.keys() if repo in task_metadata_records.keys()
]

r = repos_with_metadata[0]


fastai_learner = text.load_learner("learn.pkl")

fastai_extractor = FastAITextFeatureExtractor.build_from_learner(
    fastai_learner, max_length=48
)

texts = dependency_df.iloc[:100000]["source"]

texts

features = fastai_extractor.extract_features(texts.to_list())

features.shape

type(fastai_learner.model)

fastai_learner.dls.tfms


def make_graph_record(repo, encoding_fn):
    metadata = {"tasks": repo_task_mapping[repo], **task_metadata_records.get(repo)}
    return data_preparation.get_graph_data(
        make_repo_df_graph(get_repo_df(repo_keyed_dependency_df, repo)),
        repo,
        metadata,
        encoding_fn,
    )


graph_list = []


def get_graph_data_generator(repos, encoding_fn):
    pbar = tqdm.notebook.tqdm(repos)
    for repo in pbar:
        try:
            graph_data = make_graph_record(repo, encoding_fn)
            yield graph_data
        except RuntimeError:
            print(f"failed for repo {repo}")


repo_keyed_dependency_df.get_group("000Justin000/gnn-residual-correlation")

get_graph_data_generator(
    repos_with_metadata,
    toolz.partial(fastai_extractor.extract_features, show_progbar=False),
)


def add_graph_to_hdf_groups(
    g: torch_geometric.data.Data, x_group: h5py.Group, edge_index_group: h5py.Group
):
    if not g.graph_name in x_group.keys():
        d = x_group.create_dataset(g.graph_name, data=g.x.numpy())
        d.attrs["tasks"] = g.tasks
        d.attrs["least_common_task"] = g.least_common_task
        d.attrs["area"] = g.area
        edge_index_group.create_dataset(g.graph_name, data=g.edge_index.numpy())


def add_graph_data_list_to_file(
    h5py_file: h5py.File, graph_data_list: List[torch_geometric.data.Data]
):
    x_gp = h5py_file.create_group("x")
    edge_index_gp = h5py_file.create_group("edge_index")
    for g in tqdm.tqdm(graph_data_list):
        add_graph_to_hdf_groups(g, x_gp, edge_index_gp)


encoding_fn = toolz.partial(fastai_extractor.extract_features, show_progbar=False)
with h5py.File("graph_records.h5", "w") as f:
    add_graph_data_list_to_file(
        f, get_graph_data_generator(repos_with_metadata, encoding_fn)
    )


f = h5py.File("graph_records.h5", "r")

f.close()

gp = f["x"]

keys = [(key, subkey) for key in gp.keys() for subkey in gp[key].keys()]

k = "/".join(keys[0])

gp[k]

gp = f.get("x")["008karan"]

list(f["edge_index"]["008karan"]["SincNet_demo"])[0]

list(f["x"]["008karan"]["SincNet_demo"])


f = h5py.File("graph_records.h5", "r")


keys = datasets.HDF5Dataset._get_keys(f, True)

list(f["x"][keys[0]].attrs)

dset = datasets.HDF5Dataset(f, ["area", "tasks", "least_common_task"])

f["x"][keys[0]]

k = HDF5Dataset._get_keys(f, False)[0]

f["x"][k].keys()

ds = HDF5Dataset(f, "area")

ds[0]

ds = f["x"][ds.keys[0]]

ds.value

ds = HDF5Dataset(f, "area")

ds.get(5)

len(x_grp)

with h5py.File("/tmp/graph_data.h5", "w") as f:
    add_graph_data_list_to_file(f, graph_list)

data_preparation.get_graph_data(graph, repo, repo_task_mapping, rnn.encode)

graph.vs[2]

set(repo_df["source"])

repos.isin(paperswithcode_df["repo"]).mean()

paperswithcode_df

pd.concat(
    [file_dependencies, function_dependencies, function_function_dependencies]
).shape

function_function_dependencies

function_dependencies

function_function_dependencies
