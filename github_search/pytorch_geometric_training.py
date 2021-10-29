import os
import tqdm
import pandas as pd

from sklearn import metrics
import fasttext
from mlutil.feature_extraction import embeddings
import livelossplot
from gensim.models import KeyedVectors

import torch
from torch import optim

from github_search.pytorch_geometric_networks import *
from github_search import python_call_graph

try:
    torch.multiprocessing.set_start_method("spawn")
except:
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset(fasttext_model_path, label_file_nodes):
    dependency_records_df = pd.read_csv("data/dependency_records.csv", encoding='latin-1').dropna()
    non_root_dependency_records_df = dependency_records_df[
        dependency_records_df["source"] != "<ROOT>"
    ]
    if label_file_nodes:
        non_root_dependency_records_df = python_call_graph.get_records_df_with_labeled_files(non_root_dependency_records_df)
    fasttext_model = fasttext.load_model(fasttext_model_path)
    fasttext_embedder = embeddings.FastTextVectorizer(fasttext_model)
    dependency_graph_wrapper = PygGraphWrapper(
        fasttext_embedder.transform, non_root_dependency_records_df
    )
    return dependency_graph_wrapper


def train_unsupervised_graphsage_model(
    data, sage_layer, hidden_channels, num_layers, epochs, lr, in_jupyter=False
):
    model = SAGE(
        data.num_node_features, hidden_channels=hidden_channels, num_layers=num_layers, sage_layer_cls=sage_layer
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # x, edge_index = data.x.to(device), data.edge_index.to(device)

    train_loader = SAGENeighborSampler(
        data.edge_index,
        sizes=[15, 5],
        batch_size=2048,
        shuffle=True,
        num_nodes=data.num_nodes,
    )

    rnge = tqdm.tqdm(range(1, epochs + 1))

    plotlosses = livelossplot.PlotLosses()
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=10)

    for epoch in rnge:
        loss = train(model, data, train_loader, optimizer, device)
        scheduler.step()
        plotlosses.update(
            {"loss": loss, "lr": optimizer.state_dict()["param_groups"][0]["lr"]}
        )
        if in_jupyter:
            plotlosses.send()
        rnge.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}, ")
    plotlosses.send()
    return model


def run_graphsage_experiment(
    product,
    fasttext_model_path,
    skip_connections,
    hidden_channels,
    num_layers,
    label_file_nodes=True,
    use_x_in=True,
    epochs=50,
    lr=0.001,
):
    print("skip connections:", skip_connections)
    print("hidden channels:", hidden_channels)
    print("loading data")
    dependency_graph_wrapper = get_dataset(fasttext_model_path, label_file_nodes)
    data = dependency_graph_wrapper.dataset
    if skip_connections:
        sage_layer = ResidualSAGEConv
    else:
        sage_layer = SAGEConv

    model = train_unsupervised_graphsage_model(
        data, sage_layer, hidden_channels, num_layers, epochs, lr
    ).to(device)
    torch.save(model, str(product["model_path"]))

    model = model.to("cpu")
    graphsage_features = (
        model.full_forward(data.x, data.edge_index).cpu().detach().numpy()
    )
    graphsage_kv = KeyedVectors(graphsage_features.shape[1])
    graphsage_kv.add(
        dependency_graph_wrapper.inverse_vertex_mapping.values, graphsage_features
    )
    graphsage_kv.save(str(product["graphsage_token_embeddings"]))

    example_repo = "huggingface/transformers"

    def get_most_similar_repos(example_repo, n_repos):
        similarities = metrics.pairwise.cosine_distances(
            [graphsage_features[dependency_graph_wrapper.vertex_mapping[example_repo]]],
            graphsage_features,
        )[0]
        repo_inverse_vertex_mapping = dependency_graph_wrapper.inverse_vertex_mapping
        repo_inverse_vertex_mapping = repo_inverse_vertex_mapping[
            repo_inverse_vertex_mapping.str.contains("/")
        ]
        return dependency_graph_wrapper.inverse_vertex_mapping[
            (similarities[: len(repo_inverse_vertex_mapping)]).argsort()[:n_repos]
        ]

    print(get_most_similar_repos("open-mmlab/mmsegmentation", 20))
    print(get_most_similar_repos("pytorch/pytorch", 20))
    print(get_most_similar_repos("open-mmlab/mmsegmentation", 20))
