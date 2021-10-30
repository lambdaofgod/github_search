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
from github_search.pytorch_geometric_data import PygGraphWrapper
from github_search import python_call_graph


try:
    torch.multiprocessing.set_start_method("spawn")
except:
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name, num_node_features, hidden_channels, num_layers):
    if model_name.startswith("graphsage"):

        if model_name == "graphsage_skip_connections":
            sage_layer = ResidualSAGEConv
        elif model_name == "graphsage":
            sage_layer = SAGEConv
        model = SAGE(
            num_node_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            sage_layer_cls=sage_layer,
        )
    elif model_name == "graph_infomax":
        model = Encoder()
        model = DeepGraphInfomax(
            hidden_channels=hidden_channels, encoder=Encoder(num_node_features, hidden_features),
            summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
            corruption=corruption).to(device)
    return model


def get_dataset_wrapper(embedder, label_file_nodes):
    dependency_records_df = pd.read_csv(
        "data/dependency_records.csv", encoding="latin-1"
    ).dropna()
    non_root_dependency_records_df = dependency_records_df[
        dependency_records_df["source"] != "<ROOT>"
    ]
    if label_file_nodes:
        non_root_dependency_records_df = (
            python_call_graph.get_records_df_with_labeled_files(
                non_root_dependency_records_df
            )
        )
    dependency_graph_wrapper = PygGraphWrapper(
        embedder.transform, non_root_dependency_records_df
    )
    return dependency_graph_wrapper


def train_unsupervised_gnn_model(
    data, model, hidden_channels, num_layers, epochs, batch_size, lr, in_jupyter=False
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # x, edge_index = data.x.to(device), data.edge_index.to(device)

    train_loader = SAGENeighborSampler(
        data.edge_index,
        sizes=[15, 5],
        batch_size=batch_size,
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


def run_gnn_experiment(
    product,
    fasttext_model_path,
    model_name,
    hidden_channels,
    num_layers,
    batch_size,
    label_file_nodes=False,
    epochs=50,
    lr=0.001,
):
    print("using model:", model_name)
    print("hidden channels:", hidden_channels)
    print("loading data")
    fasttext_model = fasttext.load_model(fasttext_model_path)
    fasttext_embedder = embeddings.FastTextVectorizer(fasttext_model)

    dependency_graph_wrapper = get_dataset_wrapper(fasttext_embedder, label_file_nodes)
    data = dependency_graph_wrapper.dataset
    model = get_model(model_name, data.num_node_features, hidden_channels, num_layers)
    train_unsupervised_gnn_model(
        data, model, hidden_channels, num_layers, epochs, batch_size, lr
    )
    torch.save(model, str(product["model_path"]))

    model = model.to("cpu")
    gnn_features = model.full_forward(data.x, data.edge_index).cpu().detach().numpy()
    gnn_kv = KeyedVectors(gnn_features.shape[1])
    gnn_kv.add(dependency_graph_wrapper.inverse_vertex_mapping.values, gnn_features)
    gnn_kv.save(str(product["gnn_token_embeddings"]))

    example_repo = "huggingface/transformers"

    def get_most_similar_repos(example_repo, n_repos):
        similarities = metrics.pairwise.cosine_distances(
            [gnn_features[dependency_graph_wrapper.vertex_mapping[example_repo]]],
            gnn_features,
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
