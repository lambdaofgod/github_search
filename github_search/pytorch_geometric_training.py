import os
import tqdm
import pandas as pd

from sklearn import metrics
import fasttext
from mlutil.feature_extraction import embeddings
import livelossplot
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import DeepGraphInfomax


from github_search.pytorch_geometric_networks import *
from github_search.pytorch_geometric_data import PygGraphWrapper
from github_search import python_call_graph
from torch_geometric.nn import DeepGraphInfomax


plt.ioff()

try:
    torch.multiprocessing.set_start_method("spawn")
except:
    pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(
    model_name, num_node_features, hidden_channels, num_layers, use_self_connection
):
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
            use_self_connection=use_self_connection,
        )
    elif model_name == "graph_infomax":
        model = DeepGraphInfomax(
            hidden_channels=hidden_channels,
            encoder=Encoder(num_node_features, hidden_channels, use_self_connection),
            summary=graph_infomax_summary,
            corruption=graph_infomax_corruption,
        )
    return model


def get_loader(model_name, data, batch_size, shuffle=True):
    if model_name.startswith("graphsage"):
        return SAGENeighborSampler(
            data.edge_index,
            sizes=[15, 5],
            batch_size=batch_size,
            shuffle=shuffle,
            num_nodes=data.num_nodes,
            num_workers=0,
        )
    elif model_name == "graph_infomax":
        return NeighborSampler(
            data.edge_index,
            node_idx=None,
            sizes=[10, 10, 25],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )


def get_dataset_wrapper(csv_paths, embedder, test_run, description_mode):
    dependency_records_df = pd.concat([pd.read_csv(p).dropna() for p in csv_paths])

    source_col = "source"
    destination_col = "destination"
    if description_mode:
        source_col = "repo_description"
        destination_col = "file_description"

    dependency_graph_wrapper = PygGraphWrapper(
        embedder, dependency_records_df, source_col, destination_col 
    )
    return dependency_graph_wrapper


def get_gnn_features(model_name, model, data):
    if model_name.startswith("graphsage"):
        gnn_features = (
            model.full_forward(data.x, data.edge_index).cpu().detach().numpy()
        )
    else:
        zs = []
        loader = get_loader(model_name, data, 2056, shuffle=False)
        for i, (batch_size, n_id, adjs) in enumerate(loader):
            adjs = [adj.to("cpu") for adj in adjs]
            x = data.x[n_id].cpu()
            zs.append(model(x, adjs)[0].cpu().detach())
        gnn_features = torch.cat(zs, dim=0).numpy()
    return gnn_features


def train_unsupervised_gnn_model(
    model,
    data,
    train_loader,
    hidden_channels,
    num_layers,
    epochs,
    batch_size,
    lr,
    plot_file,
    in_jupyter=False,
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # x, edge_index = data.x.to(device), data.edge_index.to(device)
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

    # TODO: refactor
    fig = plt.figure()
    x = list(litem.step for litem in plotlosses.logger.log_history["loss"])
    y = list(litem.value for litem in plotlosses.logger.log_history["loss"])
    plt.ion()
    plt.plot(x, y)
    plt.ioff()
    plt.savefig(plot_file)


def run_gnn_experiment(
    product,
    upstream,
    fasttext_model_path,
    model_name,
    hidden_channels,
    num_layers,
    batch_size,
    use_self_connection,
    epochs=50,
    lr=0.001,
    test_run=False,
    description_mode=True
):
    print()
    print("using model:", model_name)
    print("hidden channels:", hidden_channels)
    print("loading data")
    fasttext_model = fasttext.load_model(fasttext_model_path)
    fasttext_embedder = embeddings.FastTextVectorizer(fasttext_model)

    csv_paths = [upstream["postprocess_dependency_records"]]
    dependency_graph_wrapper = get_dataset_wrapper(
        csv_paths, fasttext_embedder, test_run, description_mode
    )
    data = dependency_graph_wrapper.dataset
    print("loaded:", data.num_nodes, "nodes")
    print("dataset:", data)

    train_loader = get_loader(model_name, data, batch_size)
    model = get_model(
        model_name,
        data.num_node_features,
        hidden_channels,
        num_layers,
        use_self_connection,
    )
    plot_file = str(product["plot_file"])
    train_unsupervised_gnn_model(
        model,
        data,
        train_loader,
        hidden_channels,
        num_layers,
        epochs,
        batch_size,
        lr,
        plot_file,
    )
    torch.save(model, str(product["model_path"]))

    # TODO: split this into different pipeline part
    model = model.to("cpu")
    model.training = False

    if description_mode:
        raw_dependency_graph_wrapper = get_dataset_wrapper(
            csv_paths, fasttext_embedder, test_run, False
        )
        raw_data = dependency_graph_wrapper.dataset
    else:
        raw_data = data

    gnn_features = get_gnn_features(model_name, model, raw_data)
    gnn_kv = KeyedVectors(gnn_features.shape[1])
    gnn_kv.add(raw_dependency_graph_wrapper.inverse_vertex_mapping.str.split(":").apply(lambda s: s[-1]).values, gnn_features)
    gnn_kv.save(str(product["gnn_token_embeddings"]))

    example_repo = "huggingface/transformers"

    def get_most_similar_repos(example_repo, n_repos):
        similarities = metrics.pairwise.cosine_distances(
            [gnn_features[raw_dependency_graph_wrapper.vertex_mapping[example_repo]]],
            gnn_features,
        )[0]
        repo_inverse_vertex_mapping = raw_dependency_graph_wrapper.inverse_vertex_mapping
        repo_inverse_vertex_mapping = repo_inverse_vertex_mapping[
            repo_inverse_vertex_mapping.str.contains("/")
        ]
        return raw_dependency_graph_wrapper.inverse_vertex_mapping[
            (similarities[: len(repo_inverse_vertex_mapping)]).argsort()[:n_repos]
        ]

    print(get_most_similar_repos("open-mmlab/mmsegmentation", 20))
    print(get_most_similar_repos("pytorch/pytorch", 20))
    print(get_most_similar_repos("huggingface/transformers", 20))
