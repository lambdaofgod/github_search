import math
from dataclasses import dataclass
from typing import List, Optional

import livelossplot
import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
from gensim.models import KeyedVectors
from sklearn import metrics
from torch import optim
from torch_geometric import transforms
from torch_geometric.loader import LinkNeighborLoader, NeighborSampler
from torch_geometric.nn import DeepGraphInfomax, GraphSAGE, norm

from github_search import neptune_util
from github_search.pytorch_geometric_data import PygGraphWrapper
from github_search.pytorch_geometric_networks import train_epoch

plt.ioff()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GNNTrainingArgs:
    model_name: str
    num_neighbors: List[int]
    layers: int
    batch_size: int
    hidden_channels: int
    epochs: int
    lr: float
    weight_decay: float


@dataclass
class CheckpointingArgs:
    model_path: Optional[str]
    n_save_steps: Optional[int]


def get_model(model_name, num_node_features, hidden_channels, layers):
    if model_name.startswith("graphsage"):
        model = GraphSAGE(
            num_node_features,
            hidden_channels=hidden_channels,
            num_layers=layers,
            dropout=0.2,
            norm=norm.GraphSizeNorm(),
        )
    elif model_name == "graph_infomax":
        model = DeepGraphInfomax(
            hidden_channels=hidden_channels,
            encoder=Encoder(num_node_features, hidden_channels, use_self_connection),
            summary=graph_infomax_summary,
            corruption=graph_infomax_corruption,
        )
    return model


def get_loader(model_name, data, num_neighbors, batch_size, shuffle=True):
    if model_name.startswith("graphsage"):
        return LinkNeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            neg_sampling_ratio=25.0,
            shuffle=shuffle,
            num_workers=0,
            transform=transforms.NormalizeFeatures(),
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
            model(data.x, data.edge_index).cpu().detach().numpy()
        )
    else:
        zs = []
        loader = get_loader(model_name, data, num_neighbors, 256, shuffle=False)
        for i, (batch_size, n_id, adjs) in enumerate(loader):
            adjs = [adj.to("cpu") for adj in adjs]
            x = data.x[n_id].cpu()
            zs.append(model(x, adjs)[0].cpu().detach())
        gnn_features = torch.cat(zs, dim=0).numpy()
    return gnn_features


def get_gnn_kv(model_name, vertex_names, model, data):
    gnn_features = get_gnn_features(model_name, model, data)
    gnn_kv = KeyedVectors(gnn_features.shape[1])
    gnn_kv.add(
        vertex_names,
        gnn_features,
    )
    return gnn_kv


def trivial_callback(model, step):
    return None


def get_model_callback(model_path: str, n_save_steps: int):
    def save_model_checkpoint(model, step):
        if step % n_save_steps == 0:
            ith_model_path = model_path.replace(".pth", f"_step{step}.pth")
            torch.save(model, ith_model_path)

    if model_path is None or n_save_steps is None:
        return trivial_callback
    else:
        return save_model_checkpoint


def train_unsupervised_gnn_model(
    model,
    data,
    train_loader,
    training_args: GNNTrainingArgs,
    checkpointing_args: CheckpointingArgs,
    plot_file,
    in_jupyter=False,
):
    model = model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_args.lr,
        weight_decay=training_args.weight_decay,
    )
    # x, edge_index = data.x.to(device), data.edge_index.to(device)
    rnge = tqdm.tqdm(range(1, training_args.epochs + 1))

    plotlosses = livelossplot.PlotLosses()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=math.ceil(len(data.x) / training_args.batch_size)
    )

    grad_param_names = [
        "convs.0.lin_l.weight",
        "convs.0.lin_r.weight",
        "convs.1.lin_l.bias",
        "convs.1.lin_l.weight",
        "convs.0.lin_l.bias",
        "convs.1.lin_r.weight",
    ]

    callback = neptune_util.get_neptune_callback(
        tags=["unsupervised_graphsage"],
        param_names=["loss", "size", "lr"] + grad_param_names,
    )
    model_callback = get_model_callback(
        checkpointing_args.model_path, checkpointing_args.n_save_steps
    )
    for epoch in rnge:
        loss = train_epoch(
            model,
            data,
            train_loader,
            optimizer,
            scheduler,
            device,
            callback,
            model_callback,
        )
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
    params,
    n_save_steps,
    lr=0.001,
    test_run=False,
    description_mode=False,
):
    training_args = GNNTrainingArgs(**params)
    checkpointing_args = CheckpointingArgs(
        n_save_steps=n_save_steps, model_path=str(product["model_path"])
    )
    try:
        torch.multiprocessing.set_start_method("spawn")
    except:
        pass

    print()
    print("using model:", training_args.model_name)
    print("hidden channels:", training_args.hidden_channels)
    print("loading data")

    data = torch.load(str(upstream["gnn.make_graphsage_data"]))
    print("loaded:", data.num_nodes, "nodes")
    print("dataset:", data)

    train_loader = get_loader(
        training_args.model_name,
        data,
        training_args.num_neighbors,
        training_args.batch_size,
    )
    model = get_model(
        training_args.model_name,
        data.num_node_features,
        training_args.hidden_channels,
        training_args.layers,
    )
    plot_file = str(product["plot_file"])
    train_unsupervised_gnn_model(
        model,
        data,
        train_loader,
        training_args,
        checkpointing_args,
        plot_file,
    )
    torch.save(model, str(product["model_path"]))

    # TODO: split this into different pipeline part
    model = model.to("cpu")
    model.training = False

    if description_mode:
        dependency_graph_wrapper = get_dataset_wrapper(
            csv_paths, fasttext_embedder, test_run, False
        )
        raw_data = dependency_graph_wrapper.dataset
    else:
        raw_data = data

    vertex_names = (
        dependency_graph_wrapper.inverse_vertex_mapping.str.split(":")
        .apply(lambda s: s[-1])
        .values
    )

    gnn_kv = get_gnn_kv(training_args.model_name, vertex_names, model, raw_data)
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
    print(get_most_similar_repos("huggingface/transformers", 20))
