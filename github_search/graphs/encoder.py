from torch_geometric.data import Data as GraphData
from dataclasses import dataclass
from github_search.graphs import models
import torch_geometric.data as ptg_data
from typing import List
import torch


@dataclass
class GraphEncoder:
    def __init__(self, gnn: models.GraphNeuralNetwork):
        self.gnn = gnn
        self.device = list(gnn.parameters())[0].device

    def encode_batch(
        self,
        inputs: List[GraphData],
        batch_size: int = 32,
        pool_graph: bool = True,
        **kwargs
    ) -> torch.Tensor:
        graph_batch = inputs.to(self.device)
        batch_embeddings = self.embedding_fn(pool_graph)(
            graph_batch.x, graph_batch.edge_index, graph_batch.batch
        )
        return batch_embeddings

    def encode(
        self,
        inputs: List[GraphData],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        return_tensors: bool = True,
        pool_graph: bool = True,
        **kwargs
    ) -> torch.Tensor:
        embeddings = []
        with torch.no_grad():
            for graph_batch in ptg_data.DataLoader(inputs, batch_size):
                graph_batch = graph_batch.to(self.device)
                batch_embeddings = self.embedding_fn(pool_graph)(
                    graph_batch.x, graph_batch.edge_index, graph_batch.batch
                )
                embeddings.append(batch_embeddings)
        return torch.row_stack(embeddings)

    def embedding_fn(self, pool_graph):
        if pool_graph:
            return self.gnn.forward
        else:
            return self.gnn.forward_without_pooling
