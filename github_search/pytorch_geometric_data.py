import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


class PygGraphWrapper:
    """
    holds pytorch_geometric dataset and utils for mapping vertices to names
    """

    def __init__(
        self, featurizer, records_df, source_col="source", destination_col="destination"
    ):
        self.featurizer = featurizer
        self.records_df = records_df
        self.source_col = source_col
        self.destination_col = destination_col

        vertices = (
            pd.concat([records_df[source_col], records_df[destination_col]])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        self.vertex_mapping = pd.Series(data=vertices.index, index=vertices.values)
        self.inverse_vertex_mapping = vertices
        edge_index_source = self.vertex_mapping.loc[records_df[source_col]].values
        edge_index_destination = self.vertex_mapping.loc[
            records_df[destination_col]
        ].values
        edge_index = torch.tensor(
            np.row_stack([edge_index_source, edge_index_destination])
        )
        features = featurizer.transform(self.vertex_mapping.index)
        self.dataset = Data(torch.tensor(features), torch.tensor(edge_index))

    def get_sub_dataset_wrapper(self, vertex_subset):
        records_subdf = self.records_df[
            self.records_df[self.source_col].isin(vertex_subset)
            | self.records_df[self.destination_col].isin(vertex_subset)
        ]
        return PygGraphWrapper(self.featurizer, records_subdf)

    def get_vertex_embeddings(self, vertex_subset, model):
        sub_dataset_wrapper = self.get_sub_dataset_wrapper(vertex_subset)
        features = (
            model.full_forward(
                sub_dataset_wrapper.dataset.x, sub_dataset_wrapper.dataset.edge_index
            )
            .cpu()
            .detach()
            .numpy()
        )
        return features[sub_dataset_wrapper.vertex_mapping.loc[vertex_subset]]
