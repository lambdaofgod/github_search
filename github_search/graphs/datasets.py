import os
import pickle
from typing import Iterable, List, Union

import h5py
import torch
import torch_geometric
import torch_geometric.data as ptg_data
import tqdm

GraphDataList = List[ptg_data.Data]
GraphDatasetLike = Union[GraphDataList, ptg_data.Dataset]


def add_graph_to_hdf_groups(
    g: torch_geometric.data.Data, x_group: h5py.Group, edge_index_group: h5py.Group
):
    if not g.graph_name in x_group.keys():
        d = x_group.create_dataset(g.graph_name, data=g.x.numpy())
        d.attrs["tasks"] = g.tasks
        d.attrs["least_common_task"] = g.least_common_task
        d.attrs["area"] = g.area
        edge_index_group.create_dataset(g.graph_name, data=g.edge_index.numpy())


def write_graph_data_iter_to_h5_file(
    h5py_file: h5py.File, graph_data_list: Iterable[torch_geometric.data.Data]
):
    x_gp = h5py_file.create_group("x")
    edge_index_gp = h5py_file.create_group("edge_index")
    for g in tqdm.auto.tqdm(graph_data_list):
        add_graph_to_hdf_groups(g, x_gp, edge_index_gp)


def load_dataset(file_path: str, metadata_attrs: List[str]) -> GraphDatasetLike:
    ext = os.path.splitext(file_path)[1]
    if ext == ".pkl":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    elif ext == ".h5":
        f = h5py.File(file_path, "r")
        return HDF5Dataset.from_file(f, metadata_attrs=metadata_attrs, hierarchical_keys=True)


def filter_dataset(dataset: GraphDatasetLike, keys: List[str]):
    if type(dataset) is GraphDataList:
        return [data for data in dataset if data.graph_name in keys]
    elif type(dataset) is HDF5Dataset:
        return dataset.get_subset_by_keys(keys)



class HDF5Dataset(ptg_data.Dataset):
    def __init__(
        self,
        x,
        edge_index,
        metadata_attrs: List[str],
        hierarchical_keys: bool = True,
        used_keys: List[str] = None,
    ):
        self.x = x
        self.edge_index = edge_index
        self.hierarchical_keys = hierarchical_keys
        if used_keys is None:
            self.keys = self._get_keys(x, hierarchical_keys)
        else:
            self.keys = used_keys
        self.metadata_attrs = metadata_attrs
        self.transform = None
        self._validate_metadata()

    @classmethod
    def from_file(cls, h5file, metadata_attrs, hierarchical_keys):
        x = h5file["x"]
        edge_index = h5file["edge_index"]
        return cls(x, edge_index, metadata_attrs, hierarchical_keys)

    def len(self):
        return len(self.keys)

    def indices(self):
        return self.keys

    def get(self, idx):
        graph_record_data = self._get_graph_record_data(idx)
        return self._make_graph_record(graph_record_data)

    def _get_graph_record_data(self, key):
        graph_record_data = {
            "x": torch.Tensor(self.x[key][()]),
            "edge_index": torch.LongTensor(self.edge_index[key][()]),
            "graph_name": key,
        }
        for meta_key in self.metadata_attrs:
            graph_record_data[meta_key] = self.x[key].attrs[meta_key]
        return graph_record_data

    def _make_graph_record(self, data_dict):
        return ptg_data.Data(**data_dict)

    def _validate_metadata(self):
        key = self.keys[0]
        for meta_key in self.metadata_attrs:
            self.x[key].attrs[meta_key]

    @classmethod
    def _get_keys(cls, x, hierarchical_keys):
        if hierarchical_keys:
            return [
                "/".join([key, subkey]) for key in x.keys() for subkey in x[key].keys()
            ]
        else:
            return list(x.keys())

    def get_subset_by_keys(self, keys):
        return HDF5Dataset(
            self.x,
            self.edge_index,
            self.metadata_attrs,
            self.hierarchical_keys,
            used_keys=keys,
        )
