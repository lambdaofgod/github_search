import h5py
import torch
import torch_geometric.data as ptg_data


class HDF5Dataset(ptg_data.Dataset):
    def __init__(self, h5file, y_attr, hierarchical_keys=True):
        self.h5file = h5file
        self.hierarchical_keys = hierarchical_keys
        self.keys = self._get_keys(h5file, hierarchical_keys)
        self.y_attr = y_attr
        self.transform = None

    def len(self):
        return len(self.keys)

    def indices(self):
        return self.keys

    def get(self, idx):
        graph_record_data = self._get_graph_record_data(idx)
        return self._make_graph_record(graph_record_data)

    def _get_graph_record_data(self, key):
        return {
            "x": torch.Tensor(self.h5file["x"][key][()]),
            "edge_index": torch.LongTensor(self.h5file["edge_index"][key][()]),
            "y": self.h5file["x"][key].attrs[self.y_attr],
        }

    def _make_graph_record(self, data_dict):
        return ptg_data.Data(**data_dict)

    @classmethod
    def _get_keys(cls, h5file, hierarchical_keys):
        if hierarchical_keys:
            return [
                "/".join([key, subkey])
                for key in h5file["x"].keys()
                for subkey in h5file["x"][key].keys()
            ]
        else:
            return list(h5file["x"].keys())
