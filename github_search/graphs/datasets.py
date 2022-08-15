import h5py
import torch
import torch_geometric.data as ptg_data


class HDF5Dataset(ptg_data.Dataset):
    def __init__(self, h5file, metadata_attrs, hierarchical_keys=True):
        self.h5file = h5file
        self.hierarchical_keys = hierarchical_keys
        self.keys = self._get_keys(h5file, hierarchical_keys)
        self.metadata_attrs = metadata_attrs
        self.transform = None
        self._validate_metadata()


    def len(self):
        return len(self.keys)

    def indices(self):
        return self.keys

    def get(self, idx):
        graph_record_data = self._get_graph_record_data(idx)
        return self._make_graph_record(graph_record_data)

    def _get_graph_record_data(self, key):
        graph_record_data = {
            "x": torch.Tensor(self.h5file["x"][key][()]),
            "edge_index": torch.LongTensor(self.h5file["edge_index"][key][()]),
            "graph_name": key
        }
        for meta_key in self.metadata_attrs:
            graph_record_data[meta_key] = self.h5file["x"][key].attrs[meta_key]
        return graph_record_data

    def _make_graph_record(self, data_dict):
        return ptg_data.Data(**data_dict)

    def _validate_metadata(self):
        key = self.keys[0]
        for meta_key in self.metadata_attrs:
            self.h5file["x"][key].attrs[meta_key]

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
