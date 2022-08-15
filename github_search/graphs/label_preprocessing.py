import itertools
from collections import defaultdict
from typing import List

import findkit
import torch
import torch_geometric.data as ptg_data


def make_graph_from_label_list(
    label_lists: List[List[str]], feature_extractor: findkit.feature_extractor.FeatureExtractor
):
    """
    prepare graphs from list of list of strings
    each text from label list is encoded using `feature_extractor`
    each example list is then converted to a graph
    """

    texts, indices = _get_flattened_list_with_indices(label_lists)
    indices_groups = _groupby_index(indices)
    graph_data = ptg_data.Data(
        torch.Tensor(
            feature_extractor.extract_features(texts, show_progress_bar=False)
        ),
        edge_index=torch.LongTensor(_get_edge_index_tuples(indices_groups)).T,
        names=label_lists,
        batch=torch.LongTensor(indices),
    )
    return graph_data


def _get_flattened_list_with_indices(text_lists: List[str]):
    indices = []
    texts = []
    for (i, text_list) in enumerate(text_lists):
        texts += text_list
        indices += [i] * len(text_list)
    return texts, indices


def _groupby_index(indices: List[int]):
    d = defaultdict(list)
    for (i, idx) in enumerate(indices):
        d[idx] += [i]
    return list(d.values())


def _get_edge_index_tuples(indices_groups):
    return [
        (v1, v2)
        for grp in indices_groups
        for v1, v2 in itertools.combinations_with_replacement(grp, 2)
    ]
