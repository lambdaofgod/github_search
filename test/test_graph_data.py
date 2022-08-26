from collections import defaultdict

import igraph
from github_search.graphs import graph_preprocessor


class MockEmbedder:
    def extract_features(self, xs, **kwargs):
        return [[0, 1, -1]] * len(xs)


graph = igraph.Graph()
vertices = ["v0"] + [f"v{i}:0:0" for i in range(1, 5)]
graph.add_vertices(vertices, attributes={"names": vertices, "text": vertices})
graph.add_edges(zip(vertices, vertices[1:]))
label_mapping = {vertices[0]: {"class": "class1"}}


def test_graph_data_preprocessor():
    preprocessor = graph_preprocessor.GraphDataPreprocessor(
        MockEmbedder(), label_mapping
    )
    data_list = preprocessor.get_data_list(graph)
    assert label_mapping.get("v0") is not None
    assert len(data_list) == 1
    graph_data = data_list[0]
    assert graph_data.x.shape == (len(vertices), 3)
    assert graph_data.names == vertices
    assert graph_data.graph_name == "v0"
