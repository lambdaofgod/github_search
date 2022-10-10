from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt


def plot_networkx(subdata):
    fig = plt.figure(figsize=(16, 16))

    G = to_networkx(subdata, to_undirected=True)
    G = nx.relabel_nodes(G, mapping=dict(enumerate(subdata.vertex_names)))
    plt.axis("off")
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G, seed=0),
        with_labels=True,
        node_size=50,
        alpha=0.9,
        edge_color="gray",
        node_color="red",
        font_size=10,
    )
    plt.show()
