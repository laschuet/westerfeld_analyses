import networkx as nx

from grakel.kernels import Kernel
from grakel.utils import graph_from_networkx


def grakel_graph(G: nx.Graph, attribute=None):
    if attribute is None:
        # Inject node names as dummy labels so grakel has something to work with
        G = G.copy()
        nx.set_node_attributes(G, {n: {"label": str(n)} for n in G.nodes})
        attribute = "label"
    return next(graph_from_networkx([G], node_labels_tag=attribute))


def graph_kernel(Gs: list[nx.Graph], kernel: Kernel, label=None):
    Gs = [grakel_graph(g, attribute=label) for g in Gs]
    K = kernel.fit_transform(Gs)
    return K
