import networkx as nx

from grakel.utils import graph_from_networkx
from grakel.kernels import ShortestPath, RandomWalk, WeisfeilerLehman, Kernel
from graph.settings import USE_PHYLUM_LABELS_FOR_WEISFEILER_LEHMAN_KERNEL

THREADS_AMOUNT = 1

def generic_kernel_on_graph(Gs_nx: list[nx.Graph], kernal_to_use: Kernel, label=None):
    Gs = [to_grakel_graph(g, attribute=label)for g in Gs_nx]
    K = kernal_to_use.fit_transform(Gs)
    return K


def to_grakel_graph(G_nx: nx.Graph, attribute=None):
    G_nx = G_nx.copy()
    if (attribute == None):
        # in this case we would need to create "dummy" labels.
        # we will use the nodes index/name, and thus, we will use the (unique) species name.
        nodesAttr = dict(G_nx.nodes)
        for n in G_nx.nodes:
            nodesAttr[n] = {"label": str(n)}
        nx.set_node_attributes(G_nx, nodesAttr)
        attribute = "label"
    G = list(graph_from_networkx([G_nx], node_labels_tag=attribute))[0]
    return G


def kernel_graph(Gs_nx: list[nx.Graph], kernel: str = "", normalize=False):
    match kernel:
        case "shortest_path":
            K = generic_kernel_on_graph(Gs_nx, ShortestPath(normalize))
        case "random_walk":
            K = generic_kernel_on_graph(
                Gs_nx,
                RandomWalk(normalize=normalize)
            )
        case "weisfeiler_lehman":
            K = generic_kernel_on_graph(
                Gs_nx,
                WeisfeilerLehman(normalize=normalize),
                label="Phylum" if USE_PHYLUM_LABELS_FOR_WEISFEILER_LEHMAN_KERNEL else None
            )
        case _:
            K = generic_kernel_on_graph(Gs_nx, ShortestPath(normalize))
    return K


if __name__ == "__main__":
    G1 = nx.path_graph(4)
    G2 = nx.path_graph(5)
    print(kernel_graph([G1, G2], "shortest_path"))
    # print(kernel_graph([G1, G2],"random_walk"))
    print(kernel_graph([G1, G2], "weisfeiler_lehman"))
    print(kernel_graph([G1, G2]))
