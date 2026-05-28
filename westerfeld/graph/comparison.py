import networkx as nx
import numpy as np
import pandas as pd

from grakel import ShortestPath, WeisfeilerLehman
from grakel.kernels import Kernel
from grakel.utils import graph_from_networkx

from _utils import calc_iou


def graph_metrics(G: nx.Graph) -> dict:
    """
    Summary statistics for a single graph.

    Diameter / average shortest path length are reported on the largest
    connected component (they are undefined for disconnected graphs).
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0.0,
            "avg_degree": 0.0,
            "components": 0,
            "largest_cc": 0,
            "diameter": float("nan"),
            "avg_shortest_path": float("nan"),
            "avg_clustering": 0.0,
        }

    degrees = [d for _, d in G.degree()]
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    H = G.subgraph(largest_cc)
    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": nx.density(G),
        "avg_degree": float(np.mean(degrees)),
        "components": len(components),
        "largest_cc": len(largest_cc),
        "diameter": nx.diameter(H),
        "avg_shortest_path": nx.average_shortest_path_length(H),
        "avg_clustering": nx.average_clustering(G),
    }


def compare_graph_metrics(graphs: list[nx.Graph], labels: list[str]) -> pd.DataFrame:
    """One row per graph with `graph_metrics` columns."""
    return pd.DataFrame([graph_metrics(g) for g in graphs], index=labels)


def _canonical_edges(G: nx.Graph) -> set:
    """Return edges as a set of sorted tuples (so (a,b) == (b,a))."""
    return {tuple(sorted(e)) for e in G.edges}


def shared_nodes(G1: nx.Graph, G2: nx.Graph) -> list:
    return sorted(set(G1.nodes) & set(G2.nodes))


def shared_edges(G1: nx.Graph, G2: nx.Graph) -> list:
    return sorted(_canonical_edges(G1) & _canonical_edges(G2))


def is_subgraph(G_sub: nx.Graph, G: nx.Graph) -> bool:
    """True iff every node and every edge of G_sub is also in G."""
    if not set(G_sub.nodes) <= set(G.nodes):
        return False
    return _canonical_edges(G_sub) <= _canonical_edges(G)


def common_subgraph(G1: nx.Graph, G2: nx.Graph) -> nx.Graph:
    """Graph on the nodes both graphs share, keeping only edges they both have."""
    nodes = set(shared_nodes(G1, G2))
    edges = [e for e in shared_edges(G1, G2) if e[0] in nodes and e[1] in nodes]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def _graph_equal(g1: nx.Graph, g2: nx.Graph) -> bool:
    """Equal iff the node sets and (unordered) edge sets match."""
    return (
        set(g1.nodes) == set(g2.nodes)
        and _canonical_edges(g1) == _canonical_edges(g2)
    )


def find_similar_subgraphs(G1: nx.Graph, G2: nx.Graph, n: int = -1) -> list[nx.Graph]:
    """
    BFS enumeration of common connected substructures of `G1` and `G2`.

    Seeds with one trivial subgraph per shared node, then repeatedly extends
    each candidate by one shared edge whose `positive_association` attribute
    agrees in both graphs. Returns every reachable substructure (deepest last);
    pass `n` to return only the last `n` of them.

    Can be expensive on large dense graphs: the enumeration runs to completion
    regardless of `n` (which only slices the return).
    """
    candidate_edges = {
        e
        for e in _canonical_edges(G1) & _canonical_edges(G2)
        if G1.edges[e].get("positive_association")
        == G2.edges[e].get("positive_association")
    }

    structures: list[nx.Graph] = []
    for node in shared_nodes(G1, G2):
        h = nx.Graph()
        h.add_node(node)
        structures.append(h)

    frontier = list(structures)
    while frontier:
        new_structures: list[nx.Graph] = []
        for g in frontier:
            g_nodes = set(g.nodes)
            g_edges = _canonical_edges(g)
            for e in candidate_edges:
                if e in g_edges:
                    continue
                if e[0] not in g_nodes and e[1] not in g_nodes:
                    continue
                h = g.copy()
                h.add_edge(
                    *e,
                    positive_association=G1.edges[e].get("positive_association"),
                )
                if any(_graph_equal(h, s) for s in new_structures):
                    continue
                new_structures.append(h)
        structures.extend(new_structures)
        frontier = new_structures

    if n != -1:
        return structures[-n:]
    return structures


def _grakel_graph(G: nx.Graph, attribute=None):
    if attribute is None:
        # Inject node names as dummy labels so grakel has something to work with
        G = G.copy()
        nx.set_node_attributes(G, {n: {"label": str(n)} for n in G.nodes})
        attribute = "label"
    return next(graph_from_networkx([G], node_labels_tag=attribute))


def graph_kernel(graphs: list[nx.Graph], kernel: Kernel, label=None):
    grakel_graphs = [_grakel_graph(g, attribute=label) for g in graphs]
    return kernel.fit_transform(grakel_graphs)


def _iou_nodes(g1, g2):
    return calc_iou(list(g1.nodes), list(g2.nodes))


def _iou_edges(g1, g2):
    e1 = ["|".join(sorted(e)) for e in g1.edges]
    e2 = ["|".join(sorted(e)) for e in g2.edges]
    return calc_iou(e1, e2)


def _kernel_shortest_path(g1, g2, normalize=True):
    return graph_kernel([g1, g2], ShortestPath(normalize=normalize))[1, 0]


def _kernel_weisfeiler_lehman(g1, g2, normalize=True):
    return graph_kernel([g1, g2], WeisfeilerLehman(normalize=normalize))[1, 0]


_METRICS = {
    "nodes_iou": _iou_nodes,
    "edges_iou": _iou_edges,
    "kernel_shortest_path": _kernel_shortest_path,
    "kernel_weisfeiler_lehman": _kernel_weisfeiler_lehman,
}


def compare_graphs_pairwise(
    graphs: list[nx.Graph], labels: list[str], metric: str, **metric_kwargs
) -> pd.DataFrame:
    """
    Pairwise matrix of `metric` across `graphs`.

    Supported metrics: `nodes_iou`, `edges_iou`, `kernel_shortest_path`,
    `kernel_weisfeiler_lehman`. Kernel metrics accept `normalize` (default
    `True`).
    """
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric: {metric} (available: {sorted(_METRICS)})")
    fn = _METRICS[metric]
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, gi in enumerate(graphs):
        for j, gj in enumerate(graphs):
            matrix.iloc[i, j] = fn(gi, gj, **metric_kwargs)
    return matrix
