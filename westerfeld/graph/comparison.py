import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
            "modularity": -1.0,
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
        "modularity": nx.community.modularity(
            G, nx.community.label_propagation_communities(G)
        ),
    }


def _parse_node_kingdom(G: nx.Graph, node):
    if "kingdom" in G.nodes[node]:
        return G.nodes[node]["kingdom"]
    if isinstance(node, str) and ":" in node:
        return node.split(":", 1)[0]
    return None


def _canonical_edge_kingdom_type(kingdom_a, kingdom_b):
    if kingdom_a is None or kingdom_b is None:
        return None
    if kingdom_a == kingdom_b:
        return f"{kingdom_a}-{kingdom_a}"
    if {kingdom_a, kingdom_b} == {"Fungi", "Bacteria"}:
        return "Fungi-Bacteria"
    return "-".join(sorted([kingdom_a, kingdom_b]))


def node_kingdom(G: nx.Graph, node):
    return _parse_node_kingdom(G, node)


def edge_kingdom_type(G: nx.Graph, u, v):
    attr = G.edges[u, v].get("kingdom_edge")
    if attr is not None:
        return attr
    return _canonical_edge_kingdom_type(_parse_node_kingdom(G, u), _parse_node_kingdom(G, v))


def graph_subgraph_by_node_kingdom(G: nx.Graph, kingdom: str) -> nx.Graph:
    nodes = [n for n in G.nodes if _parse_node_kingdom(G, n) == kingdom]
    return G.subgraph(nodes).copy()


def graph_subgraph_by_edge_kingdom(G: nx.Graph, edge_type: str) -> nx.Graph:
    H = nx.Graph()
    for u, v, data in G.edges(data=True):
        if edge_kingdom_type(G, u, v) == edge_type:
            H.add_edge(u, v, **data)
    nx.set_node_attributes(H, {n: G.nodes[n] for n in H.nodes})
    return H


def graph_metrics_by_kingdom(G: nx.Graph, kingdom: str) -> dict:
    return graph_metrics(graph_subgraph_by_node_kingdom(G, kingdom))


def graph_metrics_by_edge_type(G: nx.Graph, edge_type: str) -> dict:
    return graph_metrics(graph_subgraph_by_edge_kingdom(G, edge_type))


def graph_node_type_summary(G: nx.Graph) -> pd.DataFrame:
    """Summarize each kingdom's induced subgraph by node type."""
    kingdoms = sorted({
        _parse_node_kingdom(G, n)
        for n in G.nodes
        if _parse_node_kingdom(G, n) is not None
    })
    summary = []
    for kingdom in kingdoms:
        sub = graph_subgraph_by_node_kingdom(G, kingdom)
        degrees = [d for _, d in sub.degree()]
        summary.append(
            {
                "kingdom": kingdom,
                "nodes": sub.number_of_nodes(),
                "edges": sub.number_of_edges(),
                "density": nx.density(sub),
                "avg_degree": float(np.mean(degrees)) if degrees else 0.0,
                "components": nx.number_connected_components(sub),
            }
        )
    return pd.DataFrame(summary).set_index("kingdom")


def _edge_type_edges(G: nx.Graph, edge_type: str | None) -> set[tuple]:
    return {
        tuple(sorted((u, v)))
        for u, v in G.edges()
        if edge_kingdom_type(G, u, v) == edge_type
    }


def shared_edges_by_type(G1: nx.Graph, G2: nx.Graph, edge_type: str) -> list:
    return sorted(_edge_type_edges(G1, edge_type) & _edge_type_edges(G2, edge_type))


def compare_graphs_pairwise_edge_type_iou(
    graphs: list[nx.Graph], labels: list[str], edge_type: str
) -> pd.DataFrame:
    return compare_graphs_pairwise(
        graphs, labels, "edges_iou", pair_type=edge_type
    )


def compare_graphs_pairwise_node_type_iou(
    graphs: list[nx.Graph], labels: list[str], kingdom: str
) -> pd.DataFrame:
    return compare_graphs_pairwise(
        graphs, labels, "nodes_iou", pair_type=kingdom
    )


def _node_color(G: nx.Graph, node):
    kind = _parse_node_kingdom(G, node)
    return {
        "Fungi": "#1f77b4",
        "Bacteria": "#2ca02c",
    }.get(kind, "#7f7f7f")


def _edge_color(edge_type: str):
    return {
        "Fungi-Fungi": "#1f77b4",
        "Bacteria-Bacteria": "#ff7f0e",
        "Fungi-Bacteria": "#9467bd",
    }.get(edge_type, "#7f7f7f")


def plot_graphs_side_by_side(
    graphs: list[nx.Graph],
    labels: list[str],
    path: str = "graph_side_by_side.png",
    figsize: tuple[float, float] = (14, 7),
    node_size: int = 80,
    edge_width: float = 1.0,
):
    fig, axes = plt.subplots(1, len(graphs), figsize=figsize)
    if len(graphs) == 1:
        axes = [axes]

    for ax, G, label in zip(axes, graphs, labels):
        if G.number_of_nodes() == 0:
            ax.set_axis_off()
            continue

        pos = nx.spring_layout(G, seed=42)

        for edge_type in sorted({edge_kingdom_type(G, u, v) for u, v in G.edges()}):
            edges = [e for e in G.edges() if edge_kingdom_type(G, e[0], e[1]) == edge_type]
            if not edges:
                continue
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                edge_color=_edge_color(edge_type),
                width=edge_width,
                alpha=0.8,
                ax=ax,
            )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(G.nodes),
            node_color=[_node_color(G, n) for n in G.nodes],
            node_size=node_size,
            ax=ax,
        )

        ax.set_title(label)
        ax.set_axis_on()
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=10, label="Fungi node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=10, label="Bacteria node"),
        Line2D([0], [0], color="#1f77b4", linewidth=2, label="Fungi-Fungi edge"),
        Line2D([0], [0], color="#ff7f0e", linewidth=2, label="Bacteria-Bacteria edge"),
        Line2D([0], [0], color="#9467bd", linewidth=2, label="Fungi-Bacteria edge"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def graph_edge_type_summary(G: nx.Graph, include_nodes: bool = False) -> pd.DataFrame:
    """Summarize edge-type-specific subgraphs.

    The edge sets for different types are disjoint, but nodes may appear in
    more than one edge-type subgraph. By default, the table reports only
    edge-focused metrics so it is not misleading.
    """
    counts: dict[str, int] = {}
    for u, v in G.edges():
        et = edge_kingdom_type(G, u, v)
        counts[et] = counts.get(et, 0) + 1
    summary = []
    for edge_type, count in sorted(counts.items()):
        sub = graph_subgraph_by_edge_kingdom(G, edge_type)
        row = {
            "edge_type": edge_type,
            "edges": count,
            "density": nx.density(sub),
            "avg_degree": float(np.mean([d for _, d in sub.degree()]))
            if sub.number_of_nodes() > 0
            else 0.0,
        }
        if include_nodes:
            row["active_nodes"] = sub.number_of_nodes()
        summary.append(row)
    return pd.DataFrame(summary).set_index("edge_type")

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
        "modularity": nx.community.modularity(
            G, nx.community.label_propagation_communities(G)
        ),
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
    return set(g1.nodes) == set(g2.nodes) and _canonical_edges(g1) == _canonical_edges(
        g2
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


def _filter_graph_by_node_kingdom(G: nx.Graph, kingdom: str | None) -> nx.Graph:
    if kingdom is None:
        return G
    return graph_subgraph_by_node_kingdom(G, kingdom)


def _iou_nodes(g1, g2, pair_type: str | None = None):
    if pair_type is None:
        return calc_iou(list(g1.nodes), list(g2.nodes))
    g1 = _filter_graph_by_node_kingdom(g1, pair_type)
    g2 = _filter_graph_by_node_kingdom(g2, pair_type)
    return calc_iou(list(g1.nodes), list(g2.nodes))


def _filter_graph_by_edge_type(G: nx.Graph, edge_type: str | None) -> nx.Graph:
    if edge_type is None:
        return G
    return graph_subgraph_by_edge_kingdom(G, edge_type)


def _iou_edges(g1, g2, pair_type: str | None = None):
    g1 = _filter_graph_by_edge_type(g1, pair_type)
    g2 = _filter_graph_by_edge_type(g2, pair_type)
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
    graphs: list[nx.Graph],
    labels: list[str],
    metric: str,
    pair_type: str | None = None,
    **metric_kwargs,
) -> pd.DataFrame:
    """
    Pairwise matrix of `metric` across `graphs`.

    Supported metrics: `nodes_iou`, `edges_iou`, `kernel_shortest_path`,
    `kernel_weisfeiler_lehman`. Kernel metrics accept `normalize` (default
    `True`).

    For `edges_iou`, `pair_type` can be used to restrict the comparison to
    a specific edge type:

      - `pair_type='Fungi-Fungi'`
      - `pair_type='Bacteria-Bacteria'`
      - `pair_type='Fungi-Bacteria'`

    For `nodes_iou`, `pair_type` can be used to restrict the comparison to a
    specific kingdom's node set:

      - `pair_type='Fungi'`
      - `pair_type='Bacteria'`

    The matrix itself does not label the selected pair type; it only computes
    the requested metric on the filtered node or edge set.
    """
    if metric not in _METRICS:
        raise ValueError(f"Unknown metric: {metric} (available: {sorted(_METRICS)})")
    fn = _METRICS[metric]
    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, gi in enumerate(graphs):
        for j, gj in enumerate(graphs):
            if metric in {"edges_iou", "nodes_iou"}:
                matrix.iloc[i, j] = fn(gi, gj, pair_type=pair_type, **metric_kwargs)
            else:
                if pair_type is not None:
                    raise ValueError(
                        "pair_type is only supported for metrics 'edges_iou' and 'nodes_iou'"
                    )
                matrix.iloc[i, j] = fn(gi, gj, **metric_kwargs)
    return matrix
