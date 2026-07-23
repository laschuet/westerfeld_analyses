import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

from _utils import calc_iou, edge_kingdom_type, _parse_node_kingdom

### (1) ###
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

def compare_graph_metrics(graphs: list[nx.Graph], labels: list[str]) -> pd.DataFrame:
    """One row per graph with `graph_metrics` columns."""
    return pd.DataFrame([graph_metrics(g) for g in graphs], index=labels)
### (1) ###

### (2) ###
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
### (2) ###

### (3) ###
def _filter_graph_by_node_kingdom(G: nx.Graph, kingdom: str | None) -> nx.Graph:
    if kingdom is None:
        return G
    return graph_subgraph_by_node_kingdom(G, kingdom)

def _filter_graph_by_edge_type(G: nx.Graph, edge_type: str | None) -> nx.Graph:
    if edge_type is None:
        return G
    return graph_subgraph_by_edge_kingdom(G, edge_type)

def _iou_nodes(g1, g2, pair_type: str | None = None):
    if pair_type is None:
        return calc_iou(list(g1.nodes), list(g2.nodes))
    g1 = _filter_graph_by_node_kingdom(g1, pair_type)
    g2 = _filter_graph_by_node_kingdom(g2, pair_type)
    return calc_iou(list(g1.nodes), list(g2.nodes))

def _iou_edges(g1, g2, pair_type: str | None = None):
    g1 = _filter_graph_by_edge_type(g1, pair_type)
    g2 = _filter_graph_by_edge_type(g2, pair_type)
    e1 = ["|".join(sorted(e)) for e in g1.edges]
    e2 = ["|".join(sorted(e)) for e in g2.edges]
    return calc_iou(e1, e2)

_METRICS = {
    "nodes_iou": _iou_nodes,
    "edges_iou": _iou_edges,
}
### (3) ###

### (4) ###
def _canonical_edges(G: nx.Graph):
    """Return edges as a set of sorted tuples (so (a,b) == (b,a))."""
    return {tuple(sorted(e)) for e in G.edges}

def shared_nodes(G1: nx.Graph, G2: nx.Graph):
    return sorted(set(G1.nodes) & set(G2.nodes))

def shared_edges(G1: nx.Graph, G2: nx.Graph):
    return sorted(_canonical_edges(G1) & _canonical_edges(G2))  

def _edge_type_edges(G: nx.Graph, edge_type: str | None) -> set[tuple]:
    return {
        tuple(sorted((u, v)))
        for u, v in G.edges()
        if edge_kingdom_type(G, u, v) == edge_type
    }

def shared_edges_by_type(G1: nx.Graph, G2: nx.Graph, edge_type: str) -> list:
    return sorted(_edge_type_edges(G1, edge_type) & _edge_type_edges(G2, edge_type))
### (4) ###

### (5) ###
def compare_graphs_pairwise(
    graphs: list[nx.Graph],
    labels: list[str],
    metric: str,
    pair_type: str | None = None,
    **metric_kwargs,
):
    """
    Pairwise matrix of `metric` across `graphs`.

    Supported metrics: `nodes_iou`, `edges_iou`

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

def compare_graphs_pairwise_edge_type_iou(
    graphs: list[nx.Graph], labels: list[str], edge_type: str
):
    return compare_graphs_pairwise(
        graphs, labels, "edges_iou", pair_type=edge_type
    )

def compare_graphs_pairwise_node_type_iou(
    graphs: list[nx.Graph], labels: list[str], kingdom: str
):
    return compare_graphs_pairwise(
        graphs, labels, "nodes_iou", pair_type=kingdom
    )
### (5) ###

### (6) ###
def node_kingdom(G: nx.Graph, node):
    return _parse_node_kingdom(G, node)

def graph_subgraph_by_node_kingdom(G: nx.Graph, kingdom: str):
    nodes = [n for n in G.nodes if _parse_node_kingdom(G, n) == kingdom]
    return G.subgraph(nodes).copy()

def graph_subgraph_by_edge_kingdom(G: nx.Graph, edge_type: str):
    H = nx.Graph()
    for u, v, data in G.edges(data=True):
        if edge_kingdom_type(G, u, v) == edge_type:
            H.add_edge(u, v, **data)
    nx.set_node_attributes(H, {n: G.nodes[n] for n in H.nodes})
    return H

def graph_metrics_by_kingdom(G: nx.Graph, kingdom: str):
    return graph_metrics(graph_subgraph_by_node_kingdom(G, kingdom))

def graph_metrics_by_edge_type(G: nx.Graph, edge_type: str):
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

def graph_edge_type_summary(G: nx.Graph, include_nodes: bool = False):
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
### (6) ###

### (7) ###
def plot_graphs_side_by_side(
    graphs: list[nx.Graph],
    labels: list[str],
    path: str = "graph_side_by_side.png",
    figsize: tuple[float, float] = (14, 7),
    node_size: int = 80,
    edge_width: float = 1.0,
    diff_mode: bool = False,  # NEU: Schalter für den Diff-Modus
):
    fig, axes = plt.subplots(1, len(graphs), figsize=figsize)
    if len(graphs) == 1:
        axes = [axes]

    if diff_mode and len(graphs) == 2:
        edges_g1 = set(graphs[0].edges())
        edges_g2 = set(graphs[1].edges())
        
        common_edges = edges_g1.intersection(edges_g2)
        unique_g1 = edges_g1 - edges_g2
        unique_g2 = edges_g2 - edges_g1

    for i, (ax, G, label) in enumerate(zip(axes, graphs, labels)):
        if G.number_of_nodes() == 0:
            ax.set_axis_off()
            continue

        if diff_mode and len(graphs) == 2:
            all_nodes = set(graphs[0].nodes()).union(set(graphs[1].nodes()))
            G_layout = nx.Graph()
            G_layout.add_nodes_from(all_nodes)
            G_layout.add_edges_from(edges_g1.union(edges_g2))
            pos = nx.spring_layout(G_layout, seed=42)
        else:
            pos = nx.spring_layout(G, seed=42)

        if diff_mode and len(graphs) == 2:

            if i == 0:
                current_unique = unique_g1
            else:
                current_unique = unique_g2

            if common_edges:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=list(common_edges),
                    edge_color="#d3d3d3",
                    width=edge_width * 0.5,
                    alpha=0.4,
                    ax=ax,
                )

            if current_unique:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=list(current_unique),
                    edge_color="#d62728", 
                    width=edge_width * 2.0,
                    alpha=0.9,
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

            ax.set_title(f"{label} (Diff Mode)")

        else:
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

    if diff_mode and len(graphs) == 2:
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=10, label="Fungi node"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=10, label="Bacteria node"),
            Line2D([0], [0], color="#d62728", linewidth=2, label="Unique to this habitat"),
            Line2D([0], [0], color="#d3d3d3", linewidth=2, label="Common in both"),
        ]
    else:
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

def plot_graphs_by_edge_type(
    graphs,
    labels,
    path="graph_by_edge_type.png",
    figsize=(18, 12),
    node_size_active=100,
    node_size_inactive=0,  # Auf 0 gesetzt, damit irrelevante Knoten komplett verschwinden
    edge_width=1.5,
):
    """
    Erstellt für jeden Input-Graphen drei Subplots: einen für jeden Edge-Type
    (Fungi-Fungi, Bacteria-Bacteria, Fungi-Bacteria).
    Knoten, die nicht zum aktuellen Edge-Type gehören, werden ausgeblendet.
    """

    edge_types = ["Fungi-Fungi", "Bacteria-Bacteria", "Fungi-Bacteria"]
    
    n_rows = len(graphs)
    n_cols = len(edge_types)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    edge_colors_map = {
        "Fungi-Fungi": "#1f77b4",      # Blau
        "Bacteria-Bacteria": "#ff7f0e", # Orange
        "Fungi-Bacteria": "#2ca02c",   # Grün
    }

    for i, (G, label) in enumerate(zip(graphs, labels)):
        for j, e_type in enumerate(edge_types):
            ax = axes[i, j]
            
            edges = [e for e in G.edges() if edge_kingdom_type(G, e[0], e[1]) == e_type]
            
            active_nodes = set()
            for u, v in edges:
                active_nodes.add(u)
                active_nodes.add(v)
            
            if active_nodes:
                G_sub = G.edge_subgraph(edges).copy()
                G_sub = G_sub.subgraph(active_nodes)
            else:
                G_sub = nx.Graph()

            if G_sub.number_of_nodes() == 0:
                ax.text(0.5, 0.5, "No edges", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{label}\n{e_type}")
                ax.set_axis_off()
                continue

            pos = nx.spring_layout(G_sub, seed=42)

            nx.draw_networkx_edges(
                G_sub,
                pos,
                edge_color=edge_colors_map.get(e_type, "gray"),
                width=edge_width,
                alpha=0.6,
                ax=ax,
            )

            node_colors = []
            for node in G_sub.nodes():
                if "Fungi:" in node:
                    node_colors.append("#1f77b4") # Blau für Pilze
                elif "Bacteria:" in node:
                    node_colors.append("#ff7f0e") # Orange für Bakterien
                else:
                    node_colors.append("gray")

            nx.draw_networkx_nodes(
                G_sub,
                pos,
                node_color=node_colors,
                node_size=node_size_active,
                ax=ax,
            )
        
            ax.set_title(f"{label} - {e_type}")
            ax.set_axis_off()

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=10, label="Fungi"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e", markersize=10, label="Bacteria"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Plot saved to {path}")
    return path

def plot_diff_grid(
    graphs: list[nx.Graph],
    labels: list[str],
    path: str = "graph_diff_grid.png",
    figsize: tuple[float, float] = (16, 16), # Quadratisch, da 2x2
    node_size: int = 80,
    edge_width: float = 1.0,
):

    if len(graphs) != 2:
        raise ValueError("Diese Funktion benötigt genau 2 Graphen für den 2x2 Vergleich.")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    edges_g1 = set(graphs[0].edges())
    edges_g2 = set(graphs[1].edges())
    
    common_edges = edges_g1.intersection(edges_g2)
    unique_g1 = edges_g1 - edges_g2
    unique_g2 = edges_g2 - edges_g1

    all_nodes = set(graphs[0].nodes()).union(set(graphs[1].nodes()))
    G_layout = nx.Graph()
    G_layout.add_nodes_from(all_nodes)
    G_layout.add_edges_from(edges_g1.union(edges_g2))
    pos = nx.spring_layout(G_layout, seed=42)

    def draw_subplot(ax, G, edges_to_draw, title, color_mode="common"):
        """
        Zeichnet einen einzelnen Subplot.
        color_mode: 'common' (bunt nach Typ) oder 'unique' (eine Farbe, z.B. Rot)
        """
        ax.set_title(title)
        
        if not edges_to_draw:
            ax.text(0.5, 0.5, "No edges", ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return

        if color_mode == "common":
            for edge_type in sorted({edge_kingdom_type(G, u, v) for u, v in edges_to_draw}):
                edges_of_type = [e for e in edges_to_draw if edge_kingdom_type(G, e[0], e[1]) == edge_type]
                nx.draw_networkx_edges(
                    G, pos, edgelist=edges_of_type,
                    edge_color=_edge_color(edge_type),
                    width=edge_width, alpha=0.8, ax=ax
                )
        else:
            nx.draw_networkx_edges(
                G, pos, edgelist=list(edges_to_draw),
                edge_color="#d62728", # Rot
                width=edge_width * 1.5, alpha=0.8, ax=ax
            )

        nx.draw_networkx_nodes(
            G, pos, nodelist=list(G.nodes),
            node_color=[_node_color(G, n) for n in G.nodes],
            node_size=node_size, ax=ax
        )
        
        ax.set_axis_off()

    draw_subplot(axes[0, 0], graphs[0], common_edges, f"{labels[0]} - Common Edges", color_mode="common")
    draw_subplot(axes[0, 1], graphs[0], unique_g1, f"{labels[0]} - Unique Edges", color_mode="unique")

    draw_subplot(axes[1, 0], graphs[1], common_edges, f"{labels[1]} - Common Edges", color_mode="common")
    draw_subplot(axes[1, 1], graphs[1], unique_g2, f"{labels[1]} - Unique Edges", color_mode="unique")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=10, label="Fungi node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=10, label="Bacteria node"),
        Line2D([0], [0], color="#1f77b4", linewidth=2, label="Fungi-Fungi (Common)"),
        Line2D([0], [0], color="#ff7f0e", linewidth=2, label="Bacteria-Bacteria (Common)"),
        Line2D([0], [0], color="#9467bd", linewidth=2, label="Fungi-Bacteria (Common)"),
        Line2D([0], [0], color="#d62728", linewidth=2, label="Unique Edge"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Diff-Grid saved to {path}")
    return path

def plot_common_only(
    graphs: list[nx.Graph],
    labels: list[str],
    path: str = "graph_common_only.png",
    figsize: tuple[float, float] = (10, 10),
    node_size: int = 150,
    edge_width: float = 2.0, 
):
    if len(graphs) != 2:
        raise ValueError("Exactly 2 graphs a required.")

    edges_g1 = set(graphs[0].edges())
    edges_g2 = set(graphs[1].edges())
    common_edges = edges_g1.intersection(edges_g2)

    if not common_edges:
        print("No common edges found.")
        return

    G_common = nx.edge_subgraph(graphs[0], common_edges).copy()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    pos = nx.spring_layout(G_common, seed=42, k=0.5)

    for edge_type in sorted({edge_kingdom_type(G_common, u, v) for u, v in G_common.edges()}):
        edges_of_type = [e for e in G_common.edges() if edge_kingdom_type(G_common, e[0], e[1]) == edge_type]
        nx.draw_networkx_edges(
            G_common,
            pos,
            edgelist=edges_of_type,
            edge_color=_edge_color(edge_type),
            width=edge_width,
            alpha=0.9,
            ax=ax,
        )

    nx.draw_networkx_nodes(
        G_common,
        pos,
        nodelist=list(G_common.nodes),
        node_color=[_node_color(G_common, n) for n in G_common.nodes],
        node_size=node_size,
        ax=ax,
    )
    
    ax.set_title(f"Common Edges Network ({G_common.number_of_nodes()} Nodes, {G_common.number_of_edges()} Edges)\nIntersection of {labels[0]} & {labels[1]}")
    ax.set_axis_off()

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=10, label="Fungi node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=10, label="Bacteria node"),
        Line2D([0], [0], color="#1f77b4", linewidth=2, label="Fungi-Fungi"),
        Line2D([0], [0], color="#ff7f0e", linewidth=2, label="Bacteria-Bacteria"),
        Line2D([0], [0], color="#9467bd", linewidth=2, label="Fungi-Bacteria"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"Common-Only Plot saved to {path}")
    return path

def plot_common_only_detailed(
    graphs: list[nx.Graph],
    labels: list[str],
    path: str = "graph_common_only_detailed.png",
    figsize: tuple[float, float] = (18, 6), 
    node_size: int = 300, 
    edge_width: float = 2.0,
):
    if len(graphs) != 2:
        raise ValueError("Exactly 2 graphs required.")

    edges_g1 = set(graphs[0].edges())
    edges_g2 = set(graphs[1].edges())
    common_edges = edges_g1.intersection(edges_g2)

    if not common_edges:
        print("No common edges found.")
        return

    G_common = nx.edge_subgraph(graphs[0], common_edges).copy()

    edge_types = ["Fungi-Fungi", "Bacteria-Bacteria", "Fungi-Bacteria"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, e_type in zip(axes, edge_types):
        edges = [e for e in G_common.edges() if edge_kingdom_type(G_common, e[0], e[1]) == e_type]
        
        if not edges:
            ax.text(0.5, 0.5, "No edges", ha='center', va='center')
            ax.set_title(f"{e_type}\n(0 edges)")
            ax.set_axis_off()
            continue

        G_sub = G_common.edge_subgraph(edges).copy()
        
        pos = nx.spring_layout(G_sub, seed=42, k=0.8) # k=0.8 sorgt für mehr Abstand

        nx.draw_networkx_edges(
            G_sub, pos,
            edge_color=_edge_color(e_type),
            width=edge_width,
            alpha=0.8,
            ax=ax
        )

        nx.draw_networkx_nodes(
            G_sub, pos,
            node_color=[_node_color(G_sub, n) for n in G_sub.nodes],
            node_size=node_size,
            ax=ax
        )

        labels_dict = {n: n.split(":")[-1] for n in G_sub.nodes()}
        nx.draw_networkx_labels(
            G_sub, pos, 
            labels=labels_dict, 
            font_size=9, 
            ax=ax
        )

        ax.set_title(f"{e_type}\n({G_sub.number_of_nodes()} Nodes, {G_sub.number_of_edges()} Edges)")
        ax.set_axis_off()

    fig.suptitle(f"Common Edges: {labels[0]} & {labels[1]}", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed Common-Only Plot saved to {path}")
    return path
### (7) ###

### (8) ###
def export_common_edges_to_excel(
    path: str, 
    graphs: list[nx.Graph],
    labels: list[str],
):
    if len(graphs) != 2:
        raise ValueError("Exactly 2 graphs required.")

    edges_g1 = set(graphs[0].edges())
    edges_g2 = set(graphs[1].edges())
    common_edges = edges_g1.intersection(edges_g2)

    if not common_edges:
        print("No common edges found to export.")
        return

    data = []
    G_ref = graphs[0]

    for u, v in common_edges:
        e_type = edge_kingdom_type(G_ref, u, v)
        
        weight = G_ref[u][v].get('weight', 0)
        
        name_u = u.split(":")[-1]
        name_v = v.split(":")[-1]

        data.append({
            "Edge_Type": e_type,
            "Taxon_1": name_u,
            "Taxon_2": name_v,
            "Correlation": weight,
        })

    df = pd.DataFrame(data)
    df = df.sort_values(by=["Edge_Type", "Correlation"], ascending=[True, False])

    df.to_excel(path, index=False)
    print(f"Common edges exported to {path} ({len(df)} edges).")
    return path
### (8) ###