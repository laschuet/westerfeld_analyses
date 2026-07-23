import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

from _utils import calc_iou, edge_kingdom_type, _parse_node_kingdom

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

def node_kingdom(G: nx.Graph, node):
    return _parse_node_kingdom(G, node)

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
    diff_mode: bool = False,  # NEU: Schalter für den Diff-Modus
):
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(graphs), figsize=figsize)
    if len(graphs) == 1:
        axes = [axes]

    # --- Vorbereitung für den Diff-Mode ---
    # Wir berechnen die Sets der Kanten für beide Graphen
    if diff_mode and len(graphs) == 2:
        edges_g1 = set(graphs[0].edges())
        edges_g2 = set(graphs[1].edges())
        
        # Gemeinsame Kanten (Core)
        common_edges = edges_g1.intersection(edges_g2)
        # Kanten nur in G1
        unique_g1 = edges_g1 - edges_g2
        # Kanten nur in G2
        unique_g2 = edges_g2 - edges_g1

    # --- Plotting Loop ---
    for i, (ax, G, label) in enumerate(zip(axes, graphs, labels)):
        if G.number_of_nodes() == 0:
            ax.set_axis_off()
            continue

        # Layout berechnen (wir nutzen das gleiche Layout für beide Graphen im Diff-Mode, 
        # damit die Knoten an der gleichen Stelle bleiben und man besser vergleichen kann)
        # Im Normal-Mode berechnen wir es pro Graph, da die Knotenmengen ja unterschiedlich sein könnten
        if diff_mode and len(graphs) == 2:
            # Wir nutzen die Vereinigungsmenge der Knoten für das Layout, damit beide Plots identisch ausgerichtet sind
            all_nodes = set(graphs[0].nodes()).union(set(graphs[1].nodes()))
            # Erstelle einen temporären Graphen nur für das Layout
            G_layout = nx.Graph()
            G_layout.add_nodes_from(all_nodes)
            # Füge alle Kanten hinzu, damit die Abstände stimmen
            G_layout.add_edges_from(edges_g1.union(edges_g2))
            pos = nx.spring_layout(G_layout, seed=42)
        else:
            pos = nx.spring_layout(G, seed=42)

        if diff_mode and len(graphs) == 2:
            # --- DIFF MODE LOGIK ---
            
            # Bestimme welche Kanten in diesem Graphen 'unique' sind
            if i == 0:
                current_unique = unique_g1
            else:
                current_unique = unique_g2

            # 1. Zeichne gemeinsame Kanten (Grau, dünn)
            if common_edges:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=list(common_edges),
                    edge_color="#d3d3d3", # Hellgrau
                    width=edge_width * 0.5,
                    alpha=0.4,
                    ax=ax,
                )

            # 2. Zeichne unique Kanten (Farbig, dick)
            if current_unique:
                nx.draw_networkx_edges(
                    G,
                    pos,
                    edgelist=list(current_unique),
                    edge_color="#d62728", # Rot (oder eine andere Signalfarbe)
                    width=edge_width * 2.0,
                    alpha=0.9,
                    ax=ax,
                )
            
            # Knoten zeichnen (Option A: Kingdom Colors)
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=list(G.nodes),
                node_color=[_node_color(G, n) for n in G.nodes],
                node_size=node_size,
                ax=ax,
            )

            # Titel anpassen
            ax.set_title(f"{label} (Diff Mode)")

        else:
            # --- NORMALER MODUS (Original Code) ---
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

    # --- Legende ---
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

    # Definieren der Edge-Types und Reihenfolge
    edge_types = ["Fungi-Fungi", "Bacteria-Bacteria", "Fungi-Bacteria"]
    
    # Anzahl der Zeilen = Anzahl der Input-Graphen (z.B. Field, Rhizo)
    # Anzahl der Spalten = Anzahl der Edge-Types (3)
    n_rows = len(graphs)
    n_cols = len(edge_types)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Falls nur ein Graph übergeben wird, axes ist 1D, wir brauchen 2D für konsistentes Indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Farben für die Edge-Types (optional, für bessere Unterscheidung)
    edge_colors_map = {
        "Fungi-Fungi": "#1f77b4",      # Blau
        "Bacteria-Bacteria": "#ff7f0e", # Orange
        "Fungi-Bacteria": "#2ca02c",   # Grün
    }

    for i, (G, label) in enumerate(zip(graphs, labels)):
        for j, e_type in enumerate(edge_types):
            ax = axes[i, j]
            
            # 1. Filtere Kanten des aktuellen Typs
            edges = [e for e in G.edges() if edge_kingdom_type(G, e[0], e[1]) == e_type]
            
            # 2. Bestimme die Knoten, die an diesen Kanten beteiligt sind
            active_nodes = set()
            for u, v in edges:
                active_nodes.add(u)
                active_nodes.add(v)
            
            # Erstelle einen Subgraphen, der nur diese Knoten und Kanten enthält
            # Das macht das Plotten einfacher und sorgt dafür, dass Layout-Algorithmen
            # sich nur auf den relevanten Teil konzentrieren.
            if active_nodes:
                G_sub = G.edge_subgraph(edges).copy()
                # node_subgraph würde auch gehen, aber edge_subgraph impliziert die Knoten meist schon.
                # Sicherstellen, dass nur die verbundenen Knoten drin sind:
                G_sub = G_sub.subgraph(active_nodes)
            else:
                G_sub = nx.Graph()

            if G_sub.number_of_nodes() == 0:
                ax.text(0.5, 0.5, "No edges", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{label}\n{e_type}")
                ax.set_axis_off()
                continue

            # Layout berechnen (Seed für Reproduzierbarkeit)
            # Wir nutzen hier das Layout auf dem Subgraphen, damit die Knoten nah beieinander liegen
            pos = nx.spring_layout(G_sub, seed=42)

            # Kanten zeichnen
            nx.draw_networkx_edges(
                G_sub,
                pos,
                edge_color=edge_colors_map.get(e_type, "gray"),
                width=edge_width,
                alpha=0.6,
                ax=ax,
            )

            # Knoten zeichnen
            # Wir färben die Knoten hier einfach nach ihrer Kingdom-Herkunft, 
            # damit man sieht, wer wer ist (hilfreich bei Fungi-Bacteria)
            node_colors = []
            for node in G_sub.nodes():
                # Annahme: Node-Name ist "Kingdom:Taxon", z.B. "Fungi:GenusX"
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
            
            # Optional: Labels (kann bei vielen Knoten unübersichtlich werden, erstmal auskommentiert)
            # nx.draw_networkx_labels(G_sub, pos, font_size=8, ax=ax)

            # Titel und Achsen
            ax.set_title(f"{label} - {e_type}")
            ax.set_axis_off()

    # Legende für die Knotenfarben (Kingdoms)
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
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    # Wir brauchen genau 2 Graphen für diesen Vergleich
    if len(graphs) != 2:
        raise ValueError("Diese Funktion benötigt genau 2 Graphen für den 2x2 Vergleich.")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # axes ist ein 2D Array [[ax00, ax01], [ax10, ax11]]
    # Wir flachen es nicht ab, sondern nutzen explizit die Indizes
    
    # --- Vorbereitung: Kanten berechnen ---
    edges_g1 = set(graphs[0].edges())
    edges_g2 = set(graphs[1].edges())
    
    common_edges = edges_g1.intersection(edges_g2)
    unique_g1 = edges_g1 - edges_g2
    unique_g2 = edges_g2 - edges_g1

    # --- Layout berechnen ---
    # Wir nutzen ein gemeinsames Layout für ALLE 4 Plots, damit man Knoten leicht vergleichen kann.
    # Basis ist die Vereinigung aller Knoten und Kanten.
    all_nodes = set(graphs[0].nodes()).union(set(graphs[1].nodes()))
    G_layout = nx.Graph()
    G_layout.add_nodes_from(all_nodes)
    G_layout.add_edges_from(edges_g1.union(edges_g2))
    pos = nx.spring_layout(G_layout, seed=42)

    # --- Plotting Helper ---
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

        # Kanten zeichnen
        if color_mode == "common":
            # Bunt nach Edge-Type
            for edge_type in sorted({edge_kingdom_type(G, u, v) for u, v in edges_to_draw}):
                edges_of_type = [e for e in edges_to_draw if edge_kingdom_type(G, e[0], e[1]) == edge_type]
                nx.draw_networkx_edges(
                    G, pos, edgelist=edges_of_type,
                    edge_color=_edge_color(edge_type),
                    width=edge_width, alpha=0.8, ax=ax
                )
        else:
            # Unique: Einheitliche Farbe (z.B. Rot oder Dunkelgrau)
            nx.draw_networkx_edges(
                G, pos, edgelist=list(edges_to_draw),
                edge_color="#d62728", # Rot
                width=edge_width * 1.5, alpha=0.8, ax=ax
            )

        # Knoten zeichnen (immer gleich)
        # Wir zeichnen nur Knoten, die auch in diesem Graph G existieren
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(G.nodes),
            node_color=[_node_color(G, n) for n in G.nodes],
            node_size=node_size, ax=ax
        )
        
        ax.set_axis_off()

    # --- Die 4 Plots füllen ---
    
    # Reihe 1: Field Soil (Graph 0)
    # Links: Common
    draw_subplot(axes[0, 0], graphs[0], common_edges, f"{labels[0]} - Common Edges", color_mode="common")
    # Rechts: Unique
    draw_subplot(axes[0, 1], graphs[0], unique_g1, f"{labels[0]} - Unique Edges", color_mode="unique")

    # Reihe 2: Rhizosphere (Graph 1)
    # Links: Common
    draw_subplot(axes[1, 0], graphs[1], common_edges, f"{labels[1]} - Common Edges", color_mode="common")
    # Rechts: Unique
    draw_subplot(axes[1, 1], graphs[1], unique_g2, f"{labels[1]} - Unique Edges", color_mode="unique")

    # --- Legende ---
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

    # 1. Find common edges 
    edges_g1 = set(graphs[0].edges())
    edges_g2 = set(graphs[1].edges())
    common_edges = edges_g1.intersection(edges_g2)

    if not common_edges:
        print("No common edges found.")
        return

    G_common = nx.edge_subgraph(graphs[0], common_edges).copy()
    
    # Optional: Falls du willst, dass die Knotenbeschriftungen (Labels) angezeigt werden, 
    # da es jetzt übersichtlich ist, könntest du das hier einkommentieren:
    # node_labels = {n: n.split(":")[-1] for n in G_common.nodes()} # Nur den Taxon-Namen ohne Kingdom

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Layout berechnen (nur für den Common-Graphen)
    pos = nx.spring_layout(G_common, seed=42, k=0.5) # k=0.5 zieht die Knoten etwas auseinander

    # Kanten zeichnen (Bunt nach Typ)
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

    # Knoten zeichnen
    nx.draw_networkx_nodes(
        G_common,
        pos,
        nodelist=list(G_common.nodes),
        node_color=[_node_color(G_common, n) for n in G_common.nodes],
        node_size=node_size,
        ax=ax,
    )
    
    # Optional: Labels zeichnen (Vorsicht: Bei langen Namen wird es unübersichtlich)
    # nx.draw_networkx_labels(G_common, pos, labels=node_labels, font_size=8, ax=ax)

    ax.set_title(f"Common Edges Network ({G_common.number_of_nodes()} Nodes, {G_common.number_of_edges()} Edges)\nIntersection of {labels[0]} & {labels[1]}")
    ax.set_axis_off()

    # Legende
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

def plot_common_only_detailed(
    graphs: list[nx.Graph],
    labels: list[str],
    path: str = "graph_common_only_detailed.png",
    figsize: tuple[float, float] = (18, 6), # Breit für 3 Spalten
    node_size: int = 300, # Größer, da wir weniger pro Plot haben
    edge_width: float = 2.0,
):
    import matplotlib.pyplot as plt
    import networkx as nx

    if len(graphs) != 2:
        raise ValueError("Exactly 2 graphs required.")

    edges_g1 = set(graphs[0].edges())
    edges_g2 = set(graphs[1].edges())
    common_edges = edges_g1.intersection(edges_g2)

    if not common_edges:
        print("No common edges found.")
        return

    # Wir erstellen einen temporären Graphen nur für die Common Edges
    G_common = nx.edge_subgraph(graphs[0], common_edges).copy()

    # Definiere die 3 Subplots
    edge_types = ["Fungi-Fungi", "Bacteria-Bacteria", "Fungi-Bacteria"]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, e_type in zip(axes, edge_types):
        # 1. Filtere Kanten und Knoten für diesen Typ
        edges = [e for e in G_common.edges() if edge_kingdom_type(G_common, e[0], e[1]) == e_type]
        
        if not edges:
            ax.text(0.5, 0.5, "No edges", ha='center', va='center')
            ax.set_title(f"{e_type}\n(0 edges)")
            ax.set_axis_off()
            continue

        # Subgraphen erstellen für sauberes Layout
        G_sub = G_common.edge_subgraph(edges).copy()
        
        # Layout berechnen
        pos = nx.spring_layout(G_sub, seed=42, k=0.8) # k=0.8 sorgt für mehr Abstand

        # Kanten zeichnen
        nx.draw_networkx_edges(
            G_sub, pos,
            edge_color=_edge_color(e_type),
            width=edge_width,
            alpha=0.8,
            ax=ax
        )

        # Knoten zeichnen
        nx.draw_networkx_nodes(
            G_sub, pos,
            node_color=[_node_color(G_sub, n) for n in G_sub.nodes],
            node_size=node_size,
            ax=ax
        )

        # LABELS ZEICHNEN
        # Wir nehmen nur den Genus-Namen (alles nach dem Doppelpunkt)
        labels_dict = {n: n.split(":")[-1] for n in G_sub.nodes()}
        nx.draw_networkx_labels(
            G_sub, pos, 
            labels=labels_dict, 
            font_size=9, 
            ax=ax
        )

        ax.set_title(f"{e_type}\n({G_sub.number_of_nodes()} Nodes, {G_sub.number_of_edges()} Edges)")
        ax.set_axis_off()

    # Haupttitel
    fig.suptitle(f"Common Edges: {labels[0]} & {labels[1]}", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed Common-Only Plot saved to {path}")
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
    return graph_kernel([g1, g2], WeisfeilerLehman(normalize=normalize))[1, 0]

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

_METRICS = {
    "nodes_iou": _iou_nodes,
    "edges_iou": _iou_edges,
}

