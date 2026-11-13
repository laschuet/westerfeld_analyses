import numpy as np
import networkx as nx

def calc_iou(d1, d2):
    """Jaccard index (intersection over union) of two collections."""
    union = set(d1 + d2)
    intersect = np.intersect1d(d1, d2)
    return len(intersect) / len(union)

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

def edge_kingdom_type(G: nx.Graph, u, v):
    attr = G.edges[u, v].get("kingdom_edge")
    if attr is not None:
        return attr
    return _canonical_edge_kingdom_type(_parse_node_kingdom(G, u), _parse_node_kingdom(G, v))

def _annotate_kingdoms(G):
    nx.set_node_attributes(
        G,
        {node: _parse_node_kingdom(G,node) for node in G.nodes},
        "kingdom",
    )
    nx.set_edge_attributes(
        G,
        {
            (u, v): edge_kingdom_type(G, u, v)
            for u, v in G.edges
        },
        "kingdom_edge",
    )
