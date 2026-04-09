import logging
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import entropy

from graph.comparison.kernels import kernel_graph
from graph.creation.registry import get_graph_creator
from graph.utils import calc_iou, preprocessing, visualize_graphs, create_figure_simple


def are_graphs_essentially_the_same(g1: nx.Graph, g2: nx.Graph):
    """Check if graphs are equal.

    Equality here means the same nodes and edges

    Parameters
    ----------
    g1, g2 : graph

    Returns
    -------
    bool
        True if graphs are equal, False otherwise.
    """
    if sorted(list(g1.nodes)) == sorted(list(g2.nodes)):
        return True
    if sorted([(sorted(item)) for item in g1.edges]) == sorted(
        [(sorted(item)) for item in g2.edges]
    ):
        return True
    return False


def get_shared_nodes(G1: nx.Graph, G2: nx.Graph) -> np.ndarray:
    return np.intersect1d(list(G1.nodes), list(G2.nodes))


def sort_items_universally(items):
    return list([(sorted(item)) for item in items])


def get_shared_edges(G1: nx.Graph, G2: nx.Graph) -> list:
    shared_edges = [
        list(e)
        for e in [(sorted(item)) for item in G1.edges]
        if e in [(sorted(item)) for item in G2.edges]
    ]  # sort edges, so e1=(n1,n2) is equal to e2=(n2,n1)
    return shared_edges


def contains_graph(G: nx.Graph, G_list: list[nx.Graph]):
    for g in G_list:
        if are_graphs_essentially_the_same(g, G):
            return True
    return False


def find_suitable_edges(G1: nx.Graph, G2: nx.Graph, g: nx.Graph):
    possible = get_shared_edges(G1, G2)
    possible = list(
        filter(lambda x: x[0] in list(g.nodes) or x[1] in list(g.nodes), possible)
    )
    possible = list(
        filter(lambda x: sorted(x) not in sort_items_universally(g.edges), possible)
    )
    return possible


def create_graph_from(nodes, edges) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def expand_graph_by_edge(G: nx.Graph, edge, original_graph: nx.Graph) -> nx.Graph:
    G_copy = G.copy()
    G_copy.add_edge(
        *edge, positiv_correlation=original_graph.edges[edge]["positiv_correlation"]
    )
    return G_copy


def is_subgraph(
    G_sub: nx.Graph,
    G: nx.Graph,
):
    if not np.array_equal(
        np.intersect1d(list(G_sub.nodes), list(G.nodes)), list(G_sub.nodes)
    ):
        return False
    if not np.array_equal(
        get_shared_edges(G_sub, G), sort_items_universally(list(G_sub.edges))
    ):
        return False
    return True


def find_similar_subgraphs(G1: nx.Graph, G2: nx.Graph, n=-1):
    # helping variables
    g_structures = [create_graph_from([v], []) for v in get_shared_nodes(G1, G2)]
    new_g_structures = g_structures

    while len(new_g_structures) != 0:
        new_structs = []
        logging.info(f"New structures from iteration: {len(new_g_structures)}")
        for g_expand in new_g_structures:
            available_expanded_edges = find_suitable_edges(G1, G2, g_expand)
            for e in available_expanded_edges:
                g_expanded_tilde = expand_graph_by_edge(g_expand, e, G1)
                if (
                    is_subgraph(g_expanded_tilde, G1)
                    and is_subgraph(g_expanded_tilde, G2)
                    and not contains_graph(g_expanded_tilde, new_structs)
                ):
                    if (
                        G1.edges[e]["positiv_correlation"]
                        == G2.edges[e]["positiv_correlation"]
                    ):
                        new_structs.append(g_expanded_tilde)
        new_g_structures = []
        for g in new_structs:
            new_g_structures.append(g)
            g_structures.append(g)

    if n != -1:
        return g_structures[-n:]
    else:
        return g_structures


def get_graph_metrics(G: nx.Graph, phylums: np.ndarray) -> dict:
    # Calculate basic metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees) if degrees else 0

    degree_centrality = nx.degree_centrality(G)
    mean_degree_centrality = (
        np.mean(list(degree_centrality.values())) if degree_centrality else 0
    )

    betweenness_centrality = nx.betweenness_centrality(G)
    mean_betweenness_centrality = (
        np.mean(list(betweenness_centrality.values())) if betweenness_centrality else 0
    )

    phylum_count = pd.DataFrame(
        pd.DataFrame(data=list(G.nodes.data()))[1]
        .map(lambda it: it["Phylum"])
        .fillna(0)
        .value_counts()
    )
    for phylum in phylums:
        if phylum not in phylum_count.index:
            phylum_count.loc[phylum] = 0

    phylum_distribution = phylum_count.apply(lambda x: x / len(G.nodes.data()))
    phylum_diversity = entropy(phylum_distribution, base=len(phylums))[0]

    if nx.is_connected(G):
        diameter = nx.diameter(G)
        avg_shortest_path = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        diameter = nx.diameter(subgraph)
        avg_shortest_path = nx.average_shortest_path_length(subgraph)

    avg_clustering = nx.average_clustering(G)

    return {
        "Nodes": num_nodes,
        "Edges": num_edges,
        "Average Degree": avg_degree,
        "(Mean) Degree Centrality": mean_degree_centrality,
        "(Mean) Betweenness Centrality": mean_betweenness_centrality,
        "Diameter": diameter,
        "Average Shortest Path Length": avg_shortest_path,
        "Average Clustering Coefficient": avg_clustering,
        "Phylum Diversity (Entropy)": phylum_diversity,
        "Phylum Distribution": phylum_count.to_dict()["count"],
    }


def compare_graphs(
    graphs: list[nx.Graph], names: list[str], phylums: np.ndarray, out: str | None
) -> pd.DataFrame:
    logging.info("Compare Graphs")
    res = pd.DataFrame(
        [get_graph_metrics(g, phylums) for g in graphs],
        index=["graph_" + t for t in names],
    )
    if out is not None:
        res.to_csv(out)
    return res


def compare_graphs_pairwise_on(
    graphs: list[nx.Graph],
    names: list[str],
    metric: str = "nodes_iou",
    out: str = None,
    normalized: bool = False,
):
    res = pd.DataFrame(columns=names, index=names)
    for i, _ in res.iterrows():
        for j, _ in res.iterrows():
            numeric_i = list(names).index(i)
            numeric_j = list(names).index(j)
            if metric == "nodes_iou":
                nodes_gi = list(graphs[numeric_i].nodes)
                nodes_gj = list(graphs[numeric_j].nodes)
                res.at[i, j] = calc_iou(nodes_gi, nodes_gj)
            elif metric == "edges_iou":
                # convert into list of sorted edges and then join for comparision in iou!
                edges_gi = [
                    "|".join(sorted(item)) for item in list(graphs[numeric_i].edges)
                ]
                edges_gj = [
                    "|".join(sorted(item)) for item in list(graphs[numeric_j].edges)
                ]
                res.at[i, j] = calc_iou(edges_gi, edges_gj)
            elif metric == "edit_distance":
                # metric to compare the graphs. The value is the minimal edits needed (edges/nodes) so the graphs would be isometric
                distance = -1
                for edits in nx.optimal_edit_paths(
                    graphs[numeric_i], graphs[numeric_j]
                ):
                    distance = edits
                res.at[i, j] = distance
            elif metric == "kernel_shortest_path":
                res.at[i, j] = kernel_graph(
                    [graphs[numeric_i], graphs[numeric_j]],
                    "shortest_path",
                    normalize=normalized,
                )[1, 0]
            # elif(metric == "kernel_random_walk"):
            #    res.at[i,j] = kernel_graph([graphs[numeric_i], graphs[numeric_j]], "random_walk")[1,0]
            elif metric == "kernel_weisfeiler_lehman":
                res.at[i, j] = kernel_graph(
                    [graphs[numeric_i], graphs[numeric_j]],
                    "weisfeiler_lehman",
                    normalize=normalized,
                )[1, 0]
    if out:
        res.to_csv(out.replace(".png", ".csv"))
        f = plt.figure(figsize=(12, 10))
        df = res.astype(float)
        im = plt.matshow(df, fignum=f.number, cmap="Blues")
        plt.xticks(
            range(df.select_dtypes(["number"]).shape[1]),
            df.select_dtypes(["number"]).columns,
            fontsize=14,
            rotation=45,
        )
        plt.yticks(
            range(df.select_dtypes(["number"]).shape[1]),
            df.select_dtypes(["number"]).columns,
            fontsize=14,
        )
        for (i, j), z in np.ndenumerate(df):
            plt.text(
                j,
                i,
                "{:0.2f}".format(z),
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.4),
            )
        cb = plt.colorbar(im)
        cb.ax.tick_params(labelsize=14)
        plt.title(f"Matrix ({metric})", fontsize=16)
        plt.savefig(out)


def full_multiple_graphs_evaluation(
    raw_data: pd.DataFrame,
    transformed_data: list,
    seperator: np.ndarray,
    phylums: np.ndarray,
    lookup_data: pd.DataFrame,
):
    # create graphs and figures per data
    graphs: list[nx.Graph] = []
    graph_creator = get_graph_creator()
    for data in transformed_data:
        preprossed_data, lookup_data, relative_data = preprocessing(data, lookup_data)
        graph = graph_creator.create_network(
            preprossed_data, lookup_data, relative_data
        )
        graphs.append(graph)

    visualize_graphs(graphs, raw_data, seperator, phylums)

    compare_graphs(
        graphs=graphs, names=seperator, phylums=phylums, out="out/results.csv"
    )
    compare_graphs_pairwise_on(
        graphs=graphs,
        names=seperator,
        metric="nodes_iou",
        out="out/nodes_iou_matrics.png",
    )
    compare_graphs_pairwise_on(
        graphs=graphs,
        names=seperator,
        metric="edges_iou",
        out="out/edges_iou_matrics.png",
    )
    compare_graphs_pairwise_on(
        graphs=graphs,
        names=seperator,
        metric="kernel_shortest_path",
        out="out/kernel_shortest_path_norm.png",
        normalized=True,
    )
    compare_graphs_pairwise_on(
        graphs=graphs,
        names=seperator,
        metric="kernel_weisfeiler_lehman",
        out="out/kernel_weisfeiler_lehman_norm.png",
        normalized=True,
    )
    compare_graphs_pairwise_on(
        graphs=graphs,
        names=seperator,
        metric="kernel_shortest_path",
        out="out/kernel_shortest_path.png",
        normalized=False,
    )
    compare_graphs_pairwise_on(
        graphs=graphs,
        names=seperator,
        metric="kernel_weisfeiler_lehman",
        out="out/kernel_weisfeiler_lehman.png",
        normalized=False,
    )
    # compare_graphs_pairwise_on(graphs=graphs, names=seperator, metric="edit_distance", out="out/edit_distance_matrics.png") # this is very expensive to calculate!

    for i in range(len(graphs)):
        for j in range(len(graphs)):
            if i > j:
                found_patterns = find_similar_subgraphs(
                    G1=graphs[i], G2=graphs[j], n=50
                )
                for pattern, k in zip(found_patterns, range(len(found_patterns))):
                    create_figure_simple(pattern)
                    output_dir = f"out/patterns/graph_{i}__{j}"
                    os.makedirs(output_dir, exist_ok=True)
                    plt.savefig(f"{output_dir}/{len(found_patterns) - k}.svg")
                    plt.close()
