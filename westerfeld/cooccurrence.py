import pandas as pd

from _preparation import (
    common_preparation,
    filter_prevalence,
    mclr,
    rarefied_taxa_table,
    relative_abundances,
)

from graph.comparison import (
    common_subgraph,
    compare_graph_metrics,
    compare_graphs_pairwise,
    find_similar_subgraphs,
    graph_edge_type_summary,
    graph_node_type_summary,
    is_subgraph,
)
from graph.creation import CorrelationGraph, GlassoGraph


def _scale_block(df, mode):
    """
    Per-kingdom block scaling for the multi-kingdom merge.

    Modes:
      "none"   - off
      "zscore" - per-column standardization (mean 0, std 1)
      "center" - per-column centering (mean 0, variances preserved)
      "block"  - divide each column by the kingdom's average std
                 (between-kingdom variances equalized, within-kingdom ratios preserved)
    """
    if mode is None:
        return df
    if mode == "zscore":
        return (df - df.mean()) / df.std(ddof=0).replace(0, 1)
    if mode == "center":
        return df - df.mean()
    if mode == "block":
        scale = df.std(ddof=0).mean() or 1
        return df / scale
    raise ValueError(f"Unknown mode value: {mode}")


def cooccurrence(
    kingdoms,
    graph_creator,
    years=None,
    habitats=None,
    beneficials=None,
    crops=None,
    use_mclr=True,
    mclr_pseudocount=1,
    block_scale=None,
):
    # `kingdoms` maps each kingdom to the taxonomy level to
    # aggregate it at, e.g. `{"Fungi": "Species", "Bacteria": "Genus"}`.
    # Per-kingdom: rarefy -> relative abundance -> mCLR within its own
    # composition (so the mCLR's geometric-mean reference stays coherent and the
    # different sequencing depths don't contaminate each other), optionally
    # block-scale, then inner-join the per-kingdom frames on the sample axis.
    # Columns are prefixed with the kingdom (e.g. "Fungi:species_x") so each
    # node's origin is explicit in the resulting graph.
    # Some graph creators (e.g. Graphical Lasso) cannot handle far more taxa
    # than samples and ask for a prevalence filter via `min_prevalence`; others
    # (e.g. correlation) leave it None and keep every taxon.
    min_prevalence = getattr(graph_creator, "min_prevalence", None)

    kingdom_frames = []
    for kingdom, taxonomy in kingdoms.items():
        df_long = common_preparation(kingdom, years, habitats, beneficials, crops)
        df_abs = rarefied_taxa_table(df_long, taxonomy)
        df_rel = relative_abundances(df_abs)
        if min_prevalence is not None:
            df_rel = filter_prevalence(df_rel, min_prevalence)
        if use_mclr:
            df_rel = mclr(df_rel, pseudocount=mclr_pseudocount)
        df_rel = _scale_block(df_rel, block_scale)
        df_rel.columns = [f"{kingdom}:{taxon}" for taxon in df_rel.columns]
        kingdom_frames.append(df_rel)
    df_combined = pd.concat(kingdom_frames, axis=1, join="inner")

    return graph_creator.create_network(df_combined)


def main():
    print("-----------------")
    print("| CO-OCCURRENCE |")
    print("-----------------")

    kingdoms = {"Fungi": "Genus", "Bacteria": "Genus"}
    crops = ["Winter wheat 1", "Winter wheat 2"]
    # graph_creator = CorrelationGraph()
    graph_creator = GlassoGraph()

    graph_1 = cooccurrence(
        kingdoms,
        graph_creator,
        years=2019,
        habitats="Field_Soil",
        crops=crops,
    )

    graph_2 = cooccurrence(
        kingdoms,
        graph_creator,
        years=2019,
        habitats="Rhizosphere",
        crops=crops,
    )

    print(graph_1)
    print(graph_2)

    graphs = [graph_1, graph_2]
    labels = ["Field_Soil", "Rhizosphere"]

    print("\nPer-graph metrics")
    print(compare_graph_metrics(graphs, labels))

    for graph, label in zip(graphs, labels):
        print(f"\nEdge-type summary for {label}")
        print(graph_edge_type_summary(graph))
        print(f"\nNode-type (kingdom) summary for {label}")
        print(graph_node_type_summary(graph))

    for metric in (
        "nodes_iou",
        "edges_iou",
        "kernel_shortest_path",
        "kernel_weisfeiler_lehman",
    ):
        print(f"Pairwise {metric}")
        print(compare_graphs_pairwise(graphs, labels, metric))

    print("\nPairwise edges_iou for Fungi-Fungi edges")
    print(compare_graphs_pairwise(graphs, labels, "edges_iou", pair_type="Fungi-Fungi"))
    print("\nPairwise edges_iou for Bacteria-Bacteria edges")
    print(compare_graphs_pairwise(graphs, labels, "edges_iou", pair_type="Bacteria-Bacteria"))
    print("\nPairwise edges_iou for Fungi-Bacteria edges")
    print(compare_graphs_pairwise(graphs, labels, "edges_iou", pair_type="Fungi-Bacteria"))

    print("\nPairwise nodes_iou for Fungi nodes")
    print(compare_graphs_pairwise(graphs, labels, "nodes_iou", pair_type="Fungi"))
    print("\nPairwise nodes_iou for Bacteria nodes")
    print(compare_graphs_pairwise(graphs, labels, "nodes_iou", pair_type="Bacteria"))

    cs = common_subgraph(graph_1, graph_2)
    print(
        f"\nCommon subgraph: {cs.number_of_nodes()} nodes, {cs.number_of_edges()} edges"
    )
    print(f"Graph_1 is subgraph of graph_2: {is_subgraph(graph_1, graph_2)}")
    print(f"Graph_2 is subgraph of graph_1: {is_subgraph(graph_2, graph_1)}")

    # Can be slow on dense graphs
    # similar = find_similar_subgraphs(graph_1, graph_2)
    # largest = max((s.number_of_edges() for s in similar), default=0)
    # print(
    #     f"\nFind_similar_subgraphs: {len(similar)} sign-matching substructures, "
    #     f"largest = {largest} edges"
    # )


if __name__ == "__main__":
    main()
