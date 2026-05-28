import pandas as pd

from _preparation import rarefied_taxa_table, mclr

from graph.comparison.utils import (
    common_subgraph,
    compare_graph_metrics,
    compare_graphs_pairwise,
    is_subgraph,
)
from graph.creation.correlation import CorrelationGraph
from graph.creation.inference import GlassoGraph


def _scale_block(df, mode):
    """Per-kingdom block scaling for the multi-kingdom merge.

    Modes:
      "none"   - off
      "zscore" - per-column standardisation (mean 0, std 1)
      "center" - per-column centring (mean 0, variances preserved)
      "block"  - divide each column by the kingdom's average std
                 (between-kingdom variances equalised, within-kingdom ratios preserved)
    """
    if mode in (None, "none"):
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
    mclr_c=1,
    block_scale="none",
):
    # `kingdoms` maps each kingdom to the taxonomy level to
    # aggregate it at, e.g. `{"Fungi": "Species", "Bacteria": "Genus"}`.
    # Per-kingdom: rarefy -> relative abundance -> (m)CLR within its own
    # composition (so the CLR's geometric-mean reference stays coherent and the
    # different sequencing depths don't contaminate each other), optionally
    # block-scale, then inner-join the per-kingdom frames on the sample axis.
    # Columns are prefixed with the kingdom (e.g. "Fungi:species_x") so each
    # node's origin is explicit in the resulting graph.
    kingdom_frames = []
    for type_label, taxonomy in kingdoms.items():
        df_abs = rarefied_taxa_table(
            type_label, taxonomy, years, habitats, beneficials, crops
        )
        df_rel = df_abs.div(df_abs.sum(axis=1), axis=0).fillna(0)
        if use_mclr:
            df_rel = mclr(df_rel, c=mclr_c)
        df_rel = _scale_block(df_rel, block_scale)
        df_rel.columns = [f"{type_label}:{taxon}" for taxon in df_rel.columns]
        kingdom_frames.append(df_rel)
    df_combined = pd.concat(kingdom_frames, axis=1, join="inner")

    return graph_creator.create_network(df_combined)


def main():
    print("-----------------")
    print("| CO-OCCURRENCE |")
    print("-----------------")

    kingdoms = {"Fungi": "Genus", "Bacteria": "Genus"}
    crops = ["Winter wheat 1", "Winter wheat 2"]
    graph_creator = CorrelationGraph()
    # graph_creator = GlassoGraph()

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

    for metric in (
        "nodes_iou",
        "edges_iou",
        "kernel_shortest_path",
        "kernel_weisfeiler_lehman",
    ):
        print(f"Pairwise {metric}")
        print(compare_graphs_pairwise(graphs, labels, metric))

    cs = common_subgraph(graph_1, graph_2)
    print(
        f"\nCommon subgraph: {cs.number_of_nodes()} nodes, "
        f"{cs.number_of_edges()} edges"
    )
    print(f"Graph_1 is subgraph of graph_2: {is_subgraph(graph_1, graph_2)}")
    print(f"Graph_2 is subgraph of graph_1: {is_subgraph(graph_2, graph_1)}")


if __name__ == "__main__":
    main()
