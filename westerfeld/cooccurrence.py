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
    compare_graphs_pairwise_node_type_iou,
    edge_kingdom_type,
    find_similar_subgraphs,
    graph_edge_type_summary,
    graph_metrics,
    graph_node_type_summary,
    plot_graphs_side_by_side,
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


def _sanitize_sheet_name(name):
    invalid = '[]:*?/\\'
    sanitized = ''.join('_' if ch in invalid else ch for ch in str(name))
    return sanitized[:31]


def _build_taxon_lookup(df_long, taxonomy, kingdom):
    taxon_names = (
        df_long[[taxonomy]]
        .drop_duplicates()
        .sort_values(by=taxonomy)
        [taxonomy]
        .astype(str)
    )
    lookup_index = [f"{kingdom}:{taxon}" for taxon in taxon_names]
    return pd.DataFrame(
        {
            "kingdom": kingdom,
            "taxon": taxon_names.values,
        },
        index=lookup_index,
    )


def plot_graphs_side_by_side_by_niche(
    graphs,
    labels,
    path="graph_side_by_side_niche.png",
    figsize=(14, 7),
    node_size=80,
    edge_width=1.0,
):
    import matplotlib.pyplot as plt
    import networkx as nx

    fig, axes = plt.subplots(1, len(graphs), figsize=figsize)
    if len(graphs) == 1:
        axes = [axes]

    classification_colors = {
        "Generalist": "#2ca02c",
        "Specialist": "#d62728",
        "None": "#7f7f7f",
    }

    for ax, G, label in zip(axes, graphs, labels):
        if G.number_of_nodes() == 0:
            ax.set_axis_off()
            continue

        pos = nx.spring_layout(G, seed=42)
        node_colors = [
            classification_colors.get(
                G.nodes[n].get("generalist_or_specialists", "None"),
                "#7f7f7f",
            )
            for n in G.nodes
        ]

        for edge_type in sorted({edge_kingdom_type(G, u, v) for u, v in G.edges()}):
            edges = [e for e in G.edges() if edge_kingdom_type(G, e[0], e[1]) == edge_type]
            if not edges:
                continue
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=edges,
                edge_color="#999999",
                width=edge_width,
                alpha=0.6,
                ax=ax,
            )

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=list(G.nodes),
            node_color=node_colors,
            node_size=node_size,
            ax=ax,
        )

        ax.set_title(label)
        ax.set_axis_on()
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=label)
        for label, color in (
            ("Generalist", classification_colors["Generalist"]),
            ("Specialist", classification_colors["Specialist"]),
            ("Unclassified", classification_colors["None"]),
        )
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def export_cooccurrence_results(path, graphs, labels):
    graph_metrics_df = pd.DataFrame([graph_metrics(graph) for graph in graphs], index=labels)

    pairwise_metrics = {
        metric: compare_graphs_pairwise(graphs, labels, metric)
        for metric in (
            "nodes_iou",
            "edges_iou",
            "kernel_shortest_path",
            "kernel_weisfeiler_lehman",
        )
    }
    edge_iou_sheets = {
        "Field_vs_Rhizo_Fungi-Fungi": compare_graphs_pairwise(
            graphs, labels, "edges_iou", pair_type="Fungi-Fungi"
        ),
        "Field_vs_Rhizo_Bacteria-Bacteria": compare_graphs_pairwise(
            graphs, labels, "edges_iou", pair_type="Bacteria-Bacteria"
        ),
        "Field_vs_Rhizo_Fungi-Bacteria": compare_graphs_pairwise(
            graphs, labels, "edges_iou", pair_type="Fungi-Bacteria"
        ),
    }
    node_iou_sheets = {
        "Field_vs_Rhizo_Fungi_nodes": compare_graphs_pairwise(
            graphs, labels, "nodes_iou", pair_type="Fungi"
        ),
        "Field_vs_Rhizo_Bacteria_nodes": compare_graphs_pairwise(
            graphs, labels, "nodes_iou", pair_type="Bacteria"
        ),
    }

    with pd.ExcelWriter(path) as writer:
        graph_metrics_df.to_excel(writer, sheet_name=_sanitize_sheet_name("Graph Metrics"))

        for graph, label in zip(graphs, labels):
            graph_edge_type_summary(graph).to_excel(
                writer,
                sheet_name=_sanitize_sheet_name(f"{label} Edge Summary"),
            )
            graph_node_type_summary(graph).to_excel(
                writer,
                sheet_name=_sanitize_sheet_name(f"{label} Node Summary"),
            )

            node_attrs = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
            if "generalist_or_specialists" in node_attrs.columns:
                node_attrs.to_excel(
                    writer,
                    sheet_name=_sanitize_sheet_name(f"{label} Niche Classification"),
                )
                niche_summary = (
                    node_attrs["generalist_or_specialists"]
                    .value_counts()
                    .rename_axis("classification")
                    .reset_index(name="count")
                )
                niche_summary.to_excel(
                    writer,
                    sheet_name=_sanitize_sheet_name(f"{label} Niche Summary"),
                    index=False,
                )

        for metric, df in pairwise_metrics.items():
            df.to_excel(writer, sheet_name=_sanitize_sheet_name(f"Pairwise {metric}"))

        for name, df in edge_iou_sheets.items():
            df.to_excel(writer, sheet_name=_sanitize_sheet_name(name))

        for name, df in node_iou_sheets.items():
            df.to_excel(writer, sheet_name=_sanitize_sheet_name(name))

        common_df = pd.DataFrame(
            [
                {
                    "common_nodes": common_subgraph(graphs[0], graphs[1]).number_of_nodes(),
                    "common_edges": common_subgraph(graphs[0], graphs[1]).number_of_edges(),
                    "graph_1_is_subgraph_of_graph_2": is_subgraph(graphs[0], graphs[1]),
                    "graph_2_is_subgraph_of_graph_1": is_subgraph(graphs[1], graphs[0]),
                }
            ]
        )
        common_df.to_excel(writer, sheet_name=_sanitize_sheet_name("Common Subgraph"), index=False)


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
    annotate_niche=False,
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
    kingdom_relative_frames = []
    lookup_frames = []
    for kingdom, taxonomy in kingdoms.items():
        df_long = common_preparation(kingdom, years, habitats, beneficials, crops)
        df_abs = rarefied_taxa_table(df_long, taxonomy)
        df_rel = relative_abundances(df_abs)
        if min_prevalence is not None:
            df_rel = filter_prevalence(df_rel, min_prevalence)

        df_rel_raw = df_rel.copy()
        if use_mclr:
            df_rel = mclr(df_rel, pseudocount=mclr_pseudocount)
        df_rel = _scale_block(df_rel, block_scale)

        prefix = f"{kingdom}:"
        rel_columns = [f"{prefix}{taxon}" for taxon in df_rel.columns]
        df_rel.columns = rel_columns
        df_rel_raw.columns = rel_columns

        kingdom_frames.append(df_rel)
        kingdom_relative_frames.append(df_rel_raw)
        if annotate_niche:
            lookup_frames.append(_build_taxon_lookup(df_long, taxonomy, kingdom))

    df_combined = pd.concat(kingdom_frames, axis=1, join="inner")
    df_relative = pd.concat(kingdom_relative_frames, axis=1, join="inner") if annotate_niche else None
    df_lookup = pd.concat(lookup_frames, axis=0) if annotate_niche else None

    return graph_creator.create_network(
        df_combined,
        df_lookup=df_lookup,
        df_relative=df_relative,
    )


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
        annotate_niche=True,
    )

    graph_2 = cooccurrence(
        kingdoms,
        graph_creator,
        years=2019,
        habitats="Rhizosphere",
        crops=crops,
        annotate_niche=True,
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

    plot_graphs_side_by_side(
        graphs,
        labels,
        path="graph_side_by_side.png",
        figsize=(14, 7),
        node_size=80,
        edge_width=1.0,
    )
    plot_graphs_side_by_side_by_niche(
        graphs,
        labels,
        path="graph_side_by_side_niche.png",
        figsize=(14, 7),
        node_size=80,
        edge_width=1.0,
    )

    cs = common_subgraph(graph_1, graph_2)
    print(
        f"\nCommon subgraph: {cs.number_of_nodes()} nodes, {cs.number_of_edges()} edges"
    )
    print(f"Graph_1 is subgraph of graph_2: {is_subgraph(graph_1, graph_2)}")
    print(f"Graph_2 is subgraph of graph_1: {is_subgraph(graph_2, graph_1)}")

    for graph, label in zip(graphs, labels):
        node_attrs = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient="index")
        if "generalist_or_specialists" in node_attrs.columns:
            print(f"\nNiche classification counts for {label}")
            print(node_attrs["generalist_or_specialists"].value_counts())

    print("\nExport Excel results to cooccurrence_results.xlsx")
    export_cooccurrence_results("cooccurrence_results.xlsx", graphs, labels)

    # Can be slow on dense graphs
    # similar = find_similar_subgraphs(graph_1, graph_2)
    # largest = max((s.number_of_edges() for s in similar), default=0)
    # print(
    #     f"\nFind_similar_subgraphs: {len(similar)} sign-matching substructures, "
    #     f"largest = {largest} edges"
    # )


if __name__ == "__main__":
    main()
