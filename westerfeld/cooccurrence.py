import pandas as pd

from grakel import ShortestPath, WeisfeilerLehman

from _preparation import rarefied_taxa_table, mclr
from _utils import calc_iou

from graph.comparison.kernels import graph_kernel
from graph.creation.registry import get_graph_creator
from graph.settings import USE_MCLR, MCLR_C, BLOCK_SCALE


def _scale_block(df, mode):
    """Apply per-kingdom block scaling. See `BLOCK_SCALE` in `graph.settings`."""
    if mode in (None, "none"):
        return df
    if mode == "zscore":
        return (df - df.mean()) / df.std(ddof=0).replace(0, 1)
    if mode == "center":
        return df - df.mean()
    if mode == "block":
        scale = df.std(ddof=0).mean() or 1
        return df / scale
    raise ValueError(f"Unknown BLOCK_SCALE value: {mode}")


def cooccurrence(
    kingdoms, file_name, years=None, habitats=None, beneficials=None, crops=None
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
        if USE_MCLR:
            df_rel = mclr(df_rel, c=MCLR_C)
        df_rel = _scale_block(df_rel, BLOCK_SCALE)
        df_rel.columns = [f"{type_label}:{taxon}" for taxon in df_rel.columns]
        kingdom_frames.append(df_rel)
    df_combined = pd.concat(kingdom_frames, axis=1, join="inner")

    graph_creator = get_graph_creator()
    return graph_creator.create_network(df_combined)


def main():
    print("-----------------")
    print("| CO-OCCURRENCE |")
    print("-----------------")

    kingdoms = {"Fungi": "Genus", "Bacteria": "Genus"}
    crops = ["Winter wheat 1", "Winter wheat 2"]

    graph_1 = cooccurrence(
        kingdoms,
        "cooccurrence--",
        years=2019,
        habitats="Field_Soil",
        crops=crops,
    )

    graph_2 = cooccurrence(
        kingdoms,
        "cooccurrence--",
        years=2019,
        habitats="Rhizosphere",
        crops=crops,
    )

    print(graph_1)
    print(graph_2)

    nodes_gi = list(graph_1.nodes)
    nodes_gj = list(graph_2.nodes)
    edges_gi = ["|".join(sorted(edge)) for edge in list(graph_1.edges)]
    edges_gj = ["|".join(sorted(edge)) for edge in list(graph_2.edges)]

    print(calc_iou(nodes_gi, nodes_gj))
    print(calc_iou(edges_gi, edges_gj))

    # This is expensive
    # distance = -1
    # for edits in nx.optimal_edit_paths(graph_1, graph_2):
    #     print(edits)
    #     distance = edits
    #     # TODO: Why should it always be the last element?
    # print(distance)

    k = graph_kernel(
        [graph_1, graph_2],
        ShortestPath(normalize=True),
    )  # [1, 0]
    print(k)

    k = graph_kernel(
        [graph_1, graph_2],
        WeisfeilerLehman(normalize=True),
    )  # [1, 0]
    print(k)


if __name__ == "__main__":
    main()
