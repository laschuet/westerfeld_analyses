import math

from grakel import ShortestPath, WeisfeilerLehman
from scipy.stats import gmean

from _preparation import common_preparation, rarify
from _utils import (
    EXPERIMENT_COLUMNS,
    pivot,
    taxonomy_level,
)

from graph.settings import USE_MCLR, MCLR_C
from graph.creation.registry import get_graph_creator
from graph.utils import calc_iou
from graph.comparison.kernels import graph_kernel


def cooccurrence(
    type_label, file_name, years=None, habitats=None, beneficials=None, crops=None
):
    df = common_preparation(type_label, years, habitats, beneficials, crops)

    index_grouper = EXPERIMENT_COLUMNS

    print("Get community size...", end="")
    df_abs_total_abundances = pivot(df, "Value_abs", index_grouper)
    community_size = df_abs_total_abundances["Value_abs"].min()
    print("DONE")
    print(f"Community size: {community_size}")

    print("Calculate absolute abundances per taxa...", end="")
    columns_grouper = taxonomy_level(type_label)
    df_abs_taxa_abundances = pivot(df, "Value_abs", index_grouper, columns_grouper)
    print("DONE")

    print("Perform normalization...", end="")
    df_abs_taxa_abundances = rarify(df_abs_taxa_abundances)
    print("DONE")

    print("Calculate relative abundances...", end="")
    df_rel_taxa_abundances = df_abs_taxa_abundances.div(
        df_abs_taxa_abundances.sum(axis=1), axis=0
    ).fillna(0)
    print("DONE")

    if USE_MCLR:
        # transform via mCLR
        for index, row in df_rel_taxa_abundances.iterrows():
            non_zero_elements = row[row > 0]
            geometric_mean = gmean(non_zero_elements)
            row = row.apply(lambda x: (math.log10(x / geometric_mean)) if x != 0 else 0)
            min_result = row[row != 0].min()
            epsilon = abs(min_result) + MCLR_C
            row = row.apply(
                lambda x: (x + epsilon) if x != 0 else 0,
            )
            df_rel_taxa_abundances.loc[index] = row

    graph_creator = get_graph_creator()
    return graph_creator.create_network(df_rel_taxa_abundances)


def main():
    print("-----------------")
    print("| CO-OCCURRENCE |")
    print("-----------------")

    graph_1 = cooccurrence(
        "Fungi",
        "cooccurrence--",
        years=2019,
        habitats="Field_Soil",
        beneficials="Control",
        crops=["Grain maize", "Winter wheat 1", "Winter wheat 2"],
    )

    graph_2 = cooccurrence(
        "Fungi",
        "cooccurrence--",
        years=2019,
        habitats="Rhizosphere",
        beneficials="Control",
        crops=["Grain maize", "Winter wheat 1", "Winter wheat 2"],
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
