from grakel import ShortestPath, WeisfeilerLehman

from _preparation import relative_abundances, mclr
from _utils import calc_iou

from graph.comparison.kernels import graph_kernel
from graph.creation.registry import get_graph_creator
from graph.settings import USE_MCLR, MCLR_C


def cooccurrence(
    type_label, file_name, years=None, habitats=None, beneficials=None, crops=None
):
    _, df_rel_taxa_abundances, _, _ = relative_abundances(
        type_label, years, habitats, beneficials, crops
    )

    if USE_MCLR:
        df_rel_taxa_abundances = mclr(df_rel_taxa_abundances, c=MCLR_C)

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
