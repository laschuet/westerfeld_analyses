import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.lines import Line2D
from itertools import count

import numpy as np
from scipy.stats import gmean

from graph.settings import (
    ABSOLUTE_THRESHOLD,
    B_VALUE_GENERALIST_THRESHOLD,
    B_VALUE_SPECIALIST_THRESHOLD,
    CONVERT_FROM_ABSOLUTE_TO_RELATIVE,
    MCLR_C,
    MEAN_RELATIVE_ABUNDANCES_LOWER_THRESHOLD,
    NODE_NAME,
    USE_ABSOLUTE_THRESHOLD_TO_OBTAIN_COMMON_ASVS,
    USE_MCLR,
    USE_ZERO_RATION_TO_OBTAIN_COMMON_ASVS,
    ZERO_RATION_THRESHOLD,
)

import pandas as pd
import logging
import math

from pathlib import Path
from typing import Literal, Optional, Tuple


def read_data(
    input: str | pd.DataFrame, lookup: str | pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Path("out").mkdir(parents=True, exist_ok=True)
    logging.info("Import input file/load dataframe")
    # check if filepath or a dataframe directly
    if type(input) is not pd.DataFrame:
        df = pd.read_csv(input)
        df = df.set_index(NODE_NAME)
    else:
        df = input
    if type(lookup) is not pd.DataFrame:
        look_up_frame = pd.read_csv(lookup)
        look_up_frame = look_up_frame[
            [
                "Species",
                "Kingdom",
                "Phylum",
                "Class",
                "Order",
                "Family",
                "Genus",
                "Trophic_Mode",
                "Guild",
                "Confidence_Ranking",
                "Growth_Morphology",
                "Trait",
            ]
        ].drop_duplicates()
        look_up_frame = look_up_frame.set_index(NODE_NAME)
    else:
        look_up_frame = lookup
    return df, look_up_frame


def preprocessing(
    input: str | pd.DataFrame, lookup: str | pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # read in file and preprocess data
    df, look_up_frame = read_data(input, lookup)

    before_reduction_amount = df.shape[0]

    if USE_ABSOLUTE_THRESHOLD_TO_OBTAIN_COMMON_ASVS:
        logging.info(
            f"Remove all avs with total abundances less than threshold of {ABSOLUTE_THRESHOLD}."
        )
        shape_before = df.shape
        # transpose, so we can itterate over rows
        for index, row in df.iterrows():
            abundances_sum = row.sum()
            if abundances_sum < ABSOLUTE_THRESHOLD:
                df = df.drop(index=index)
        logging.info(
            f"Removed: {shape_before[0] - df.shape[0]} asvs! ({shape_before[0]} -> {df.shape[0]}); removed {((shape_before[0] - df.shape[0]) / shape_before[0]):.3f}%!"
        )

    if USE_ZERO_RATION_TO_OBTAIN_COMMON_ASVS:
        logging.info(
            f"Remove all avs that have a zero abundance ration above {ZERO_RATION_THRESHOLD}."
        )
        shape_before = df.shape
        # transpose, so we can itterate over rows
        for index, row in df.iterrows():
            zero_ratio = (len(row) - row.count()) / len(row)
            if zero_ratio > ZERO_RATION_THRESHOLD:
                df = df.drop(index=index)
        logging.info(
            f"Removed: {shape_before[0] - df.shape[0]} asvs! ({shape_before[0]} -> {df.shape[0]}); removed {((shape_before[0] - df.shape[0]) / shape_before[0]):.3f}%!"
        )

    logging.info(
        f"OVERALL: Removed {before_reduction_amount - df.shape[0]} asvs! ({before_reduction_amount} -> {df.shape[0]}); removed {((before_reduction_amount - df.shape[0]) / before_reduction_amount):.3f}%!"
    )

    # Transpose so samples are rows and species are columns
    df_t = df.T

    # fill nans with zeros (just in case)
    df_t.fillna(0, inplace=True)

    relative_df = df_t.copy(deep=True)
    logging.info("Convert absolute abundances to relative abundances")
    for index, row in relative_df.iterrows():
        abundances_sum = row.sum()
        relative_df.loc[index] = relative_df.loc[index].apply(
            lambda x: x / abundances_sum if abundances_sum != 0 else 0
        )

    if CONVERT_FROM_ABSOLUTE_TO_RELATIVE:
        df_t = relative_df.copy(deep=True)

    if USE_MCLR:
        # transform via mCLR
        for index, row in df_t.iterrows():
            non_zero_elements = row[row > 0]
            geometric_mean = gmean(non_zero_elements)
            row = row.apply(lambda x: (math.log10(x / geometric_mean)) if x != 0 else 0)
            min_result = row[row != 0].min()
            epsilon = abs(min_result) + MCLR_C
            row = row.apply(
                lambda x: (x + epsilon) if x != 0 else 0,
            )
            df_t.loc[index] = row
    return df_t, look_up_frame, relative_df


def calc_iou(d1, d2):
    union = set(d1 + d2)
    sect = np.intersect1d(d1, d2)
    return len(sect) / len(union)


def identifiy_generalists_or_specialists(
    Pj: np.ndarray,
) -> Optional[Literal["Specialist", "Generalist"]]:
    """
    implementation of the niche breadth approach. described in https://www.researchgate.net/publication/309922017_The_importance_of_neutral_and_niche_processes_for_bacterial_community_assembly_differs_between_habitat_generalists_and_specialists?enrichId=rgreq-e53157fd5b364b94816e907d1105b272-XXX&enrichSource=Y292ZXJQYWdlOzMwOTkyMjAxNztBUzo4NTY0NDI0NzkzMjUxODlAMTU4MTIwMzIwNzYwNA%3D%3D&el=1_x_2&_esc=publicationCoverPdf (doi not correct??)
    """
    Pj = Pj / Pj.sum()
    mean_average_relative_abudances = Pj.mean()
    if mean_average_relative_abudances < MEAN_RELATIVE_ABUNDANCES_LOWER_THRESHOLD:
        return None
    Bj = 1 / (Pj**2).sum()
    # 1 / np.sum(Pj**2, axis=0)  #1 / (Pj ** 2).sum() (alternative)
    if Bj > B_VALUE_GENERALIST_THRESHOLD:
        return "Generalist", mean_average_relative_abudances, Bj
    elif Bj < B_VALUE_SPECIALIST_THRESHOLD:
        return "Specialist", mean_average_relative_abudances, Bj
    return None, mean_average_relative_abudances, Bj


def create_figure(
    G: nx.Graph,
    figure_number: int = 1,
    ax=None,
    save_file: str = "out.svg",
    mapping=None,
    visualizing_attr="Phylum",
):
    colormap = plt.cm.jet
    edge_colors = [
        "green" if G.edges[e].get("positiv_correlation", False) else "red"
        for e in G.edges()
    ]

    if mapping == None:
        groups = set(nx.get_node_attributes(G, visualizing_attr).values())
        mapping = dict(zip(sorted(groups), count()))

    colors = [mapping[G.nodes[n][visualizing_attr]] for n in G.nodes()]
    edge_colors = [1 if G.edges[e]["positiv_correlation"] else 0 for e in G.edges]

    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(
        G, pos, node_size=20, node_color=colors, node_shape="^", cmap=colormap, ax=ax
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax, edge_color=edge_colors)
    if ax:
        ax.axis("on")
        fig = ax.get_figure()
    else:
        plt.axis("off")
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="^",
                color=color,
                label=phylum,
                lw=0,
                markerfacecolor=color,
                markersize=10,
            )
            for phylum, color in zip(
                list(set(nx.get_node_attributes(G, "Phylum").values())),
                list(
                    map(
                        lambda x: colormap(x / len(list(groups))),
                        range(len(list(groups))),
                    )
                ),
            )
        ]
        ax = plt.gca()
        ax.legend(handles=legend_elements, loc="upper right")


def create_figure_simple(G: nx.Graph):
    plt.figure(figsize=(3, 3))
    edge_colors = [
        "green" if G.edges[e].get("positiv_correlation", False) else "red"
        for e in G.edges()
    ]
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_shape="^")
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color=edge_colors)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes()}, font_size=4)
    plt.axis("off")  # Turn off axis


def visualize_graphs(
    graphs,
    raw_data: pd.DataFrame,
    seperator: np.ndarray,
    phylums: np.ndarray,
):
    n_sep = len(seperator)

    if n_sep == 2:
        fig, axs = plt.subplots(1, 2, figsize=(1.35 * 5 * 2, 5))
        axs_coordinates = [(0, 0), (0, 1)]
    elif n_sep == 4:
        fig, axs = plt.subplots(2, 2, figsize=(1.35 * 5 * 2, 5 * 2))
        axs_coordinates = [(0, 0), (0, 1), (1, 0), (1, 1)]
    axs = np.array(axs).reshape(-1)

    # prepare legend elements
    colormap = plt.cm.jet
    groups = raw_data["Phylum"].unique()
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="^",
            color=colormap(i / len(groups)),
            label=phylum,
            lw=0,
            markerfacecolor=colormap(i / len(groups)),
            markersize=10,
        )
        for i, phylum in enumerate(sorted(groups))
    ]
    mapping = dict(zip(sorted(groups), count()))

    # visualize by phylum
    for graph, ax, sep in zip(graphs, axs, seperator):
        ax.set_title(f"{sep}")
        create_figure(graph, ax=ax, save_file=f"graph_{sep}.svg", mapping=mapping)

    # global legend
    fig.legend(handles=legend_elements, loc="center left", ncol=1, frameon=False)
    plt.tight_layout(rect=[0.15, 0, 1, 1])  # leave space left for legend
    plt.savefig("out/graphs-by-phylum.svg")

    plt.close(fig)

    # --- Second figure ---
    if n_sep == 2:
        fig, axs = plt.subplots(1, 2, figsize=(1.35 * 5 * 2, 5))
        axs_coordinates = [(0, 0), (0, 1)]
    elif n_sep == 4:
        fig, axs = plt.subplots(2, 2, figsize=(1.35 * 5 * 2, 5 * 2))
        axs_coordinates = [(0, 0), (0, 1), (1, 0), (1, 1)]
    axs = np.array(axs).reshape(-1)

    # prepare legend elements
    colormap = plt.cm.jet
    groups = ["Generalist", "Specialist", "None"]
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="^",
            color=colormap(i / len(groups)),
            label=genspec,
            lw=0,
            markerfacecolor=colormap(i / len(groups)),
            markersize=10,
        )
        for i, genspec in enumerate(sorted(groups))
    ]
    mapping = dict(zip(sorted(groups), count()))

    # visualize by specialist/generalists
    for graph, ax, sep in zip(graphs, axs, seperator):
        ax.set_title(f"{sep}")
        create_figure(
            graph,
            ax=ax,
            save_file=f"graph_{sep}.svg",
            mapping=mapping,
            visualizing_attr="generalist_or_specialists",
        )

    # global legend
    fig.legend(handles=legend_elements, loc="center left", ncol=1, frameon=False)
    plt.tight_layout(rect=[0.15, 0, 1, 1])  # leave space left for legend
    plt.savefig("out/graphs-by-specialist-or-generalists.svg")
