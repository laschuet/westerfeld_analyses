# imports
import math
import pandas as pd
import matplotlib.pyplot as plt
import logging

from graph.settings import INPUT_FILE, LOOKUP_FILE, USE_MCLR, MCLR_C
from graph.creation.registry import get_graph_creator
from graph.utils import read_data
from graph.utils import preprocessing
from graph.utils import create_figure
from graph.comparison.utils import full_multiple_graphs_evaluation

from scipy.stats import gmean

from _preparation import common_preparation, rarify
from _utils import (
    EXPERIMENT_COLUMNS,
    pivot,
    taxonomy_level,
)


pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)


def example_single_graph():
    # create one graph example
    preprossed_data, lookup_data, relative_data = preprocessing(INPUT_FILE, LOOKUP_FILE)
    graph_creator = get_graph_creator()
    graph = graph_creator.create_network(preprossed_data, lookup_data, relative_data)
    create_figure(graph)
    plt.savefig("out.svg")


def example_treatment_seperation():
    # create two graphs example
    raw_data, lookup_data = read_data(INPUT_FILE, LOOKUP_FILE)

    # seperate original data by 'Treatment'
    treatments = raw_data["Treatment"].unique()
    phylums = raw_data["Phylum"].unique()
    seperated_samples = [
        raw_data[raw_data["Treatment"] == treatment] for treatment in treatments
    ]

    # add split column
    for samples in seperated_samples:
        samples["Crop_Treatment_Replicate"] = (
            "Crop_"
            + samples["Crop"]
            + "_Treatment"
            + samples["Treatment"]
            + "_Replicate"
            + samples["Replicate"]
        )
    transformed_data = [
        samples.pivot_table(
            index="Species",
            columns="Crop_Treatment_Replicate",
            values="Value",
            aggfunc="sum",
        )
        for samples in seperated_samples
    ]

    full_multiple_graphs_evaluation(
        raw_data=raw_data,
        transformed_data=transformed_data,
        seperator=treatments,
        phylums=phylums,
        lookup_data=lookup_data,
    )


def example_replica_seperation():
    # create two graphs example
    raw_data, lookup_data = read_data(INPUT_FILE, LOOKUP_FILE)

    # seperate original data by 'Replicate' and on treatment at index 0
    replicates = raw_data["Replicate"].unique()
    treatments = raw_data["Treatment"].unique()
    seperated_samples = [raw_data[(raw_data["Replicate"] == rep)] for rep in replicates]

    # add split column
    for samples in seperated_samples:
        samples["Crop_Treatment_Replicate"] = (
            "Crop_"
            + samples["Crop"]
            + "_Treatment"
            + samples["Treatment"]
            + "_Replicate"
            + samples["Replicate"]
        )
    transformed_data = [
        samples.pivot_table(
            index="Species",
            columns="Crop_Treatment_Replicate",
            values="Value",
            aggfunc="sum",
        )
        for samples in seperated_samples
    ]
    full_multiple_graphs_evaluation(
        raw_data=raw_data,
        transformed_data=transformed_data,
        seperator=replicates,
        lookup_data=lookup_data,
    )


def example_two_graphs():
    # create two graphs example
    raw_data_1_F, lookup_data = read_data("../df_Fungi_19_F_Co_.csv", LOOKUP_FILE)
    raw_data_1_R, _ = read_data("../df_Fungi_19_R_Co_.csv", LOOKUP_FILE)
    raw_data = pd.concat([raw_data_1_F, raw_data_1_R])

    # seperate original data by 'Treatment'
    habitats = raw_data["Habitat"].unique()
    phylums = raw_data["Phylum"].unique()
    seperated_samples = [
        raw_data[raw_data["Habitat"] == habitat] for habitat in habitats
    ]

    # add split column
    for samples in seperated_samples:
        samples["Crop_Treatment_Replicate"] = (
            "Crop_"
            + samples["Crop"]
            + "_Treatment"
            + samples["Treatment"]
            + "_Replicate"
            + samples["Replicate"]
        )
    transformed_data = [
        samples.pivot_table(
            index="Species",
            columns="Crop_Treatment_Replicate",
            values="Value_abs",
            aggfunc="sum",
        )
        for samples in seperated_samples
    ]

    transformed_data[0].to_csv("out/transformed_1.csv")
    transformed_data[1].to_csv("out/transformed_2.csv")

    full_multiple_graphs_evaluation(
        raw_data=raw_data,
        transformed_data=transformed_data,
        seperator=habitats,
        phylums=phylums,
        lookup_data=lookup_data,
    )


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


if __name__ == "__main__":
    main()
