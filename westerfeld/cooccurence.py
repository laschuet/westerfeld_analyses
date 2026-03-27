# imports
import pandas as pd
import matplotlib.pyplot as plt
import logging

from graph.settings import INPUT_FILE, LOOKUP_FILE
from graph.creation.registry import get_graph_creator
from graph.utils import read_data
from graph.utils import preprocessing
from graph.utils import create_figure
from graph.comparison.utils import full_multiple_graphs_evaluation

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
    raw_data, lookup_data = read_data("data/df_fungi.csv", LOOKUP_FILE)

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
    raw_data, lookup_data = read_data("data/df_fungi.csv", LOOKUP_FILE)

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
    raw_data_1_F, lookup_data = read_data("data/df_Fungi_19_F_Co_.csv", LOOKUP_FILE)
    raw_data_1_R, _ = read_data("data/df_Fungi_19_R_Co_.csv", LOOKUP_FILE)
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


if __name__ == "__main__":
    example_two_graphs()
