import numpy as np
import pandas as pd

from _utils import (
    create_label,
    EXPERIMENT_COLUMNS,
    merge_and_unpivot,
    pivot,
    split,
    postprocess_label,
    resolve_filter,
    taxonomy_level,
)


def prepare_experiment(df):
    # Load CSV files
    df_plot = pd.read_csv("../lte_westerfeld.V1_0_PLOT.csv")
    df_experimental_setup = pd.read_csv("../lte_westerfeld.V1_0_EXPERIMENTAL_SETUP.csv")
    df_crop = pd.read_csv("../lte_westerfeld.V1_0_CROP.csv")
    df_treatment = pd.read_csv("../lte_westerfeld.V1_0_TREATMENT.csv")
    df_factor_1_level = pd.read_csv("../lte_westerfeld.V1_0_FACTOR_1_LEVEL.csv")
    df_factor_2_level = pd.read_csv("../lte_westerfeld.V1_0_FACTOR_2_LEVEL.csv")

    # Check if Crop_ID is already available
    if "Crop_ID" in df.columns:
        # If yes, join table PLOT by Treatment_ID
        df = pd.merge(
            df,
            df_plot[["Plot_ID", "Treatment_ID", "Block", "Replication"]],
            on="Plot_ID",
            how="left",
        )
    else:
        # If no, join table EXPERIMENTAL_SETUP by Crop_ID and Treatment_ID
        df = pd.merge(
            df,
            df_experimental_setup[
                ["Plot_ID", "Experimental_Year", "Crop_ID", "Treatment_ID"]
            ],
            on=["Experimental_Year", "Plot_ID"],
            how="left",
        )
        df = pd.merge(
            df, df_plot[["Plot_ID", "Block", "Replication"]], on="Plot_ID", how="left"
        )

    # Add CROP information
    # Differentiation between Winter wheat 1 and Winter wheat 2
    df_crop.loc[df_crop["Crop_ID"] == 5, "Name_EN"] = "Winter wheat 1"
    df_crop.loc[df_crop["Crop_ID"] == 4, "Name_EN"] = "Winter wheat 2"

    df = pd.merge(df, df_crop[["Crop_ID", "Name_EN"]], on="Crop_ID", how="left")
    df = df.rename(columns={"Name_EN": "Crop"})

    # Add TREATEMENT information for 'Factor_1_Level_ID' and 'Factor_2_Level_ID'
    df = pd.merge(
        df,
        df_treatment[["Treatment_ID", "Factor_1_Level_ID", "Factor_2_Level_ID"]],
        on="Treatment_ID",
        how="left",
    )

    # Add FACTOR_1_LEVEL information for Tillage
    df = pd.merge(
        df,
        df_factor_1_level[["Factor_1_Level_ID", "Name_EN"]],
        on="Factor_1_Level_ID",
        how="left",
    )
    df = df.rename(columns={"Name_EN": "Tillage"})

    # Add FACTOR_2_LEVEL information for Fertilization
    df = pd.merge(
        df,
        df_factor_2_level[["Factor_2_Level_ID", "Name_EN"]],
        on="Factor_2_Level_ID",
        how="left",
    )
    df = df.rename(columns={"Name_EN": "Fertilization"})

    # Drop merged identifier columns
    df = df.drop(
        columns=["Factor_2_Level_ID", "Factor_1_Level_ID", "Treatment_ID", "Crop_ID"]
    )

    # Rename Value in Value_abs
    df = df.rename(columns={"Value": "Value_abs"})

    return df


def prepare_taxonomy(df):
    # Load CSV files
    df_kingdom = pd.read_csv("../lte_westerfeld.V1_0_KINGDOM.csv")
    df_phylum = pd.read_csv("../lte_westerfeld.V1_0_PHYLUM.csv")
    df_class = pd.read_csv("../lte_westerfeld.V1_0_CLASS.csv")
    df_family = pd.read_csv("../lte_westerfeld.V1_0_FAMILY.csv")
    df_order = pd.read_csv("../lte_westerfeld.V1_0_ORDER.csv")
    df_genus = pd.read_csv("../lte_westerfeld.V1_0_GENUS.csv")
    df_species = pd.read_csv("../lte_westerfeld.V1_0_SPECIES.csv")

    # Add KINGDOM information
    df = pd.merge(df, df_kingdom[["Kingdom_ID", "Name"]], on="Kingdom_ID", how="left")
    df = df.rename(columns={"Name": "Kingdom"})

    # Add PHYLUM information
    df = pd.merge(df, df_phylum[["Phylum_ID", "Name"]], on="Phylum_ID", how="left")
    df = df.rename(columns={"Name": "Phylum"})

    # Add CLASS information
    df = pd.merge(df, df_class[["Class_ID", "Name"]], on="Class_ID", how="left")
    df = df.rename(columns={"Name": "Class"})

    # Add ORDER information
    df = pd.merge(df, df_order[["Order_ID", "Name"]], on="Order_ID", how="left")
    df = df.rename(columns={"Name": "Order"})

    # Add FAMILY information
    df = pd.merge(df, df_family[["Family_ID", "Name"]], on="Family_ID", how="left")
    df = df.rename(columns={"Name": "Family"})

    # Add GENUS information
    df = pd.merge(df, df_genus[["Genus_ID", "Name"]], on="Genus_ID", how="left")
    df = df.rename(columns={"Name": "Genus"})

    # Add SPECIES information
    df = pd.merge(df, df_species[["Species_ID", "Name"]], on="Species_ID", how="left")
    df = df.rename(columns={"Name": "Species"})

    # Drop merged identifier columns
    df = df.drop(
        columns=[
            "Species_ID",
            "Genus_ID",
            "Family_ID",
            "Order_ID",
            "Class_ID",
            "Phylum_ID",
            "Kingdom_ID",
        ]
    )

    return df


def prepare_fungi():
    # Load CSV files
    df_beneficial = pd.read_csv("../lte_westerfeld.V1_0_BENEFICIAL.csv")
    df_bioproject = pd.read_csv("../lte_westerfeld.V1_0_BIOPROJECT.csv")
    df_habitat = pd.read_csv("../lte_westerfeld.V1_0_HABITAT.csv")
    dtypes = {"Seq_ID": "str", "ACC_Num": "str"}
    df_fungi = pd.read_csv("../lte_westerfeld.V1_0_FUNGI.csv", dtype=dtypes)

    # Add BENEFICIAL information
    df_fungi = pd.merge(
        df_fungi,
        df_beneficial[["Beneficial_ID", "Name_EN"]],
        on="Beneficial_ID",
        how="left",
    )
    df_fungi = df_fungi.rename(columns={"Name_EN": "Beneficial"})

    # Add HABITAT information
    df_fungi = pd.merge(
        df_fungi, df_habitat[["Habitat_ID", "Name_EN"]], on="Habitat_ID", how="left"
    )
    df_fungi = df_fungi.rename(columns={"Name_EN": "Habitat"})

    # Add BIOPROJECT information
    df_fungi = pd.merge(
        df_fungi,
        df_bioproject[["BioProject_ID", "Name"]],
        on="BioProject_ID",
        how="left",
    )
    df_fungi = df_fungi.rename(columns={"Name": "BioProject"})

    # Drop merged identifier columns
    df_fungi = df_fungi.drop(columns=["Beneficial_ID", "Habitat_ID", "BioProject_ID"])

    # Add experiment information
    df_fungi = prepare_experiment(df_fungi)

    # Add taxonomy information
    df_fungi = prepare_taxonomy(df_fungi)

    return df_fungi


def prepare_bacteria():
    # Load CSV files
    df_beneficial = pd.read_csv("../lte_westerfeld.V1_0_BENEFICIAL.csv")
    df_bioproject = pd.read_csv("../lte_westerfeld.V1_0_BIOPROJECT.csv")
    df_habitat = pd.read_csv("../lte_westerfeld.V1_0_HABITAT.csv")
    dtypes = {"Seq_ID": "str", "ACC_Num": "str", "OTU_ID": "str"}
    df_bacteria = pd.read_csv("../lte_westerfeld.V1_0_BACTERIA.csv", dtype=dtypes)

    # Add BENEFICIAL information
    df_bacteria = pd.merge(
        df_bacteria,
        df_beneficial[["Beneficial_ID", "Name_EN"]],
        on="Beneficial_ID",
        how="left",
    )
    df_bacteria = df_bacteria.rename(columns={"Name_EN": "Beneficial"})

    # Add HABITAT information
    df_bacteria = pd.merge(
        df_bacteria, df_habitat[["Habitat_ID", "Name_EN"]], on="Habitat_ID", how="left"
    )
    df_bacteria = df_bacteria.rename(columns={"Name_EN": "Habitat"})

    # Add BIOPROJECT information
    df_bacteria = pd.merge(
        df_bacteria,
        df_bioproject[["BioProject_ID", "Name"]],
        on="BioProject_ID",
        how="left",
    )
    df_bacteria = df_bacteria.rename(columns={"Name": "BioProject"})

    # Drop merged identifier columns
    df_bacteria = df_bacteria.drop(
        columns=["Beneficial_ID", "Habitat_ID", "BioProject_ID"]
    )

    # Add experiment information
    df_bacteria = prepare_experiment(df_bacteria)

    # Add taxonomy information
    df_bacteria = prepare_taxonomy(df_bacteria)

    return df_bacteria


def treatment(row):
    if row["Tillage"] == "Cultivator":
        if row["Fertilization"] == "extensive":
            return "CT-EXT"
        else:
            return "CT-INT"
    else:
        if row["Fertilization"] == "extensive":
            return "MP-EXT"
        else:
            return "MP-INT"


def replicate(row):
    if row["Crop"] == "Grain maize":
        if row["Experimental_Year"] == 2019:
            mapping_2019 = {"A1": "Rep1", "A3": "Rep2", "B1": "Rep3", "B3": "Rep4"}
            return mapping_2019[row["Block"]]
        elif row["Experimental_Year"] in [2020, 2021]:
            mapping_2020_2021 = {"A2": "Rep1", "A3": "Rep2", "B1": "Rep3", "B2": "Rep4"}
            return mapping_2020_2021[row["Block"]]
    else:
        mapping_rest = {"A": "Rep1", "B": "Rep2", "C": "Rep3", "D": "Rep4"}
        return mapping_rest[row["Block"]]


def categorize_taxa(row):
    """
    1) Always abundant taxa (AAT) were defined as taxa with abundance ≥ 1% in all samples
    2) Always rare taxa (ART) were defined as the OTUs with abundance < 0.01% in all samples
    3) Moderate taxa (MT) were defined as OTUs with abundance between 0.01 and 1% in all samples
    4) Conditionally rare taxa (CRT) were defined as with abundance below 1% in all samples and < 0.01% in some samples
    5) Conditionally abundant taxa (CAT) were defined as taxa with abundance ≥ 0.01% in all samples and ≥ 1% in some samples but never rare (< 0.01%)
    6) Conditionally rare and abundant taxa (CRAT) were defined as OTUs with abundance varying from rare (< 0.01%) to abundant (≥ 1%).
    Source: https://doi.org/10.1186/s40168-019-0749-8 ("Stochastic processes shape microeukaryotic community assembly in a subtropical river across wet and dry seasons")
    """
    RARE = 0.0001
    ABUNDANT = 0.01

    min_value = row.min()
    max_value = row.max()

    if min_value >= ABUNDANT:
        return "AAT"
    if max_value < RARE:
        return "ART"
    if min_value >= RARE and max_value < ABUNDANT:
        return "MT"
    if max_value < ABUNDANT and min_value < RARE:
        return "CRT"
    if min_value >= RARE and max_value >= ABUNDANT:
        return "CAT"
    if min_value < RARE and max_value >= ABUNDANT:
        return "CRAT"

    return "?"


def precrop(row):
    if row["Crop"] == "Winter wheat 1":
        return "Grain maize"
    elif row["Crop"] == "Winter wheat 2":
        return "Winter rapeseed "
    elif row["Crop"] == "Grain maize":
        return "Winter wheat 2"
    else:
        return "Winter barley"


def rarify(df, sample_total=0):
    """
    Perform normalization.

    Parameters
    ----------
    df : pandas.DataFrame
        Absolute abundances.
    sample_total : int, optional
        Target read depth. If 0 or larger than the maximum sample depth,
        the minimum sample depth is used.

    Returns
    -------
    pandas.DataFrame
        Rarefied table with absolute abundances.
    """
    sample_totals = df.sum(axis=1)
    min_sample_total = sample_totals.min()
    max_sample_total = sample_totals.max()

    if sample_total == 0 or sample_total >= max_sample_total:
        sample_total = min_sample_total

    df = df.loc[sample_totals >= sample_total]

    rng = np.random.default_rng(666)
    for i, counts in enumerate(df.to_numpy()):
        total = counts.sum()

        if total == sample_total:
            continue

        positions = rng.choice(total, size=sample_total, replace=False)
        cumsum = np.cumsum(counts)
        taxa = np.searchsorted(cumsum, positions)
        new_counts = np.bincount(taxa, minlength=len(counts))
        df.iloc[i] = new_counts

    return df


def reorder_columns(df, type_label, sort_order=None):
    print("Change order of the columns...", end="")
    if sort_order is None:
        sort_order = [
            "Experimental_Year",
            "Date",
            "Crop",
            "Precrop",
            "Replicate",
            "Tillage",
            "Fertilization",
            "Treatment",
            "Habitat",
            "Beneficial",
            "Value_abs",
            "Value_rel",
            "Kingdom",
            "Phylum",
            "Class",
            "Order",
            "Family",
            "Genus",
        ]
        if type_label == "Fungi":
            sort_order.append("Species")
        sort_order.append("Category")
    return df[sort_order]


# def funguild(df):
#     print("Export data to Excel and CSV...", end="")
#     df.loc[:, "Taxon"] = df["Genus"].str.replace("_gen", "", regex=False)
#     df_funguild = pd.read_excel("../funguild.xlsx")
#
#     df_funguild_unqique = df_funguild[
#         [
#             "Taxon",
#             "Trophic Mode",
#             "Guild",
#             "Confidence Ranking",
#             "Growth Morphology",
#             "Trait",
#         ]
#     ].drop_duplicates()
#     df_funguild_unqique = df_funguild_unqique.rename(
#         columns={
#             "Trophic Mode": "Trophic_Mode",
#             "Confidence Ranking": "Confidence_Ranking",
#             "Growth Morphology": "Growth_Morphology",
#         }
#     )
#
#     df_fungi_addon = pd.merge(
#         df, df_funguild_unqique, on="Taxon", how="left"
#     )
#     df_fungi_addon = df_fungi_addon.drop(columns=["Taxon"])
#     df_fungi_addon.to_excel("../df_fungi.xlsx", sheet_name="Population", index=False)
#     df_fungi_addon.to_csv("../df_fungi.csv", index=False)
#
#     print("DONE")


def export(df, type_label, years=None, habitats=None, beneficials=None, crops=None):
    print("Export data to Excel and CSV...", end="")
    year_label = create_label(years)
    habitat_label = create_label(habitats)
    beneficial_label = create_label(beneficials)
    crop_label = create_label(crops)
    file_name = f"../df_{postprocess_label(type_label)}_{year_label}_{habitat_label}_{beneficial_label}_{crop_label}"
    df.to_excel(file_name + ".xlsx", sheet_name="Population", index=False)
    df.to_csv(file_name + ".csv", index=False)


def common_preparation(
    type_label, years=None, habitats=None, beneficials=None, crops=None
):
    print("Join all related tables and keep important columns only...", end="")
    if type_label == "Fungi":
        df = prepare_fungi()
        drop_columns = [
            "Fungi_ID",
            "Plot_ID",
            "Seq_ID",
            "ACC_Num",
            "SH_Code",
            "BioProject",
        ]
    else:
        df = prepare_bacteria()
        drop_columns = [
            "Bacteria_ID",
            "Plot_ID",
            "Seq_ID",
            "ACC_Num",
            "OTU_ID",
            "BioProject",
            "Species",
        ]
    df = df.drop(columns=drop_columns)
    print("DONE")

    if type_label == "Bacteria":
        print("Fix a wrong taxonomic ranking...")
        df.loc[df["Order"] == "Burkholderiales", "Class"] = "Betaproteobacteria"
        print("DONE")

    print("Unique taxonomic strings ensured...", end="")
    column_names = [col for col in df.columns if col != "Value_abs"]
    df = df.groupby(column_names, as_index=False)["Value_abs"].sum()
    print("DONE")

    print("Convert column Date to datetime...", end="")
    df["Date"] = pd.to_datetime(df["Date"])
    print("DONE")

    print("Exclude the November samples...", end="")
    df = df[df["Date"].dt.month != 11]
    print("DONE")

    print("Exclude Winter rapeseed...", end="")
    df = df[df["Crop"] != "Winter rapeseed"]
    print("DONE")

    print("Remove singletons...", end="")
    df = df[df["Value_abs"] > 1]
    print("DONE")

    print("Filter on input parameters...", end="")
    years = resolve_filter(years, df, "Experimental_Year")
    habitats = resolve_filter(habitats, df, "Habitat")
    beneficials = resolve_filter(beneficials, df, "Beneficial")
    crops = resolve_filter(crops, df, "Crop")
    df = df[df["Experimental_Year"].isin(years)]
    df = df[df["Habitat"].isin(habitats)]
    df = df[df["Beneficial"].isin(beneficials)]
    df = df[df["Crop"].isin(crops)]
    print("DONE")

    print("Add column Replicate (Rep1 to Rep4)...", end="")
    # For grain maize, a sampling was performend only in blocks A & B.
    # Add replication to column "Block".
    df.loc[df["Crop"] == "Grain maize", "Block"] = df["Block"] + df[
        "Replication"
    ].astype(str)
    df["Replicate"] = df.apply(replicate, axis=1)
    # In all crops except grain maize, the species appear three times (3 replications in each block).
    # However, it was only sampled once in each block, so duplicates must be removed.
    df = df.drop(columns=["Block", "Replication"]).drop_duplicates()
    print("DONE")

    print("Add column Treatment...", end="")
    df["Treatment"] = df.apply(treatment, axis=1)
    print("DONE")

    print("Add column Precrop...", end="")
    df["Precrop"] = df.apply(precrop, axis=1)
    print("DONE")

    df.reset_index(inplace=True, drop=True)

    return df

    # def extra_preparation(df, type_label):
    #     print("Categorize taxa...", end="")
    #     df_tax = df_num.copy().T
    #     df_tax = df_tax.rename_axis(agg_column)
    #     df_tax["Category"] = df_tax.apply(categorize_taxa, axis=1)
    #     df_tax = df_tax.reset_index()[[agg_column, "Category"]]
    #     df = df.merge(df_tax, on=agg_column, how="left")
    #     print("DONE")

    return df
