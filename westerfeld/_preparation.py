import numpy as np
import pandas as pd


EXPERIMENT_COLUMNS = [
    "Experimental_Year",
    "Date",
    "Crop",
    "Replicate",
    "Tillage",
    "Fertilization",
    "Treatment",
    "Habitat",
    "Beneficial",
    "Precrop",
]

YEARS = 2019
CROPS = ["Winter wheat 1", "Winter wheat 2"]
HABITATS = ["Field_Soil", "Rhizosphere"]
KINGDOMS = ["Fungi", "Bacteria"]
TAXONOMY = "Genus"

# Samples with a low share of reads in one phylum, per kingdom: (phylum,
# threshold). In 2019 winter wheat, the share of Actinomycetota reads splits the
# bacterial samples into two groups: about half have at most 4.3 % (median
# 1.5 %), the others at least 5.1 % (median 21 %). The low samples cluster in
# winter wheat 1 (suspected technical cause: extraction/PCR/sequencing batch).
LOW_PHYLUM = {"Bacteria": ("Actinomycetota", 0.05)}

# Global switch: if True, `common_preparation` removes the samples flagged by
# LOW_PHYLUM, so every analysis (ordination, NCM, co-occurrence, taxa
# prediction) works without them. Set to False to keep them.
EXCLUDE_LOW_PHYLUM = False

# Minimum number of reads per sample: `rarefy` removes samples with fewer reads
# before rarefying to the smallest remaining sample, so one very shallow sample
# does not set the depth for all others. In 2019 winter wheat, this affects
# only one bacterial sample (2,392 reads; next smallest 6,585), which is also a
# low-Actinomycetota sample. 0 disables the filter.
MIN_READS = 6000


def resolve_filter(values, df, column):
    """
    Normalize a filter argument into the list of values to keep.
    """
    if values is None:
        return list(np.unique(df[column]))
    if isinstance(values, (int, np.integer, str)):
        return [values]
    return values


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
    df_fungi = pd.read_csv("../lte_westerfeld.V1_0_FUNGI.csv", low_memory=False)

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
    df_bacteria = pd.read_csv("../lte_westerfeld.V1_0_BACTERIA.csv", low_memory=False)

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


def precrop(row):
    if row["Crop"] == "Winter wheat 1":
        return "Grain maize"
    elif row["Crop"] == "Winter wheat 2":
        return "Winter rapeseed "
    elif row["Crop"] == "Grain maize":
        return "Winter wheat 2"
    else:
        return "Winter barley"


def rarefy(df, seed=42):
    """
    Perform rarefaction: remove samples with fewer than MIN_READS reads, then
    subsample the reads of every remaining sample without replacement to the
    read count of the smallest remaining sample.

    Parameters
    ----------
    df : pandas.DataFrame
        Absolute abundances.
    seed : int, optional
        Seed of the random subsampling.

    Returns
    -------
    pandas.DataFrame
        Rarefied table with absolute abundances.
    """
    sample_totals = df.sum(axis=1)
    too_shallow = sample_totals < MIN_READS
    if too_shallow.any():
        print(
            f"Rarefaction drops {too_shallow.sum()} sample(s) with fewer than "
            f"{MIN_READS} reads: {sorted(sample_totals[too_shallow])}"
        )
    df = df.loc[~too_shallow]
    sample_total = df.sum(axis=1).min()

    rng = np.random.default_rng(seed)
    for i, counts in enumerate(df.to_numpy()):
        total = counts.sum()

        if total == sample_total:
            continue

        positions = rng.choice(total, size=sample_total, replace=False)
        cumsum = np.cumsum(counts)
        # Read positions 0..total-1; taxon j owns [cumsum[j-1], cumsum[j]).
        # side="right" maps a position equal to cumsum[j-1] to taxon j (with
        # the default side="left" it would go to taxon j-1, shifting one read).
        taxa = np.searchsorted(cumsum, positions, side="right")
        new_counts = np.bincount(taxa, minlength=len(counts))
        df.iloc[i] = new_counts

    return df


def phylum_reads(df):
    return df.pivot_table(
        index=EXPERIMENT_COLUMNS,
        columns="Phylum",
        values="Value_abs",
        aggfunc="sum",
        fill_value=0,
    )


def low_phylum_flags(df, kingdom):
    """
    True if the sample's share of reads in the phylum given in LOW_PHYLUM is
    below the threshold. Computed on the long table as returned by
    `common_preparation` (unrarefied reads). None if LOW_PHYLUM defines no
    rule for `kingdom`.
    """
    if kingdom not in LOW_PHYLUM:
        return None

    phylum, threshold = LOW_PHYLUM[kingdom]
    reads = phylum_reads(df)
    in_phylum = reads[phylum] if phylum in reads.columns else 0

    return (in_phylum / reads.sum(axis=1)) < threshold


def drop_low_phylum(df, kingdom):
    """Remove all rows of the samples flagged by `low_phylum_flags`."""
    flags = low_phylum_flags(df, kingdom)
    if flags is None or not flags.any():
        return df

    phylum, threshold = LOW_PHYLUM[kingdom]
    print(
        f"{flags.sum()} of {len(flags)} samples with "
        f"< {threshold:.0%} {phylum} reads affected...",
        end="",
    )
    flagged_samples = flags.index[flags]
    row_samples = pd.MultiIndex.from_frame(df[EXPERIMENT_COLUMNS])

    return df[~row_samples.isin(flagged_samples)]


def common_preparation(
    kingdom, years=None, habitats=None, beneficials=None, crops=None
):
    print("Join all related tables and keep important columns only...", end="")
    if kingdom == "Fungi":
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

    if kingdom == "Bacteria":
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

    if EXCLUDE_LOW_PHYLUM and kingdom in LOW_PHYLUM:
        print("Exclude low phylum...", end="")
        df = drop_low_phylum(df, kingdom)
        print("DONE")

    df.reset_index(inplace=True, drop=True)

    return df


def taxa_table(df, taxonomy):
    """
    Pivot the long table into a wide table of absolute abundances
    (rows = samples, columns = taxa) aggregated at the given taxonomy level.

    Parameters
    ----------
    df : pandas.DataFrame
        Long table (as returned by `common_preparation`).
    taxonomy : str
        Taxonomy column to aggregate at (e.g. "Species", "Genus", "Family").
    """
    return df.pivot_table(
        index=EXPERIMENT_COLUMNS,
        columns=taxonomy,
        values="Value_abs",
        aggfunc="sum",
        fill_value=0,
    )


def rarefied_taxa_table(df, taxonomy):
    """
    Pivot the long table into a wide table of absolute abundances
    (rows = samples, columns = taxa) aggregated at the given taxonomy level,
    and rarefy it to a common read depth (see `rarefy`).

    Parameters
    ----------
    df : pandas.DataFrame
        Long table (as returned by `common_preparation`).
    taxonomy : str
        Taxonomy column to aggregate at (e.g. "Species", "Genus", "Family").
    """
    return rarefy(taxa_table(df, taxonomy))


def relative_abundances(df):
    """
    Convert absolute abundances into relative abundances.

    Parameters
    ----------
    df : pandas.DataFrame
        Absolute abundances (rows = samples, columns = taxa).

    Returns
    -------
    pandas.DataFrame
        Relative abundances per taxon; each row sums to 1 (empty rows stay 0).
    """
    return df.div(df.sum(axis=1), axis=0).fillna(0)


def filter_prevalence(df, min_prevalence):
    """Keep taxa (columns) present in at least `min_prevalence` of the samples."""
    prevalence = (df > 0).mean(axis=0)
    return df.loc[:, prevalence >= min_prevalence]


def mclr(df, pseudocount=1):
    """
    Modified centered log-ratio transform (Yoon et al. 2019; rows = samples,
    columns = taxa).

    The CLR is taken over each sample's non-zero entries; zeros stay zero. All
    non-zero values are then shifted by one global constant (|min CLR| +
    `pseudocount`) so they stay positive and distinct from the zeros. As the
    shift is the same for every sample, the CLR differences between samples are kept.
    """
    log = np.log(df.where(df > 0))
    clr = log.sub(log.mean(axis=1), axis=0)
    shift = abs(min(clr.min().min(), 0)) + pseudocount
    return (clr + shift).fillna(0)
