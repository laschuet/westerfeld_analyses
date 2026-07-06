import argparse

import numpy as np
import pandas as pd


def load_taxa_bounds(type_label: str) -> pd.DataFrame:
    """Load the two habitat sheets for the selected type label."""
    fs_path = f"taxa_bounds_{type_label}_Field_Soil.xlsx"
    rh_path = f"taxa_bounds_{type_label}_Rhizosphere.xlsx"

    df_fs = pd.read_excel(fs_path)
    df_fs["Habitat"] = "FS"
    df_rh = pd.read_excel(rh_path)
    df_rh["Habitat"] = "RH"

    df = pd.concat([df_fs, df_rh], ignore_index=True)
    df["Prediction"] = df["Prediction"].replace(
        {
            "above prediction": "above",
            "below prediction": "below",
        }
    )

    # Ensure every Taxa x Habitat combination exists so downstream pivot
    # produces explicit entries for taxa missing in one habitat.
    if "Taxa" in df.columns:
        taxa = df["Taxa"].unique()
        habitats = ["FS", "RH"]
        full = pd.DataFrame([(t, h) for t in taxa for h in habitats], columns=["Taxa", "Habitat"]) 
        df = pd.merge(full, df, on=["Taxa", "Habitat"], how="left")

        # Fill defaults for missing combinations
        df["Prediction"] = df["Prediction"].fillna("neutral").astype(str).str.strip().replace({"": "neutral"})
        if "Mean Relative Abundance" in df.columns:
            df["Mean Relative Abundance"] = df["Mean Relative Abundance"].fillna(0)
        if "Occurrence Frequency" in df.columns:
            df["Occurrence Frequency"] = df["Occurrence Frequency"].fillna(0)

    return df


def load_ncm_summary(type_label: str) -> dict[str, float]:
    """Load habitat-specific community size N values from the NCM summary CSV."""
    path = f"ncm_summary_{type_label}.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.to_series().replace({"Field_Soil": "FS", "Rhizosphere": "RH"})
    return df["N"].astype(int).to_dict()


def compute_core_metrics(
    pivot: pd.DataFrame, community_sizes: dict[str, float] | None = None
) -> pd.DataFrame:
    """Add ratio, fold change and category columns to the pivot table."""
    RA = pivot["Mean Relative Abundance"]
    #OC = pivot["Occurrence Frequency"]

    pivot["FS_RA"] = RA["FS"]
    pivot["RH_RA"] = RA["RH"]

    ra_rh = pivot["RH_RA"]
    ra_fs = pivot["FS_RA"]

    if community_sizes is None or "FS" not in community_sizes or "RH" not in community_sizes:
        eps_ra = float(RA.stack().min()) / 2
        pivot["FC_RA"] = (ra_rh + eps_ra) / (ra_fs + eps_ra)
    else:
        eps_ra_fs = 1.0 / community_sizes["FS"]
        eps_ra_rh = 1.0 / community_sizes["RH"]
        pivot["FC_RA"] = (ra_rh + eps_ra_rh) / (ra_fs + eps_ra_fs)

    pivot["log2FC_RA"] = np.log2(pivot["FC_RA"])


    #pivot["FS_Occ"] = OC["FS"]
    #pivot["RH_Occ"] = OC["RH"]

    #occ_rh = pivot["RH_Occ"]
    #occ_fs = pivot["FS_Occ"]

    #if community_sizes is None or "FS" not in community_sizes or "RH" not in community_sizes:
    #    eps_occ = float(OC.stack().min()) / 2
    #    pivot["FC_Occ"] = (occ_rh + eps_occ) / (occ_fs + eps_occ)
    #else:
    #    eps_occ_fs = 1.0 / community_sizes["FS"]
    #    eps_occ_rh = 1.0 / community_sizes["RH"]
    #    pivot["FC_Occ"] = (occ_rh + eps_occ_rh) / (occ_fs + eps_occ_fs)

    #pivot["log2FC_Occ"] = np.log2(pivot["FC_Occ"])
    return pivot


def classify(row: pd.Series) -> str:
    fs = row["Prediction"]["FS"]
    rh = row["Prediction"]["RH"]

    if fs == rh:
        return f"Consistently {fs}"
    if (fs == "above" and rh == "below") or (fs == "below" and rh == "above"):
        return "Opposite"
    if fs == "above" and rh == "neutral":
        return "FS Above"
    if fs == "below" and rh == "neutral":
        return "FS Below"
    if fs == "neutral" and rh == "above":
        return "RH Above"
    if fs == "neutral" and rh == "below":
        return "RH Below"
    return "Other"


def build_category_splits(pivot: pd.DataFrame) -> dict:
    pivot["Category"] = pivot.apply(classify, axis=1)
    return {
        cat: pivot[pivot["Category"] == cat]
        for cat in pivot["Category"].dropna().unique()
    }


def export_report(pivot: pd.DataFrame, categories: dict, type_label: str) -> None:
    drop_columns = [
        "FS_RA",
        "RH_RA",
        "FS_Occ",
        "RH_Occ",
        "FC_RA",
        "FC_Occ",
    ]

    pivot_export = (
        pivot.drop(columns=drop_columns, errors="ignore")
        .reindex(pivot["log2FC_RA"].abs().sort_values(ascending=False).index)
    )

    output_path = f"ncm_result_analysis_{type_label}.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        pivot_export.to_excel(writer, sheet_name="All_taxa")

        for name, table in categories.items():
            table_export = (
                table.drop(columns=drop_columns, errors="ignore")
                .reindex(table["log2FC_RA"].abs().sort_values(ascending=False).index)
            )
            sheet_name = name[:31]
            table_export.to_excel(writer, sheet_name=sheet_name)

    print(f"Report written: {output_path}")


def main():
    print("-------------------------------")
    print("| NCM RESULT ANALYSIS SCRIPT |")
    print("-------------------------------")

    type_label = "Fungi"

    df = load_taxa_bounds(type_label)
    community_sizes = load_ncm_summary(type_label)
    pivot = df.pivot_table(
        index="Taxa",
        columns="Habitat",
        values=[
            "Mean Relative Abundance",
            "Occurrence Frequency",
            "Prediction",
        ],
        aggfunc="first",
    )

    pivot = compute_core_metrics(pivot, community_sizes=community_sizes)
    categories = build_category_splits(pivot)
    export_report(pivot, categories, type_label)


if __name__ == "__main__":
    main()

