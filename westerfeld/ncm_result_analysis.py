import argparse

import numpy as np
import pandas as pd


def load_taxa_bounds(type_label: str) -> pd.DataFrame:
    """Load the two habitat sheets for the selected type label."""
    fs_path = f"taxa_bounds_{type_label}_Field_Soil.xlsx"
    rh_path = f"taxa_bounds_{type_label}_Rhizosphere.xlsx"

    df_fs = pd.read_excel(fs_path)
    df_fs["Habitat"] = "Field_Soil"
    df_rh = pd.read_excel(rh_path)
    df_rh["Habitat"] = "Rhizosphere"

    df = pd.concat([df_fs, df_rh], ignore_index=True)
    df["Prediction"] = df["Prediction"].replace(
        {
            "above prediction": "above",
            "below prediction": "below",
        }
    )
    df["Habitat"] = df["Habitat"].replace(
        {
            "Field_Soil": "FS",
            "Rhizosphere": "RH",
        }
    )

    return df


def compute_core_metrics(pivot: pd.DataFrame) -> pd.DataFrame:
    """Add ratio, fold change and category columns to the pivot table."""
    RA = pivot["Mean Relative Abundance"]
    OC = pivot["Occurrence Frequency"]

    pivot["FS_RA"] = RA["FS"]
    pivot["RH_RA"] = RA["RH"]
    pivot["FS_Occ"] = OC["FS"]
    pivot["RH_Occ"] = OC["RH"]

    eps_ra = float(RA.stack().min()) / 2
    ra_num = pivot["RH_RA"]
    ra_den = pivot["FS_RA"]
    ra_only = (ra_num.isna() | ra_num.eq(0)) | (ra_den.isna() | ra_den.eq(0))
    pivot["FC_RA"] = np.where(
        ra_only,
        (ra_num.fillna(0) + eps_ra) / (ra_den.fillna(0) + eps_ra),
        ra_num / ra_den,
    )
    pivot["log2FC_RA"] = np.log2(pivot["FC_RA"])

    eps_occ = float(OC.stack().min()) / 2
    occ_num = pivot["RH_Occ"]
    occ_den = pivot["FS_Occ"]
    occ_only = (occ_num.isna() | occ_num.eq(0)) | (occ_den.isna() | occ_den.eq(0))
    pivot["FC_Occ"] = np.where(
        occ_only,
        (occ_num.fillna(0) + eps_occ) / (occ_den.fillna(0) + eps_occ),
        occ_num / occ_den,
    )
    pivot["log2FC_Occ"] = np.log2(pivot["FC_Occ"])

    return pivot


def classify(row: pd.Series) -> str:
    fs = row["Prediction"]["FS"]
    rh = row["Prediction"]["RH"]

    if pd.isna(fs) and pd.isna(rh):
        return "No data"
    if pd.isna(fs):
        return f"{rh} RH only"
    if pd.isna(rh):
        return f"{fs} FS only"
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

    type_label = "Bacteria"

    df = load_taxa_bounds(type_label)
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

    pivot = compute_core_metrics(pivot)
    categories = build_category_splits(pivot)
    export_report(pivot, categories, type_label)


if __name__ == "__main__":
    main()

