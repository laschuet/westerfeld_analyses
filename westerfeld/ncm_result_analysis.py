import pandas as pd
import numpy as np

# =========================
# 1. Load data
# =========================
df_fs = pd.read_excel("taxa_bounds_Fungi_Field_Soil.xlsx")
df_fs["Habitat"] = "Field_Soil"
df_rh = pd.read_excel("taxa_bounds_Fungi_Rhizosphere.xlsx")
df_rh["Habitat"] = "Rhizosphere"

df = pd.concat([df_fs, df_rh], ignore_index=True)                                               

# optional: unify naming
df["Prediction"] = df["Prediction"].replace({
    "above prediction": "above",
    "below prediction": "below"
})
df["Habitat"] = df["Habitat"].replace({
    "Field_Soil": "FS",
    "Rhizosphere": "RH"
})

# =========================
# 2. Pivot table
# =========================
pivot = df.pivot_table(
    index="Taxa",
    columns="Habitat",
    values=[
        "Mean Relative Abundance",
        "Occurrence Frequency",
        "Prediction"
    ],
    aggfunc="first"
)

# simplify column access
RA = pivot["Mean Relative Abundance"]
OC = pivot["Occurrence Frequency"]
PR = pivot["Prediction"]

# =========================
# 3. Core metrics
# =========================

pivot["FS_RA"] = RA["FS"]
pivot["RH_RA"] = RA["RH"]

pivot["FS_Occ"] = OC["FS"]
pivot["RH_Occ"] = OC["RH"]

# FC_RA = RH_RA / FS_RA
eps_ra = float(RA.stack().min()) / 2

ra_num = pivot["RH_RA"]
ra_den = pivot["FS_RA"]
ra_only = (ra_num.isna() | ra_num.eq(0)) | (ra_den.isna() | ra_den.eq(0))
pivot["FC_RA"] = np.where(
    ra_only,
    (ra_num.fillna(0) + eps_ra) / (ra_den.fillna(0) + eps_ra),
    ra_num / ra_den
)

pivot["log2FC_RA"] = np.log2(pivot["FC_RA"])

# FC_Occ = RH_Occ / FS_Occ
eps_occ = float(OC.stack().min()) / 2
occ_num = pivot["RH_Occ"]
occ_den = pivot["FS_Occ"]
occ_only = (occ_num.isna() | occ_num.eq(0)) | (occ_den.isna() | occ_den.eq(0))
pivot["FC_Occ"] = np.where(
    occ_only,
    (occ_num.fillna(0) + eps_occ) / (occ_den.fillna(0) + eps_occ),
    occ_num / occ_den
)
pivot["log2FC_Occ"] = np.log2(pivot["FC_Occ"])

# =========================
# 4. Category function
# =========================
def classify(row):
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

pivot["Category"] = pivot.apply(classify, axis=1)

# =========================
# 5. Category splits
# =========================

categories = {
    cat: pivot[pivot["Category"] == cat]
    for cat in pivot["Category"].dropna().unique()
}

# =========================
# 6. Write Excel report
# =========================

# Remove unnecessary columns for export
pivot_export = pivot.drop(
    columns=[
        "FS_RA",
        "RH_RA",
        "FS_Occ",
        "RH_Occ",
        "FC_RA",
        "FC_Occ"
    ],
    errors='ignore'
).reindex(pivot["log2FC_RA"].abs().sort_values(ascending=False).index)

with pd.ExcelWriter("ncm_result_analysis.xlsx") as writer:

    pivot_export.to_excel(writer, sheet_name="All_taxa")
    for name, table in categories.items():
        sheet_name = name[:31]  # Excel limit
        table_export = table.drop(
            columns=[
                "FS_RA",
                "RH_RA",
                "FS_Occ",
                "RH_Occ",
                "FC_RA",
                "FC_Occ"
            ],
            errors='ignore'
        ).reindex(table["log2FC_RA"].abs().sort_values(ascending=False).index)
        table_export.to_excel(writer, sheet_name=sheet_name)

print("Report written: ncm_result_analysis.xlsx")