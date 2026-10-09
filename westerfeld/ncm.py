import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from dataclasses import dataclass

from lmfit import Parameters, Model

from _preparation import common_preparation, rarefied_taxa_table, relative_abundances
from _utils import calc_iou


@dataclass
class NCMResult:
    type_label: str
    label: str
    x: np.ndarray
    y: np.ndarray
    taxa: np.ndarray
    best_fit: np.ndarray
    low_bound: np.ndarray
    high_bound: np.ndarray
    rsquared: float
    N: float
    m: float

    @property
    def Nm(self):
        return self.N * self.m


def custom_beta_cdf(p, N, m):
    """
    Only integrate over theoretically detectable abundances.
    The smallest possible non-zero abundance in a sample with N reads is 1/N.
    """
    a = N * m * p
    b = N * m * (1.0 - p)
    return sp.stats.beta.cdf(1.0, a, b) - sp.stats.beta.cdf(1.0 / N, a, b)


def wilson_confidence_interval(p, n, alpha=0.05):
    """
    Wilson confidence interval for a binomial proportion p observed over n
    samples.
    """
    z = sp.stats.norm.ppf(1.0 - alpha / 2.0)
    denominator = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denominator
    half_width = (z / denominator) * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2))
    return center - half_width, center + half_width


def ncm(
    type_label, label, taxonomy, years=None, habitats=None, beneficials=None, crops=None
):
    df_long = common_preparation(type_label, years, habitats, beneficials, crops)
    df_abs = rarefied_taxa_table(df_long, taxonomy)
    community_size = int(df_abs.sum(axis=1).min())
    print(f"Community size: {community_size}")
    df_rel_taxa_abundances = relative_abundances(df_abs)

    print("Computing mean relative abundances and occurrence frequencies...", end="")
    n_samples, n_taxa = df_rel_taxa_abundances.shape
    taxa = df_rel_taxa_abundances.columns.to_numpy()
    mean_rel_taxa_abundances = df_rel_taxa_abundances.mean().to_numpy()
    occurrence_frequencies = (
        np.count_nonzero(df_rel_taxa_abundances, axis=0) / n_samples
    )

    mask = (mean_rel_taxa_abundances > 0) & (occurrence_frequencies > 0)
    taxa = taxa[mask]
    mean_rel_taxa_abundances = mean_rel_taxa_abundances[mask]
    occurrence_frequencies = occurrence_frequencies[mask]
    n_taxa = mask.sum()

    sorted_indices = np.argsort(mean_rel_taxa_abundances)
    x = mean_rel_taxa_abundances[sorted_indices]
    y = occurrence_frequencies[sorted_indices]
    taxa = taxa[sorted_indices]
    print("DONE")

    print(
        f"Fitting neutral community model (#samples={n_samples}, #taxa={n_taxa})...",
        end="",
    )
    model = Model(custom_beta_cdf)
    params = Parameters()
    params.add("N", value=community_size, vary=False)
    params.add("m", value=0.5, min=0.0, max=1.0)
    ncm_result = model.fit(y, params, p=x, verbose=True)
    print(ncm_result.fit_report())

    N = ncm_result.params["N"].value
    m = ncm_result.params["m"].value
    print("DONE")
    print(f"Nm: {N * m}")

    print("Computing confidence bounds...", end="")
    best_fit = ncm_result.best_fit

    low_bound, high_bound = wilson_confidence_interval(best_fit, n_samples)
    print("DONE")

    return NCMResult(
        type_label=type_label,
        label=label,
        x=x,
        y=y,
        taxa=taxa,
        best_fit=best_fit,
        low_bound=low_bound,
        high_bound=high_bound,
        rsquared=ncm_result.rsquared,
        N=N,
        m=m,
    )


def plot_ncm(result, ax):
    """Draw a single NCM panel onto `ax`"""
    ax.set_title(f"{result.label} ($R^2 = {result.rsquared:.4f}$)")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)

    x, y = result.x, result.y
    ax.plot(x, result.best_fit, color="#4169E1", linewidth=1)

    ax.plot(x, result.low_bound, color="#B0C4DE", linewidth=1, linestyle="--")
    ax.plot(x, result.high_bound, color="#B0C4DE", linewidth=1, linestyle="--")
    ax.fill_between(x, result.low_bound, result.high_bound, color="#B5BABB")

    below = y < result.low_bound
    above = y > result.high_bound
    neutral = (y >= result.low_bound) & (y <= result.high_bound)
    ax.scatter(x[below], y[below], color="#FF8C00", s=1, label="Below prediction")
    ax.scatter(x[above], y[above], color="#228B22", s=1, label="Above prediction")
    ax.scatter(x[neutral], y[neutral], color="#696969", s=1)

    return ax


def plot_ncm_combined_grid(results_nested, path="Fig2_ncm.png"):
    """
        Creates a 2x2 plot grid for Fungi and Bacteria.
        Top: Fungi, Bottom: Bacteria.
        Left: Field Soil, Right: Rhizosphere.
    """
    
    types_order = ["Fungi", "Bacteria"]
    habitats_order = ["Field_Soil", "Rhizosphere"]
    
    fig, axs = plt.subplots(
        2, 2,
        figsize=(12, 10),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    panel_labels = ["A", "B", "C", "D"]

    label_idx = 0
    for row_idx, type_label in enumerate(types_order):
        results = results_nested.get(type_label, [])
        for col_idx, habitat_label in enumerate(habitats_order):
            res = next((r for r in results if r.label == habitat_label), None)
            
            ax = axs[row_idx, col_idx]
            
            if res:
                plot_ncm(res, ax=ax)
                ax.set_title("")  # No titel

                # Panel Label (A, B, C, D) in the top left corner
                ax.text(0.02, 0.98, panel_labels[label_idx], transform=ax.transAxes,
                        fontsize=14, fontweight='bold', verticalalignment='top',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8))

                # R^2 value positioned slightly below the panel label
                text_str = f"$R^2 = {res.rsquared:.4f}$"
                ax.text(0.05, 0.88, text_str, transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"))

                # Y-axis-Label only left
                if col_idx == 0:
                    ax.set_ylabel("Occurrence frequency")
                else:
                    ax.set_ylabel("")
                    
                # X-axis-Label only below
                if row_idx == 1:
                    ax.set_xlabel("log(Mean relative abundance)")
                else:
                    ax.set_xlabel("")
            else:
                ax.text(0.5, 0.5, "No Data", ha="center")

            label_idx += 1

    # Row labels (Fungi / Bacteria)
    for ax, row_name in zip(axs[:, 0], types_order):
        ax.annotate(row_name, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90, fontweight='bold')

    # Column labels (Field Soil / Rhizosphere)
    for ax, col_name in zip(axs[0, :], habitats_order):
        display_name = col_name.replace("_", " ")
        ax.annotate(display_name, xy=(0.5, 1), xytext=(0, 10),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='bottom', fontweight='bold')


    # Add legend only once (from the first plot)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", markerscale=5)

    fig.savefig(path, dpi=300)
    print(f"Combined plot written: {path}")
    return fig


def taxa_bounds(result):
    """
        Return taxa partitions as a single DataFrame.
        The returned DataFrame contains columns:
        `Taxa`, `Mean Relative Abundance`, `Occurrence Frequency`, `Prediction`, `Habitat`
    """
    y = result.y
    below_mask = y < result.low_bound
    above_mask = y > result.high_bound
    neutral_mask = (y >= result.low_bound) & (y <= result.high_bound)

    taxa_below = result.taxa[below_mask]
    taxa_above = result.taxa[above_mask]
    taxa_neutral = result.taxa[neutral_mask]

    habitat_map = {"Field_Soil": "FS", "Rhizosphere": "RH"}
    habitat_code = habitat_map.get(result.label, result.label)

    df_below = pd.DataFrame({
        "Taxa": taxa_below,
        "Mean Relative Abundance": result.x[below_mask],
        "Occurrence Frequency": result.y[below_mask],
        "Prediction": "below",
        "Habitat": habitat_code,
    })

    df_above = pd.DataFrame({
        "Taxa": taxa_above,
        "Mean Relative Abundance": result.x[above_mask],
        "Occurrence Frequency": result.y[above_mask],
        "Prediction": "above",
        "Habitat": habitat_code,
    })

    df_neutral = pd.DataFrame({
        "Taxa": taxa_neutral,
        "Mean Relative Abundance": result.x[neutral_mask],
        "Occurrence Frequency": result.y[neutral_mask],
        "Prediction": "neutral",
        "Habitat": habitat_code,
    })

    df = pd.concat([df_below, df_above, df_neutral], ignore_index=True)
    return df


def ncm_results(results):
    """One row per result with the fit parameters and partition sizes."""
    rows = []
    for result in results:
        df_taxa = taxa_bounds(result)
        below = df_taxa[df_taxa["Prediction"] == "below"]
        above = df_taxa[df_taxa["Prediction"] == "above"]
        neutral = df_taxa[df_taxa["Prediction"] == "neutral"]
        n_taxa = len(result.taxa)
        rows.append(
            {
                "m": result.m,
                "N": result.N,
                "Nm": result.Nm,
                "R^2": result.rsquared,
                "Below": len(below),
                "Above": len(above),
                "Neutral": len(neutral),
                "Neutral fraction": len(neutral) / n_taxa if n_taxa else float("nan"),
            }
        )
    return pd.DataFrame(rows, index=[result.label for result in results])

def compute_core_metrics(pivot, community_sizes):
    """Add ratio, fold change and category columns to the pivot table."""
    RA = pivot["Mean Relative Abundance"]

    pivot["FS_RA"] = RA["FS"]
    pivot["RH_RA"] = RA["RH"]

    ra_rh = pivot["RH_RA"]
    ra_fs = pivot["FS_RA"]

    if community_sizes is None or "FS" not in community_sizes or "RH" not in community_sizes:
        pivot["FC_RA"] = np.nan
    else:
        eps_ra_fs = 1.0 / community_sizes["FS"]
        eps_ra_rh = 1.0 / community_sizes["RH"]
        pivot["FC_RA"] = (ra_rh + eps_ra_rh) / (ra_fs + eps_ra_fs)

    pivot["log2FC_RA"] = np.log2(pivot["FC_RA"])

    return pivot


def classify(row):
    fs = row["Prediction"]["FS"]
    rh = row["Prediction"]["RH"]

    if fs == rh:
        return f"Consistently {fs.capitalize()}"
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


def build_category_splits(pivot):
    pivot["Category"] = pivot.apply(classify, axis=1)
    return {
        cat: pivot[pivot["Category"] == cat]
        for cat in pivot["Category"].dropna().unique()
    }


def build_prediction_contingency(pivot):
    fs_pred = pivot["Prediction"]["FS"].fillna("neutral")
    rh_pred = pivot["Prediction"]["RH"].fillna("neutral")
    contingency = pd.crosstab(
        fs_pred,
        rh_pred,
        rownames=["FS"],
        colnames=["RH"],
        dropna=False,
    )
    contingency = contingency.reindex(index=["above", "below", "neutral"], columns=["above", "below", "neutral"], fill_value=0)
    return contingency

def plot_combined_contingency(pivot_fungi, pivot_bacteria, path="ncm_combined_contingency.png"):
    """
        Creates a figure with two contingency tables stacked vertically.
        Top: Fungi (A), Bottom: Bacteria (B).
    """

    fig, axs = plt.subplots(2, 1, figsize=(10, 8)) 
    
    data = [(pivot_fungi, "Fungi", "A"), (pivot_bacteria, "Bacteria", "B")]

    for ax, (pivot, type_name, panel_label) in zip(axs, data):
        contingency = build_prediction_contingency(pivot)
        
        rows = contingency.index.tolist()
        cols = contingency.columns.tolist()
        cell_text = contingency.astype(int).values.tolist()

        cell_colors = []
        for fs_pred in rows:
            row_colors = []
            for rh_pred in cols:
                if fs_pred == rh_pred:
                    row_colors.append("#B5BABB") 
                elif {fs_pred, rh_pred} == {"above", "below"}:
                    row_colors.append("#4169E1")
                else:
                    row_colors.append("#B0C4DE")
            cell_colors.append(row_colors)

        ax.axis("off")
        
        table = ax.table(
            cellText=cell_text,
            rowLabels=rows,
            colLabels=cols,
            cellColours=cell_colors,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        # 1. Panel Label (A / B) in the top left corner of the axis
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', verticalalignment='top')

        # 2. Type label (Fungi / Bacteria) centered at the top
        ax.annotate(type_name, xy=(0.5, 1.02), xycoords='axes fraction',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 3. Rhizosphere (Above the table)
        ax.annotate("Rhizosphere", xy=(0.5, 0.75), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 4. Field Soil (To the left of the table)
        ax.annotate("Field Soil", xy=(-0.2, 0.5), xycoords='axes fraction',
            ha='right', va='center', fontsize=10, fontweight='bold', rotation=90)

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1, hspace=0.3)
    
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Combined contingency plot written: {path}")

def export_report(pivot, categories, type_label):
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
    print("---------------------------")
    print("| NEUTRAL COMMUNITY MODEL |")
    print("---------------------------")

    crops = ["Winter wheat 1", "Winter wheat 2"]
    habitats = ["Field_Soil", "Rhizosphere"]
    type_labels = ["Fungi", "Bacteria"] 

    all_results_nested = {}
    all_pivots = {}

    for type_label in type_labels:
        print(f"\nProcessing Type: {type_label}")
        results = []
        for habitat in habitats:
            result = ncm(
                type_label,
                habitat,
                "Genus",
                years=2019,
                habitats=habitat,
                crops=crops,
            )
            results.append(result)
        
        all_results_nested[type_label] = results

        taxa_dfs = []
        for r in results:
            df_taxa = taxa_bounds(r)
            taxa_dfs.append(df_taxa)
        
        if taxa_dfs:
            df_all = pd.concat(taxa_dfs, ignore_index=True)

            taxa = df_all["Taxa"].unique()
            habitats_codes = ["FS", "RH"]
            full = pd.DataFrame([(t, h) for t in taxa for h in habitats_codes], columns=["Taxa", "Habitat"])
            df_all = pd.merge(full, df_all, on=["Taxa", "Habitat"], how="left")

            # fill defaults
            df_all["Prediction"] = df_all["Prediction"].fillna("neutral").astype(str).str.strip().replace({"": "neutral"})
            if "Mean Relative Abundance" in df_all.columns:
                df_all["Mean Relative Abundance"] = df_all["Mean Relative Abundance"].fillna(0)
            if "Occurrence Frequency" in df_all.columns:
                df_all["Occurrence Frequency"] = df_all["Occurrence Frequency"].fillna(0)

            pivot = df_all.pivot_table(
                index="Taxa",
                columns="Habitat",
                values=[
                    "Mean Relative Abundance",
                    "Occurrence Frequency",
                    "Prediction",
                ],
                aggfunc="first",
            )

            label_map = {"Field_Soil": "FS", "Rhizosphere": "RH"}
            community_sizes = {
                label_map.get(r.label, r.label): int(r.N) for r in results
            }

            pivot = compute_core_metrics(pivot, community_sizes=community_sizes)
            categories = build_category_splits(pivot)

            all_pivots[type_label] = pivot

            export_report(pivot, categories, type_label)

        summary = ncm_results(results)
        summary.to_csv(f"ncm_summary_{type_label}.csv")

    print("\nGenerating combined NCM plot...")
    plot_ncm_combined_grid(all_results_nested, path="Fig2_ncm.png")

    print("\nGenerating combined contingency plot...")
    if "Fungi" in all_pivots and "Bacteria" in all_pivots:
        plot_combined_contingency(all_pivots["Fungi"], all_pivots["Bacteria"], path="Fig3_contingency.png")


if __name__ == "__main__":
    main()