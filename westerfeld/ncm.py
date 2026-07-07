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
    """Draw a single NCM panel onto `ax` (figure-level framing is the caller's job)."""
    ax.set_title(f"{result.label} ($R^2 = {result.rsquared:.4f}$)")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)

    x, y = result.x, result.y
    ax.plot(x, result.best_fit, color="#0D18B3", linewidth=1)

    ax.plot(x, result.low_bound, color="#E72F52", linewidth=1, linestyle="--")
    ax.plot(x, result.high_bound, color="#E72F52", linewidth=1, linestyle="--")
    ax.fill_between(x, result.low_bound, result.high_bound, color="#B5BABB")

    below = y < result.low_bound
    above = y > result.high_bound
    neutral = (y >= result.low_bound) & (y <= result.high_bound)
    ax.scatter(x[below], y[below], color="#BACA08", s=1, label="below prediction")
    ax.scatter(x[above], y[above], color="#089453", s=1, label="above prediction")
    ax.scatter(x[neutral], y[neutral], color="#000000", s=1)

    return ax


def plot_ncm_grid(results, path="ncm.pdf", ncols=2):
    n = len(results)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 5 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    axs = axs.reshape(-1)

    for ax, result in zip(axs, results):
        plot_ncm(result, ax=ax)
    for ax in axs[n:]:
        ax.set_visible(False)

    type_labels = {result.type_label for result in results}
    title = "Neutral community model"
    if len(type_labels) == 1:
        title += f" of {next(iter(type_labels))}"
    fig.suptitle(title)
    fig.supxlabel("log(Mean relative abundance)")
    fig.supylabel("Occurrence frequency")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", markerscale=5)

    fig.savefig(path)
    return fig


def taxa_bounds(result):
    """Return taxa partitions for a result as a single DataFrame.

    The returned DataFrame contains columns:
    `Taxa`, `Mean Relative Abundance`, `Occurrence Frequency`, `Prediction`, `Habitat`.
    """
    y = result.y
    below_mask = y < result.low_bound
    above_mask = y > result.high_bound
    neutral_mask = (y >= result.low_bound) & (y <= result.high_bound)

    taxa_low = result.taxa[below_mask]
    taxa_high = result.taxa[above_mask]
    taxa_neutral = result.taxa[neutral_mask]

    habitat_map = {"Field_Soil": "FS", "Rhizosphere": "RH"}
    habitat_code = habitat_map.get(result.label, result.label)

    df_low = pd.DataFrame({
        "Taxa": taxa_low,
        "Mean Relative Abundance": result.x[below_mask],
        "Occurrence Frequency": result.y[below_mask],
        "Prediction": "below",
        "Habitat": habitat_code,
    })

    df_high = pd.DataFrame({
        "Taxa": taxa_high,
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

    df = pd.concat([df_low, df_high, df_neutral], ignore_index=True)
    return df


def compare_ncm_results(results):
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


def compare_ncm_partitions(results, partition="above"):
    """Pairwise IoU (Jaccard) of one partition's taxa across results."""
    labels = [result.label for result in results]
    taxa_sets = []
    for result in results:
        df_taxa = taxa_bounds(result)
        taxa_sets.append(df_taxa[df_taxa["Prediction"] == partition]["Taxa"].tolist())

    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, taxa_i in enumerate(taxa_sets):
        for j, taxa_j in enumerate(taxa_sets):
            if not taxa_i and not taxa_j:
                matrix.iloc[i, j] = 1.0
            else:
                matrix.iloc[i, j] = calc_iou(taxa_i, taxa_j)
    return matrix

def compute_core_metrics(
    pivot: pd.DataFrame, community_sizes: dict[str, float] | None = None
) -> pd.DataFrame:
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


def classify(row: pd.Series) -> str:
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


def build_category_splits(pivot: pd.DataFrame) -> dict:
    pivot["Category"] = pivot.apply(classify, axis=1)
    return {
        cat: pivot[pivot["Category"] == cat]
        for cat in pivot["Category"].dropna().unique()
    }


def build_prediction_contingency(pivot: pd.DataFrame) -> pd.DataFrame:
    fs_pred = pivot["Prediction"]["FS"].fillna("neutral").replace({"neutral": "within"})
    rh_pred = pivot["Prediction"]["RH"].fillna("neutral").replace({"neutral": "within"})
    contingency = pd.crosstab(
        fs_pred,
        rh_pred,
        rownames=["FS"],
        colnames=["RH"],
        dropna=False,
    )
    contingency = contingency.reindex(index=["above", "below", "within"], columns=["above", "below", "within"], fill_value=0)
    return contingency


def plot_category_counts(pivot: pd.DataFrame, type_label: str) -> None:
    order = [
        "Consistently Neutral",
        "Consistently Above",
        "Consistently Below",
        "FS Above",
        "FS Below",
        "RH Above",
        "RH Below",
        "Opposite",
    ]

    counts = pivot["Category"].value_counts().reindex(order, fill_value=0)

    color_map = {
        "Consistently Neutral": "lightgray",
        "Consistently Above": "lightgray",
        "Consistently Below": "lightgray",
        "FS Above": "skyblue",
        "FS Below": "skyblue",
        "RH Above": "skyblue",
        "RH Below": "skyblue",
        "Opposite": "royalblue",
    }
    colors = [color_map[cat] for cat in counts.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(counts.index, counts.values, color=colors)

    ax.set_title(f"Taxa pro Kategorie - {type_label}")
    ax.set_ylabel("Anzahl Taxa")
    ax.set_xlabel("Kategorie")
    ax.set_xticks(range(len(counts.index)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right")

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    output_path = f"ncm_category_counts_{type_label}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Category count plot written: {output_path}")


def plot_prediction_contingency_table(pivot: pd.DataFrame, type_label: str) -> None:
    contingency = build_prediction_contingency(pivot)

    rows = contingency.index.tolist()
    cols = contingency.columns.tolist()
    cell_text = contingency.astype(int).values.tolist()

    cell_colors = []
    for fs_pred in rows:
        row_colors = []
        for rh_pred in cols:
            if fs_pred == rh_pred:
                row_colors.append("lightgray")
            elif {fs_pred, rh_pred} == {"above", "below"}:
                row_colors.append("royalblue")
            else:
                row_colors.append("skyblue")
        cell_colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(8, 5))
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

    ax.set_title(f"Kontingenztabelle Prediction - {type_label}", pad=20)

    output_path = f"ncm_prediction_contingency_{type_label}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction contingency plot written: {output_path}")


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

        contingency = build_prediction_contingency(pivot)
        contingency.to_excel(writer, sheet_name="Contingency")

    print(f"Report written: {output_path}")


def main():
    print("---------------------------")
    print("| NEUTRAL COMMUNITY MODEL |")
    print("---------------------------")

    crops = ["Winter wheat 1", "Winter wheat 2"]
    habitats = ["Field_Soil", "Rhizosphere"]
    type_label = "Fungi"

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

    # Build taxa table from results (replaces previous external taxa_bounds files)
    taxa_dfs = []
    for r in results:
        df_taxa = taxa_bounds(r)
        taxa_dfs.append(df_taxa)
    if taxa_dfs:
        df_all = pd.concat(taxa_dfs, ignore_index=True)

        # ensure full Taxa x Habitat combinations (match previous behavior)
        taxa = df_all["Taxa"].unique()
        habitats_codes = ["FS", "RH"]
        full = pd.DataFrame([(t, h) for t in taxa for h in habitats_codes], columns=["Taxa", "Habitat"])
        df_all = pd.merge(full, df_all, on=["Taxa", "Habitat"], how="left")

        # fill defaults for missing combinations
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

        # community sizes from fitted results (map labels to FS/RH if needed)
        label_map = {"Field_Soil": "FS", "Rhizosphere": "RH"}
        community_sizes = {
            label_map.get(r.label, r.label): int(r.N) for r in results
        }

        pivot = compute_core_metrics(pivot, community_sizes=community_sizes)
        categories = build_category_splits(pivot)

        # generate plots and report
        plot_category_counts(pivot, type_label)
        plot_prediction_contingency_table(pivot, type_label)
        export_report(pivot, categories, type_label)

    plot_ncm_grid(results, path=f"ncm_{type_label}.pdf")

    summary = compare_ncm_results(results)
    print(summary)
    summary.to_csv(f"ncm_summary_{type_label}.csv")

    for partition in ("above", "below", "neutral"):
        overlap = compare_ncm_partitions(results, partition)
        print(overlap)
        overlap.to_csv(f"ncm_overlap_{type_label}_{partition}.csv")


if __name__ == "__main__":
    main()
