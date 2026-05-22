import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from dataclasses import dataclass

from lmfit import Parameters, Model

from _preparation import relative_abundances
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
    lower: np.ndarray
    upper: np.ndarray
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


def ncm(type_label, label, years=None, habitats=None, beneficials=None, crops=None):
    _, df_rel_taxa_abundances, community_size, _ = relative_abundances(
        type_label, years, habitats, beneficials, crops
    )

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

    uncertainty = ncm_result.eval_uncertainty(sigma=2)
    lower = best_fit - uncertainty
    upper = best_fit + uncertainty
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
        lower=lower,
        upper=upper,
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

    ax.plot(x, result.lower, color="#0D18B3", linewidth=1, linestyle="--")
    ax.plot(x, result.upper, color="#0D18B3", linewidth=1, linestyle="--")
    ax.fill_between(x, result.lower, result.upper, color="#7EE7FC")

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
    y = result.y
    below = y < result.low_bound
    above = y > result.high_bound
    neutral = (y >= result.low_bound) & (y <= result.high_bound)
    return result.taxa[below], result.taxa[above], result.taxa[neutral]


def export_taxa_bounds(result, path="taxa_bounds.xlsx"):
    print("Analyze taxa...", end="")
    taxa_low, taxa_high, taxa_neutral = taxa_bounds(result)

    df_low = pd.DataFrame(taxa_low, columns=["Low Bound Taxa"])
    df_high = pd.DataFrame(taxa_high, columns=["High Bound Taxa"])
    df_neutral = pd.DataFrame(taxa_neutral, columns=["Neutral Taxa"])

    with pd.ExcelWriter(path) as writer:
        df_low.to_excel(writer, sheet_name="Low Bound", index=False)
        df_high.to_excel(writer, sheet_name="High Bound", index=False)
        df_neutral.to_excel(writer, sheet_name="Neutral", index=False)
    print("DONE")


def compare_ncm_results(results):
    """One row per result with the fit parameters and partition sizes."""
    rows = []
    for result in results:
        below, above, neutral = taxa_bounds(result)
        n_taxa = len(result.taxa)
        rows.append(
            {
                "m": result.m,
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
    index = {"below": 0, "above": 1, "neutral": 2}[partition]
    labels = [result.label for result in results]
    taxa_sets = [list(taxa_bounds(result)[index]) for result in results]

    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, taxa_i in enumerate(taxa_sets):
        for j, taxa_j in enumerate(taxa_sets):
            if not taxa_i and not taxa_j:
                matrix.iloc[i, j] = 1.0
            else:
                matrix.iloc[i, j] = calc_iou(taxa_i, taxa_j)
    return matrix


def main():
    print("---------------------------")
    print("| NEUTRAL COMMUNITY MODEL |")
    print("---------------------------")

    crops = ["Grain maize", "Winter wheat 1", "Winter wheat 2"]
    habitats = ["Field_Soil", "Rhizosphere"]

    results = []
    for habitat in habitats:
        result = ncm(
            "Fungi",
            habitat,
            years=2019,
            habitats=habitat,
            beneficials="Control",
            crops=crops,
        )
        export_taxa_bounds(result, path=f"taxa_bounds_{habitat}.xlsx")
        results.append(result)

    plot_ncm_grid(results, path="ncm.pdf")

    summary = compare_ncm_results(results)
    print(summary)
    summary.to_csv("ncm_summary.csv")

    for partition in ("above", "below"):
        overlap = compare_ncm_partitions(results, partition)
        print(overlap)
        overlap.to_csv(f"ncm_overlap_{partition}.csv")


if __name__ == "__main__":
    main()
