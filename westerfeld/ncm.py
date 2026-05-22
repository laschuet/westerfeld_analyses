import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from dataclasses import dataclass

from lmfit import Parameters, Model

from _preparation import relative_abundances


@dataclass
class NCMResult:
    type_label: str
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


def ncm(
    type_label, file_name, years=None, habitats=None, beneficials=None, crops=None
) -> NCMResult:
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


def plot_ncm(result, ax=None):
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots()

    ax.set_title(
        f"Neutral community model of {result.type_label} ($R^2 = {result.rsquared:.4f}$)"
    )
    ax.set_xlabel("log(Mean relative abundance)")
    ax.set_ylabel("Occurrence frequency")
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
    ax.legend()

    if standalone:
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(f"ncm_{result.type_label}.pdf")

    return ax


def taxa_bounds(result: NCMResult):
    y = result.y
    below = y < result.low_bound
    above = y > result.high_bound
    neutral = (y >= result.low_bound) & (y <= result.high_bound)
    return result.taxa[below], result.taxa[above], result.taxa[neutral]


def export_taxa_bounds(result: NCMResult, path="taxa_bounds.xlsx"):
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


def main():
    print("---------------------------")
    print("| NEUTRAL COMMUNITY MODEL |")
    print("---------------------------")

    ncm_1 = ncm(
        "Fungi",
        "ncm--",
        years=2019,
        habitats="Field_Soil",
        beneficials="Control",
        crops=["Grain maize", "Winter wheat 1", "Winter wheat 2"],
    )
    plot_ncm(ncm_1)
    export_taxa_bounds(ncm_1)


if __name__ == "__main__":
    main()
