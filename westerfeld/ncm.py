import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from lmfit import Parameters, Model
from scipy.stats import bootstrap

from _preparation import relative_abundances


def custom_beta_cdf(p, N, m):
    """
    Only integrate over theoretically detectable abundances.
    The smallest possible non-zero abundance in a sample with N reads is 1/N.
    """
    a = N * m * p
    b = N * m * (1.0 - p)
    return sp.stats.beta.cdf(1.0, a, b) - sp.stats.beta.cdf(1.0 / N, a, b)


def ncm(type_label, file_name, years=None, habitats=None, beneficials=None, crops=None):
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
    print("DONE")

    print(
        f"Fitting neutral community model (#samples={n_samples}, #taxa={n_taxa})...",
        end="",
    )

    ncm = Model(custom_beta_cdf)
    params = Parameters()
    params.add("N", value=community_size, vary=False)
    params.add("m", value=0.5, min=0.0, max=1.0)
    ncm_result = ncm.fit(y, params, p=x, verbose=True)
    print(ncm_result.fit_report())

    N = ncm_result.params["N"].value
    m = ncm_result.params["m"].value
    Nm = N * m
    print("DONE")
    print(f"Nm: {Nm}")

    print("Plotting result...", end="")
    fig, ax = plt.subplots()
    ax.set_title(
        f"Neutral community model of {type_label} ($R^2 = {ncm_result.rsquared:.4f}$)"
    )
    ax.set_xlabel("log(Mean relative abundance)")
    ax.set_ylabel("Occurrence frequency")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    plt.plot(
        x,
        ncm_result.best_fit,
        color="#0D18B3",
        linewidth=1,
    )

    data = (ncm_result.best_fit,)
    result = bootstrap(
        data,
        np.mean,
        n_resamples=9999,
        confidence_level=0.95,
        random_state=1,
        method="percentile",
    )  # Doku angucken bzgl. Parametrisierung
    low, high = result.confidence_interval
    low_bound = ncm_result.best_fit - low
    high_bound = ncm_result.best_fit + high

    plt.plot(x, low_bound, color="#E72F52", linewidth=1, linestyle="--")
    plt.plot(x, high_bound, color="#E72F52", linewidth=1, linestyle="--")
    plt.fill_between(x, low_bound, high_bound, color="#B5BABB")

    uncertainty = ncm_result.eval_uncertainty(sigma=2)
    lower = ncm_result.best_fit - uncertainty
    upper = ncm_result.best_fit + uncertainty

    plt.plot(x, lower, color="#0D18B3", linewidth=1, linestyle="--")
    plt.plot(x, upper, color="#0D18B3", linewidth=1, linestyle="--")
    plt.fill_between(x, lower, upper, color="#7EE7FC")

    plt.scatter(
        x[y < low_bound],
        y[y < low_bound],
        color="#BACA08",
        s=1,
        label="below prediction",
    )
    plt.scatter(
        x[y > high_bound],
        y[y > high_bound],
        color="#089453",
        s=1,
        label="above prediction",
    )
    plt.scatter(
        x[(y >= low_bound) & (y <= high_bound)],
        y[(y >= low_bound) & (y <= high_bound)],
        color="#000000",
        s=1,
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ncm_{type_label}.pdf")
    print("DONE")

    print("Analyze taxa...", end="")
    taxa_sorted = taxa[sorted_indices]
    idx_low_bound = np.where(y < low_bound)[0]
    idx_high_bound = np.where(y > high_bound)[0]
    idx_neutral = np.where((y >= low_bound) & (y <= high_bound))

    taxa_low = taxa_sorted[idx_low_bound]
    taxa_high = taxa_sorted[idx_high_bound]
    taxa_neutral = taxa_sorted[idx_neutral]

    df_low = pd.DataFrame(taxa_low, columns=["Low Bound Taxa"])
    df_high = pd.DataFrame(taxa_high, columns=["High Bound Taxa"])
    df_neutral = pd.DataFrame(taxa_neutral, columns=["Neutral Taxa"])

    with pd.ExcelWriter("taxa_bounds.xlsx") as writer:
        df_low.to_excel(writer, sheet_name="Low Bound", index=False)
        df_high.to_excel(writer, sheet_name="High Bound", index=False)
        df_neutral.to_excel(writer, sheet_name="Neutral", index=False)

    print("DONE")


def main():
    print("---------------------------")
    print("| NEUTRAL COMMUNITY MODEL |")
    print("---------------------------")

    ncm(
        "Fungi",
        "ncm--",
        years=2019,
        habitats="Field_Soil",
        beneficials="Control",
        crops=["Grain maize", "Winter wheat 1", "Winter wheat 2"],
    )
    # ncm(
    #     "Fungi",
    #     "ncm--",
    #     years=2019,
    #     habitats="Rhizosphere",
    #     beneficials="Control",
    #     crops=["Grain maize", "Winter wheat 1", "Winter wheat 2"],
    # )


if __name__ == "__main__":
    main()
