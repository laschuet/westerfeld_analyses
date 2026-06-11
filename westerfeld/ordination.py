import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataclasses import dataclass

from matplotlib.lines import Line2D
from sklearn.manifold import TSNE, trustworthiness

from _preparation import (
    common_preparation,
    rarefied_taxa_table,
    relative_abundances,
)


# Bray-Curtis is the standard dissimilarity for (relative) abundance data, so
# ordination runs on relative abundances directly -- no mCLR. A non-Euclidean
# metric also forces TSNE's exact method (Barnes-Hut only supports Euclidean).
METRIC = "braycurtis"


@dataclass
class OrdinationResult:
    type_label: str
    # Embedding rows = samples (indexed by EXPERIMENT_COLUMNS metadata), columns = ["x", "y"].
    embedding: pd.DataFrame
    perplexity: float
    kl_divergence: float
    trustworthiness: float


def _prepare_relative_abundances(
    type_label, taxonomy, years, habitats, beneficials, crops
):
    df_long = common_preparation(type_label, years, habitats, beneficials, crops)
    df_abs = rarefied_taxa_table(df_long, taxonomy)
    return relative_abundances(df_abs)


def _tsne(df_relative, perplexity):
    tsne = TSNE(
        metric=METRIC,
        n_components=2,
        method="exact",
        perplexity=perplexity,
        n_jobs=-1,
        random_state=42,
    )
    embedding = tsne.fit_transform(df_relative)
    trust = trustworthiness(df_relative, embedding, metric=METRIC)
    return embedding, tsne.kl_divergence_, trust


def scan_perplexity(
    type_label,
    taxonomy,
    years=None,
    habitats=None,
    beneficials=None,
    crops=None,
    step=5,
):
    """
    Compute t-SNE quality metrics across a range of perplexity values to help
    choose one. Returns a DataFrame indexed by perplexity with the KL divergence
    and trustworthiness of each embedding (perplexity must stay below the sample
    count, so the range tops out there).
    """
    df_relative = _prepare_relative_abundances(
        type_label, taxonomy, years, habitats, beneficials, crops
    )
    n_samples = df_relative.shape[0]
    print(f"Scanning perplexity over {n_samples} samples...")

    rows = []
    for perplexity in range(step, n_samples, step):
        _, kl_divergence, trust = _tsne(df_relative, perplexity)
        rows.append(
            {
                "perplexity": perplexity,
                "kl_divergence": kl_divergence,
                "trustworthiness": trust,
            }
        )
    return pd.DataFrame(rows).set_index("perplexity")


def ordination(
    type_label,
    taxonomy,
    perplexity,
    years=None,
    habitats=None,
    beneficials=None,
    crops=None,
):
    """
    Compute a 2D t-SNE ordination of community composition (Bray-Curtis on
    relative abundances). The returned embedding keeps the per-sample metadata
    index, so it can be coloured/marked by any experimental factor afterwards.
    """
    df_relative = _prepare_relative_abundances(
        type_label, taxonomy, years, habitats, beneficials, crops
    )
    n_samples = df_relative.shape[0]
    print(f"Learning manifold over {n_samples} samples...")

    # t-SNE requires perplexity < n_samples; clamp so a fixed value never crashes
    # on a small filtered subset.
    if perplexity >= n_samples:
        perplexity = max(1, n_samples - 1)
        print(f"Perplexity reduced to {perplexity} (must stay below n_samples).")

    embedding, kl_divergence, trust = _tsne(df_relative, perplexity)
    embedding_df = pd.DataFrame(embedding, columns=["x", "y"], index=df_relative.index)
    print(f"KL divergence: {kl_divergence:.4f}, trustworthiness: {trust:.4f}")

    return OrdinationResult(
        type_label=type_label,
        embedding=embedding_df,
        perplexity=perplexity,
        kl_divergence=kl_divergence,
        trustworthiness=trust,
    )


def plot_perplexity_scan(scan, path="ordination_perplexity.jpg"):
    """Plot KL divergence and trustworthiness against perplexity side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(scan.index, scan["kl_divergence"])
    axes[0].set_xlabel("perplexity")
    axes[0].set_ylabel(r"$D_{KL}$")
    axes[0].grid(True)
    axes[1].plot(scan.index, scan["trustworthiness"])
    axes[1].set_xlabel("perplexity")
    axes[1].set_ylabel("trustworthiness")
    axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(path)
    return fig


def plot_ordination(result, color_by, marker_by=None, path="ordination.jpg"):
    """
    Scatter the t-SNE embedding, colouring points by `color_by` (the name of an
    experimental factor in the sample index). Optionally overlay a second factor
    via `marker_by`. Leave it `None` for a single-factor plot. t-SNE axes are not
    interpretable on their own. Only the relative arrangement of points carries
    meaning.
    """
    x = result.embedding["x"].to_numpy()
    y = result.embedding["y"].to_numpy()
    color_values = result.embedding.index.get_level_values(color_by)
    colormap = plt.get_cmap("tab10")

    fig, ax = plt.subplots()
    ax.set_title(
        f"t-SNE of {result.type_label} abundances "
        f"(n={len(x)}, perplexity={result.perplexity}, "
        rf"$D_{{KL}}$={result.kl_divergence:.4f}, "
        f"trustworthiness={result.trustworthiness:.4f})"
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    if marker_by is None:
        for i, color_category in enumerate(np.unique(color_values)):
            mask = color_values == color_category
            ax.scatter(
                x[mask],
                y[mask],
                color=colormap(i % colormap.N),
                label=str(color_category),
            )
    else:
        marker_values = result.embedding.index.get_level_values(marker_by)
        markers = Line2D.filled_markers
        for i, color_category in enumerate(np.unique(color_values)):
            for j, marker_category in enumerate(np.unique(marker_values)):
                mask = (color_values == color_category) & (
                    marker_values == marker_category
                )
                if not mask.any():
                    continue
                ax.scatter(
                    x[mask],
                    y[mask],
                    marker=markers[j % len(markers)],
                    color=colormap(i % colormap.N),
                    label=f"{color_category} / {marker_category}",
                )

    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path)
    return fig


def main():
    print("--------------")
    print("| ORDINATION |")
    print("--------------")

    crops = ["Winter wheat 1", "Winter wheat 2"]

    # If a good perplexity is unknown, scan first and inspect the plot.
    scan = scan_perplexity("Fungi", "Genus", years=2019, crops=crops)
    plot_perplexity_scan(scan, path="ordination_perplexity.jpg")

    result = ordination(
        "Fungi",
        "Genus",
        perplexity=30,
        years=2019,
        crops=crops,
    )
    result.embedding.to_csv("ordination.csv")
    plot_ordination(result, "Habitat", path="ordination.jpg")

    # To quantify the separation this plot shows (PERMANOVA) and check it is a
    # location shift rather than unequal dispersion (PERMDISP), see
    # beta_diversity.py, which runs on the same Bray-Curtis distances.


if __name__ == "__main__":
    main()
