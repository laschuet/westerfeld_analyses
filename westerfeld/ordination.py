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


def plot_perplexity_scan_multi(scans, path="FigS1_perplexity_combined.png"):
    """
        Plot KL divergence and trustworthiness against perplexity for multiple type labels.
        `scans` is a dictionary: {'Bacteria': scan_df, 'Fungi': scan_df}
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    labels = ['A', 'C', 'B', 'D']
    label_idx = 0
    
    for col_idx, (type_label, scan) in enumerate(scans.items()):
        
        # 1. Zuerst das Maximum der Trustworthiness finden (für beide Plots gleich)
        max_trust_idx = scan["trustworthiness"].idxmax()
        max_trust_val = scan["trustworthiness"].max()
        
        # Den zugehörigen KL-Wert an dieser Stelle finden
        kl_at_max_trust = scan.loc[max_trust_idx, "kl_divergence"]

        # --- Trustworthiness Plot ---
        axes[0, col_idx].plot(scan.index, scan["trustworthiness"])
        axes[0, col_idx].set_title(f"{type_label}: Trustworthiness")
        axes[0, col_idx].set_ylabel("trustworthiness")
        axes[0, col_idx].grid(True)

        # Punkt bei der Trustworthiness einzeichnen
        axes[0, col_idx].scatter(max_trust_idx, max_trust_val, color='red', s=100, zorder=5, label='Maximum')
        axes[0, col_idx].legend()

        # --- KL Divergence Plot ---
        axes[1, col_idx].plot(scan.index, scan["kl_divergence"])
        axes[1, col_idx].set_title(f"{type_label}: KL Divergence")
        axes[1, col_idx].set_xlabel("perplexity")
        axes[1, col_idx].set_ylabel(r"$D_{KL}$")
        axes[1, col_idx].grid(True)
        
        # Punkt bei der KL-Divergenz einzeichnen, der zur max Trustworthiness gehört
        axes[1, col_idx].scatter(max_trust_idx, kl_at_max_trust, color='red', s=100, zorder=5, label='Selected Perplexity')
        axes[1, col_idx].legend()

        # Label für den oberen Plot
        axes[0, col_idx].text(0.05, 0.95, labels[label_idx], transform=axes[0, col_idx].transAxes, 
                              fontsize=16, fontweight='bold', va='top')
        label_idx += 1
        
        # Label für den unteren Plot
        axes[1, col_idx].text(0.05, 0.95, labels[label_idx], transform=axes[1, col_idx].transAxes, 
                              fontsize=16, fontweight='bold', va='top')
        label_idx += 1

    fig.tight_layout()
    fig.savefig(path)
    return fig


def plot_ordination_multi(results, color_by, marker_by=None, path="Fig1_ordination_combined.png"):
    """
        Scatter the t-SNE embedding for multiple type labels side by side.
        `results` is a dictionary: {'Bacteria': result_obj, 'Fungi': result_obj}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    my_colors = ['#FF8C00', '#2E8B57'] 
    
    for ax, (type_label, result) in zip(axes, results.items()):
        x = result.embedding["x"].to_numpy()
        y = result.embedding["y"].to_numpy()
        color_values = result.embedding.index.get_level_values(color_by)
        
        ax.set_title(
            f"{type_label}\n"
            f"(n={len(x)}, perplexity={result.perplexity}, $D_{{KL}}$={result.kl_divergence:.4f}, trust={result.trustworthiness:.4f})"
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        if marker_by is None:
            for i, color_category in enumerate(np.unique(color_values)):
                mask = color_values == color_category
                current_color = my_colors[i % len(my_colors)]
                ax.scatter(
                    x[mask],
                    y[mask],
                    color=current_color,
                    label=str(color_category),
                )
        else:
            marker_values = result.embedding.index.get_level_values(marker_by)
            markers = ['o', '^'] 

            abbreviations = {
                "Field_Soil": "FS",
                "Winter wheat 1": "WW1",
                "Winter wheat 2": "WW2",
                "Rhizosphere": "RH"
            }
            
            for i, color_category in enumerate(np.unique(color_values)):
                for j, marker_category in enumerate(np.unique(marker_values)):
                    mask = (color_values == color_category) & (
                        marker_values == marker_category
                    )
                    if not mask.any():
                        continue
                    
                    current_color = my_colors[i % len(my_colors)]
                    current_marker = markers[j % len(markers)]
                    
                    ax.scatter(
                        x[mask],
                        y[mask],
                        marker=current_marker,
                        color=current_color,
                    )

    legend_elements = []

    unique_colors = np.unique([abbreviations.get(cat, cat) for result in results.values() for cat in result.embedding.index.get_level_values(color_by)])
    unique_markers = np.unique([abbreviations.get(cat, cat) for result in results.values() for cat in result.embedding.index.get_level_values(marker_by)])

    color_map = {cat: my_colors[i % len(my_colors)] for i, cat in enumerate(np.unique([abbreviations.get(cat, cat) for result in results.values() for cat in result.embedding.index.get_level_values(color_by)]))}
    
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Habitat', markerfacecolor='none', markersize=0))
    for cat in unique_colors:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map.get(cat, 'black'), markersize=10, label=cat))

    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Crop', markerfacecolor='none', markersize=0))
    
    marker_map = {'WW1': 'o', 'WW2': '^'}
    for cat in unique_markers:
        if cat in marker_map:
            legend_elements.append(plt.Line2D([0], [0], marker=marker_map[cat], color='black', markerfacecolor='black', markersize=10, label=cat))

    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5))

    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    return fig


def main():
    print("--------------")
    print("| ORDINATION |")
    print("--------------")

    crops = ["Winter wheat 1", "Winter wheat 2"]
    type_labels = ["Fungi", "Bacteria"]
    years = 2019

    all_scans = {}
    all_results = {}
    
    for type_label in type_labels:
        print(f"Scanning perplexity for {type_label}...")
        scan = scan_perplexity(type_label, "Genus", years=years, crops=crops)
        all_scans[type_label] = scan

    plot_perplexity_scan_multi(all_scans, path="FigS1_perplexity_combined.png")

    optimal_perplexities = {}
    for type_label, scan in all_scans.items():
        best_perp = scan["trustworthiness"].idxmax()      
        optimal_perplexities[type_label] = best_perp

    for type_label in type_labels:
        current_perplexity = optimal_perplexities[type_label]
        
        print(f"Calculating Ordination for {type_label} with perplexity={current_perplexity}...")
        
        result = ordination(
            type_label,
            "Genus",
            perplexity=current_perplexity,  
            years=years,
            crops=crops,
        )
        
        result.embedding.to_csv(f"ordination_{type_label}.csv")
        all_results[type_label] = result

    plot_ordination_multi(
        all_results, 
        "Habitat", 
        "Crop", 
        path="Fig1_ordination_combined.png"
    )

    # To quantify the separation this plot shows (PERMANOVA) and check it is a
    # location shift rather than unequal dispersion (PERMDISP), see
    # beta_diversity.py, which runs on the same Bray-Curtis distances.


if __name__ == "__main__":
    main()
