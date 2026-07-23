import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataclasses import dataclass

from matplotlib.lines import Line2D
from sklearn.manifold import TSNE, trustworthiness

from skbio import DistanceMatrix
from skbio.stats.distance import (
    permanova as skbio_permanova,
    permdisp as skbio_permdisp,
)
from sklearn.metrics.pairwise import pairwise_distances


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


def _distance_matrix(df_relative):
    """
    Bray-Curtis distances of the relative abundances as a skbio DistanceMatrix.

    skbio validates that the matrix is symmetric and hollow, so floating-point
    asymmetry from pairwise_distances is averaged out and the diagonal zeroed.
    """
    distances = pairwise_distances(df_relative.values, metric=METRIC)
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0.0)
    ids = [str(i) for i in range(len(df_relative))]
    return DistanceMatrix(distances, ids)


def permanova(
    type_label,
    taxonomy,
    factor,
    years=None,
    habitats=None,
    beneficials=None,
    crops=None,
    permutations=999,
    seed=42,
):
    """
    PERMANOVA: does community composition differ across the groups of `factor`
    (an experimental factor in the sample index)?

    Runs on the full Bray-Curtis distances of the relative abundances -- the same
    dissimilarities the ordination visualizes, NOT the lossy 2D embedding.
    """
    df_relative = _prepare_relative_abundances(
        type_label, taxonomy, years, habitats, beneficials, crops
    )
    grouping = df_relative.index.get_level_values(factor).to_numpy()
    return skbio_permanova(
        _distance_matrix(df_relative), grouping, permutations=permutations, seed=seed
    )


def permdisp(
    type_label,
    taxonomy,
    factor,
    years=None,
    habitats=None,
    beneficials=None,
    crops=None,
    test="median",
    permutations=999,
    seed=42,
):
    """
    PERMDISP / betadisper: do the groups of `factor` differ in multivariate
    dispersion (spread around their centroid)?

    The companion to PERMANOVA: a significant PERMANOVA can stem from a centroid
    shift (location), unequal dispersion, or both. A non-significant PERMDISP
    supports reading the PERMANOVA result as a genuine location difference.
    Runs on the full Bray-Curtis distances of the relative abundances.
    """
    df_relative = _prepare_relative_abundances(
        type_label, taxonomy, years, habitats, beneficials, crops
    )
    grouping = df_relative.index.get_level_values(factor).to_numpy()
    return skbio_permdisp(
        _distance_matrix(df_relative),
        grouping,
        test=test,
        permutations=permutations,
        seed=seed,
    )

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
        
        max_trust_idx = scan["trustworthiness"].idxmax()
        max_trust_val = scan["trustworthiness"].max()
        
        kl_at_max_trust = scan.loc[max_trust_idx, "kl_divergence"]

        axes[0, col_idx].plot(scan.index, scan["trustworthiness"])
        axes[0, col_idx].set_title(f"{type_label}: Trustworthiness")
        axes[0, col_idx].set_ylabel("trustworthiness")
        axes[0, col_idx].grid(True)

        axes[0, col_idx].scatter(max_trust_idx, max_trust_val, color='red', s=100, zorder=5, label='Maximum')
        axes[0, col_idx].legend()

        axes[1, col_idx].plot(scan.index, scan["kl_divergence"])
        axes[1, col_idx].set_title(f"{type_label}: KL Divergence")
        axes[1, col_idx].set_xlabel("perplexity")
        axes[1, col_idx].set_ylabel(r"$D_{KL}$")
        axes[1, col_idx].grid(True)
        
        axes[1, col_idx].scatter(max_trust_idx, kl_at_max_trust, color='red', s=100, zorder=5, label='Selected Perplexity')
        axes[1, col_idx].legend()

        axes[0, col_idx].text(0.05, 0.95, labels[label_idx], transform=axes[0, col_idx].transAxes, 
                              fontsize=16, fontweight='bold', va='top')
        label_idx += 1
        
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
        
        # --- Statistik berechnen ---
        stat_text = ""
        try:           
            perm_res = permanova(
                type_label, 
                "Genus", 
                "Habitat", 
                years=2019, 
                crops=["Winter wheat 1", "Winter wheat 2"]
            )
            p_val_perm = perm_res['p-value']
            
            disp_res = permdisp(
                type_label, 
                "Genus", 
                "Habitat", 
                years=2019, 
                crops=["Winter wheat 1", "Winter wheat 2"], 
                test="centroid"
            )
            p_val_disp = disp_res['p-value']
            
            # Text formatieren (z.B. p < 0.001 oder p = 0.045)
            perm_str = f"p < 0.001" if p_val_perm < 0.001 else f"p = {p_val_perm:.3f}"
            disp_str = f"p < 0.001" if p_val_disp < 0.001 else f"p = {p_val_disp:.3f}"
            
            stat_text = (f"PERMANOVA: {perm_str}\n"
                         f"PERMDISP:  {disp_str}")
                         
        except Exception as e:
            print(f"Warnung: Statistik konnte für {type_label} nicht berechnet werden: {e}")
            stat_text = "Stats: N/A"

        ax.set_title(
            f"{type_label}\n"
            f"(n={len(x)}, perplexity={result.perplexity}, $D_{{KL}}$={result.kl_divergence:.4f}, trust={result.trustworthiness:.4f})"
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        ax.text(0.02, 0.98, stat_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', color='black',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

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

if __name__ == "__main__":
    main()
