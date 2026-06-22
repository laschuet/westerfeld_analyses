import numpy as np

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


# Beta diversity: between-sample community dissimilarity. Bray-Curtis is the
# standard dissimilarity for (relative) abundance data; the same metric backs
# the ordination so the visual and the tests describe the same geometry.
METRIC = "braycurtis"


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


def main():
    print("------------------")
    print("| BETA DIVERSITY |")
    print("------------------")

    crops = ["Winter wheat 1", "Winter wheat 2"]

    print("\n=== FUNGI ===")
    print("PERMANOVA:")
    print(permanova("Fungi", "Species", "Habitat", years=2019, crops=crops))
    print("\nPERMDISP:")
    print(permdisp("Fungi", "Species", "Habitat", years=2019, crops=crops, test="centroid"))

    print("\n=== BACTERIA ===")
    print("PERMANOVA:")
    print(permanova("Bacteria", "Genus", "Habitat", years=2019, crops=crops))
    print("\nPERMDISP:")
    print(permdisp("Bacteria", "Genus", "Habitat", years=2019, crops=crops, test="centroid"))


if __name__ == "__main__":
    main()
