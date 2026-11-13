# import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from scipy.stats import spearmanr, t as student_t
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler

from graph.niche import _annotate_niche
from _utils import _annotate_kingdoms

class GraphCreationMethod(ABC):
    @abstractmethod
    def create_network(
        self,
        df: pd.DataFrame,
        df_lookup: pd.DataFrame | None = None,
        niche: pd.DataFrame | None = None,
    ) -> nx.Graph:
        pass

class CorrelationGraph(GraphCreationMethod):
    # Only taxa present in at least 50 % of the samples are kept (Goberna &
    # Verdú 2022). Rare taxa that co-occur in one or two samples reach |rho| ~ 1
    # by chance: without the filter, ~20 % of the edges also appear when each
    # taxon is shuffled independently (see `chance_edges`); with it, < 1 %.
    min_prevalence = 0.5
    # Sample-index levels whose group means are removed from each taxon before
    # correlating, e.g. ("Crop", "Treatment"): edges then show co-variation
    # within the same crop and management instead of shared responses to them.
    # None pools all samples.
    design_factors = None

    def __init__(self, coefficient="spearman", threshold=0.68):
        self.coefficient = coefficient
        self.threshold = threshold

    def calculate_correlations(self, df):
        if self.coefficient == "spearman":
            corr, pval = spearmanr(df)
        elif self.coefficient == "pearson":
            X = df.to_numpy(dtype=float)
            n = X.shape[0]
            corr = np.corrcoef(X, rowvar=False)
            with np.errstate(divide="ignore", invalid="ignore"):
                stat = corr * np.sqrt((n - 2) / (1.0 - corr**2))
            pval = 2.0 * student_t.sf(np.abs(stat), n - 2)
            np.fill_diagonal(corr, 1.0)
            np.fill_diagonal(pval, 0.0)
        else:
            raise ValueError(
                f"Unknown correlation coefficient: {self.coefficient} "
                "(expected 'spearman' or 'pearson')"
            )

        corr_df = pd.DataFrame(corr, index=df.columns, columns=df.columns)
        pval_df = pd.DataFrame(pval, index=df.columns, columns=df.columns)
        return corr_df, pval_df

    def edges(self, df):
        """
        Column positions (i, j) with i < j and correlation of all taxon pairs
        with |correlation| >= threshold and p-value <= 0.05. With ~30 samples,
        |rho| >= 0.68 already means p < 0.001, so the p-value only matters for
        very few samples (n < ~10).
        """
        corr_df, pval_df = self.calculate_correlations(df)
        corr, pval = corr_df.to_numpy(), pval_df.to_numpy()
        i, j = np.triu_indices(len(corr), k=1)
        keep = (np.abs(corr[i, j]) >= self.threshold) & (pval[i, j] <= 0.05)
        return i[keep], j[keep], corr[i[keep], j[keep]]

    def chance_edges(self, df, n_shuffles=10, seed=42):
        """
        Expected number of edges without any association: the values of each
        taxon are shuffled independently across samples (keeping each taxon's
        distribution), and the edges of the shuffled data are counted. Returns
        the mean over `n_shuffles` shuffles.
        """
        rng = np.random.default_rng(seed)
        counts = []
        for _ in range(n_shuffles):
            shuffled = pd.DataFrame(
                {taxon: rng.permutation(df[taxon].to_numpy()) for taxon in df.columns}
            )
            counts.append(len(self.edges(shuffled)[0]))
        return float(np.mean(counts))

    def create_network(
        self,
        df: pd.DataFrame,
        df_lookup: pd.DataFrame | None = None,
        niche: pd.DataFrame | None = None,
    ) -> nx.Graph:
        if self.design_factors:
            df = df - df.groupby(level=list(self.design_factors)).transform("mean")
        G = nx.Graph()
        taxa = df.columns
        for i, j, rho in zip(*self.edges(df)):
            G.add_edge(taxa[i], taxa[j], weight=rho, positive_association=rho > 0)
        G.graph["chance_edges"] = self.chance_edges(df)

        _annotate_kingdoms(G)
        if df_lookup is not None and niche is not None:
            _annotate_niche(G, df_lookup, niche)
        return G

class GlassoGraph(GraphCreationMethod):
    # Graphical Lasso inverts the taxa covariance, which is singular when there
    # are far more taxa than samples (m >> n). The empirical covariance has rank
    # <= n, so with ~1000 taxa and ~30 samples no L1 penalty can recover a usable
    # precision matrix: the solver divides by collapsed diagonal entries (a wall
    # of RuntimeWarnings) and ultimately raises a FloatingPointError. Restricting
    # to prevalent taxa brings m/n down to a regime the solver can handle.
    min_prevalence = 0.8

    def __init__(
        self, alphas=None, max_iter=1000, inverse_variance_zero_threshold=1e-2
    ):
        # An explicit, strictly positive alpha grid. Passing an integer instead
        # lets GraphicalLassoCV auto-build a grid that reaches near-zero
        # penalties, which try to invert the (near-)singular covariance and blow
        # up, so a positive floor keeps every candidate penalty regularizing.
        self.alphas = alphas if alphas is not None else [0.4, 0.6, 0.8, 1.0]
        self.max_iter = max_iter
        self.inverse_variance_zero_threshold = inverse_variance_zero_threshold

    def create_network(
        self,
        df: pd.DataFrame,
        df_lookup: pd.DataFrame | None = None,
        niche: pd.DataFrame | None = None,
    ) -> nx.Graph:
        # Graphical Lasso
        # (https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html)
        self.plot_covariance_matrix(np.cov(df.T.values, bias=True), "estimated")
        self.plot_covariance_matrix(np.corrcoef(df.T.values), "correlation")

        # Standardise each taxon over samples. GraphicalLassoCV.fit expects raw
        # samples x features and computes the empirical covariance itself.
        X = StandardScaler().fit_transform(df.values)
        model = GraphicalLassoCV(
            alphas=self.alphas, max_iter=self.max_iter, verbose=True, n_jobs=-1
        )
        # On a high-dimensional covariance, the CV grid search briefly probes
        # penalties that are still mildly ill-conditioned, emitting transient
        # divide-by-zero / invalid-value RuntimeWarnings even when the final fit
        # is fine. Silence just those; a genuinely unusable result is caught
        # below by the precision-diagonal guard.
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=RuntimeWarning)
        #     model.fit(X)
        model.fit(X)

        # Edges come from the sparse precision matrix: a non-zero off-diagonal
        # theta_ij encodes a direct (conditional) dependence between i and j.
        # Keep the entries whose magnitude clears the threshold.
        theta = model.precision_
        sparse_theta = np.where(
            np.abs(theta) < self.inverse_variance_zero_threshold, 0, theta
        )
        self.plot_covariance_matrix(sparse_theta, "glasso_precision")

        # Edge weight is the partial correlation -theta_ij / sqrt(theta_ii *
        # theta_jj): bounded in [-1, 1] and signed like the association (positive
        # for a positive co-occurrence). Normalize by the precision diagonal,
        # i.e. the true conditional variances, which the threshold never touches.
        # A well-conditioned precision has a strictly positive diagonal; guard
        # against any non-positive entry so the division can't emit NaN/inf.
        diag_values = np.diag(theta)
        if np.any(diag_values <= 0):
            raise FloatingPointError(
                "Graphical Lasso returned a non-positive precision diagonal; "
                "the covariance is too ill-conditioned (try a higher "
                "min_prevalence or fewer taxa)."
            )
        diag = np.sqrt(diag_values)
        partial_corr = -theta / np.outer(diag, diag)

        sparse_df = pd.DataFrame(sparse_theta, index=df.columns, columns=df.columns)
        weight_df = pd.DataFrame(partial_corr, index=df.columns, columns=df.columns)

        G = nx.Graph()
        for i, taxon_i in enumerate(df.columns):
            for j, taxon_j in enumerate(df.columns):
                if i < j and sparse_df.iloc[i, j] != 0:
                    weight = weight_df.iloc[i, j]
                    G.add_edge(
                        taxon_i,
                        taxon_j,
                        weight=weight,
                        positive_association=weight > 0,
                    )

        _annotate_kingdoms(G)
        if df_lookup is not None and niche is not None:
            _annotate_niche(G, df_lookup, niche)
        return G

    def plot_covariance_matrix(self, covariance_matrix, postfix=""):
        df = pd.DataFrame(covariance_matrix).astype(float)

        f = plt.figure(figsize=(12, 10))
        im = plt.matshow(df, fignum=f.number, cmap="Blues")
        plt.xticks(
            range(df.select_dtypes(["number"]).shape[1]),
            df.select_dtypes(["number"]).columns,
            fontsize=14,
            rotation=45,
        )
        plt.yticks(
            range(df.select_dtypes(["number"]).shape[1]),
            df.select_dtypes(["number"]).columns,
            fontsize=14,
        )
        cb = plt.colorbar(im)
        cb.ax.tick_params(labelsize=14)
        path = f"covariance_matrix_{postfix}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")