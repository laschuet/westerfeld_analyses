# import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from scipy.stats import spearmanr, t as student_t
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler

from graph.niche import identify_generalists_or_specialists


class GraphCreationMethod(ABC):
    @abstractmethod
    def create_network(
        self,
        df: pd.DataFrame,
        df_lookup: pd.DataFrame | None = None,
        df_relative: pd.DataFrame | None = None,
    ) -> nx.Graph:
        pass


def _annotate_niche(G, df_lookup, df_relative):
    """Attach the lookup attributes plus a niche classification to each node."""
    nodes_attr = dict(G.nodes)
    for node in G.nodes:
        attributes = df_lookup.loc[node]
        spec_or_gen, _, _ = identify_generalists_or_specialists(
            df_relative[node].to_numpy()
        )
        attributes.loc["generalist_or_specialists"] = (
            spec_or_gen if spec_or_gen is not None else "None"
        )
        nodes_attr[node] = attributes
    nx.set_node_attributes(G, nodes_attr)


def _parse_kingdom_from_node(node):
    if isinstance(node, str) and ":" in node:
        return node.split(":", 1)[0]
    return None


def _edge_kingdom_type(node_a, node_b):
    kingdom_a = _parse_kingdom_from_node(node_a)
    kingdom_b = _parse_kingdom_from_node(node_b)
    if kingdom_a is None or kingdom_b is None:
        return None
    if kingdom_a == kingdom_b:
        return f"{kingdom_a}-{kingdom_a}"
    return "Fungi-Bacteria" if {kingdom_a, kingdom_b} == {"Fungi", "Bacteria"} else f"{kingdom_a}-{kingdom_b}"


def _annotate_kingdoms(G):
    nx.set_node_attributes(
        G,
        {node: _parse_kingdom_from_node(node) for node in G.nodes},
        "kingdom",
    )
    nx.set_edge_attributes(
        G,
        {
            (u, v): _edge_kingdom_type(u, v)
            for u, v in G.edges
        },
        "kingdom_edge",
    )


class CorrelationGraph(GraphCreationMethod):
    # Pairwise correlations are well-defined for any number of taxa, so no
    # prevalence filtering is required.
    min_prevalence = None

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

    def create_network(
        self,
        df: pd.DataFrame,
        df_lookup: pd.DataFrame | None = None,
        df_relative: pd.DataFrame | None = None,
    ) -> nx.Graph:
        corr_df, pval_df = self.calculate_correlations(df)

        G = nx.Graph()
        for i, taxon_i in enumerate(df.columns):
            for j, taxon_j in enumerate(df.columns):
                if (
                    i < j
                    and abs(corr_df.loc[taxon_i, taxon_j]) >= self.threshold
                    and pval_df.loc[taxon_i, taxon_j] <= 0.05
                ):
                    G.add_edge(
                        taxon_i,
                        taxon_j,
                        weight=corr_df.loc[taxon_i, taxon_j],
                        positive_association=corr_df.loc[taxon_i, taxon_j] > 0,
                    )

        _annotate_kingdoms(G)
        if df_lookup is not None and df_relative is not None:
            _annotate_niche(G, df_lookup, df_relative)
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
        df_relative: pd.DataFrame | None = None,
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
        if df_lookup is not None and df_relative is not None:
            _annotate_niche(G, df_lookup, df_relative)
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
