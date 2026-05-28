import os

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


class CorrelationGraph(GraphCreationMethod):
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

        if df_lookup is not None and df_relative is not None:
            _annotate_niche(G, df_lookup, df_relative)
        return G


class GlassoGraph(GraphCreationMethod):
    def __init__(
        self,
        alphas=7,
        max_iter=500,
        inverse_variance_zero_threshold=1e-2,
        as_partial_correlation=False,
    ):
        self.alphas = alphas
        self.max_iter = max_iter
        self.inverse_variance_zero_threshold = inverse_variance_zero_threshold
        # If True, edge weight stores the partial correlation
        # (-theta_ij / sqrt(theta_ii * theta_jj), bounded in [-1, 1]) instead of
        # the raw precision entry theta_ij
        self.as_partial_correlation = as_partial_correlation

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
        model.fit(X)

        # Edges come from the sparse precision matrix: a non-zero off-diagonal
        # theta_ij encodes a direct (conditional) dependence between i and j. The
        # edge weight is the raw precision entry by default; set
        # `as_partial_correlation=True` to store the partial correlation instead.
        precision = model.precision_
        precision = np.where(
            np.abs(precision) < self.inverse_variance_zero_threshold, 0, precision
        )
        self.plot_covariance_matrix(precision, "glasso_precision")

        if self.as_partial_correlation:
            diag = np.sqrt(np.abs(np.diag(model.precision_)))
            weight_matrix = -precision / np.outer(diag, diag)
            np.fill_diagonal(weight_matrix, 1.0)
        else:
            weight_matrix = precision

        prec_df = pd.DataFrame(precision, index=df.columns, columns=df.columns)
        weight_df = pd.DataFrame(weight_matrix, index=df.columns, columns=df.columns)

        G = nx.Graph()
        for i, taxon_i in enumerate(df.columns):
            for j, taxon_j in enumerate(df.columns):
                if i < j and prec_df.iloc[i, j] != 0:
                    G.add_edge(
                        taxon_i,
                        taxon_j,
                        weight=weight_df.iloc[i, j],
                        positive_association=prec_df.iloc[i, j] < 0,
                    )

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
        path = f"out/covariance_matrix_{postfix}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
