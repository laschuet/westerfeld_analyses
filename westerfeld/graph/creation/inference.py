import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.covariance import GraphicalLassoCV
from sklearn.discriminant_analysis import StandardScaler

from .utils import GraphCreationMethod
from graph.utils import identifiy_generalists_or_specialists


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

    def calculate_covariance(self, df):
        return np.cov(df)

    def create_network(
        self,
        df: pd.DataFrame,
        df_lookup: pd.DataFrame | None = None,
        df_relative: pd.DataFrame | None = None,
    ) -> nx.Graph:
        # Lasso method (https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html)
        cov_est = np.cov(df.T.values, bias=True)
        cov_corr = np.corrcoef(df.T.values)
        self.plot_covariance_matrix(cov_est, "estimated")
        self.plot_covariance_matrix(cov_corr, "correlation")

        # Standardise each taxon over samples
        X = StandardScaler().fit_transform(df.values)
        # GraphicalLassoCV.fit expects raw samples x features and
        # computes the empirical covariance itself
        model = GraphicalLassoCV(
            alphas=self.alphas, max_iter=self.max_iter, verbose=True, n_jobs=-1
        )
        model.fit(X)

        # Edges come from the sparse precision matrix:
        # A non-zero off-diagonal theta_ij encodes a direct (conditional)
        # dependence between i and j. The edge weight is the raw precision
        # entry by default. Set `as_partial_correlation=True` in the constructor
        # to store the partial correlation instead.
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

        nodes_attr = dict(G.nodes)
        if df_lookup is not None and df_relative is not None:
            for node in G.nodes:
                attributes = df_lookup.loc[node]
                spec_or_gen, _, _ = identifiy_generalists_or_specialists(
                    df_relative[node].to_numpy()
                )
                attributes.loc["generalist_or_specialists"] = (
                    spec_or_gen if spec_or_gen is not None else "None"
                )
                nodes_attr[node] = attributes

        nx.set_node_attributes(G, nodes_attr)
        return G

    def plot_covariance_matrix(self, covariance_matrix, postfix=""):
        df_cov = pd.DataFrame(covariance_matrix)

        f = plt.figure(figsize=(12, 10))
        df = df_cov.astype(float)
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
        plt.savefig(
            f"out/covariance_matrix_{postfix}.png", dpi=300, bbox_inches="tight"
        )
