import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.covariance import GraphicalLassoCV
from sklearn.discriminant_analysis import StandardScaler

from .utils import GraphCreationMethod
from graph.settings import (
    GLASSO_ALPHAS,
    GLASSO_MAX_ITER,
    INVERSE_VARIANCE_ZERO_THRESHOLD,
)
from graph.utils import identifiy_generalists_or_specialists


class GlassoGraphCreationMethod(GraphCreationMethod):
    @classmethod
    def calculate_covariance(cls, df: pd.DataFrame) -> np.ndarray:
        return np.cov(df)

    @classmethod
    def create_network(
        cls,
        df: pd.DataFrame,
        df_lookup: pd.DataFrame | None = None,
        df_relative: pd.DataFrame | None = None,
    ) -> nx.Graph:
        # Lasso method (https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html)
        cov_est = np.cov(df.T.values, bias=True)
        cov_corr = np.corrcoef(df.T.values)
        cls.plot_covariance_matrix(cov_est, "estimated")
        cls.plot_covariance_matrix(cov_corr, "correlation")

        X = StandardScaler().fit_transform(cov_est)

        model = GraphicalLassoCV(
            alphas=GLASSO_ALPHAS, max_iter=GLASSO_MAX_ITER, verbose=True
        )
        model.fit(X)

        cov = model.covariance_

        cov = np.where(np.abs(cov) < INVERSE_VARIANCE_ZERO_THRESHOLD, 0, cov)
        cls.plot_covariance_matrix(cov, "glasso_estimated")

        cov_df = pd.DataFrame(cov, index=df.columns, columns=df.columns)

        G = nx.Graph()
        for i, taxon_i in enumerate(df.columns):
            for j, taxon_j in enumerate(df.columns):
                if i < j and abs(cov_df.loc[taxon_i, taxon_j]) != 0:
                    G.add_edge(
                        taxon_i,
                        taxon_j,
                        correlation=cov_df.loc[taxon_i, taxon_j],
                        positiv_correlation=cov_df.loc[taxon_i, taxon_j] > 0,
                    )

        nodes_attr = dict(G.nodes)
        nodes_bjs = pd.DataFrame({"mean_average_relative_abudances", "spec_or_gen", "bj"})
        if df_lookup is not None and df_relative is not None:
            for node in G.nodes:
                attributes = df_lookup.loc[node]
                spec_or_gen, mean_average_relative_abudances, bj = (
                    identifiy_generalists_or_specialists(df_relative[node].to_numpy())
                )
                attributes.loc["generalist_or_specialists"] = (
                    spec_or_gen if spec_or_gen is not None else "None"
                )
                nodes_attr[node] = attributes
                nodes_bjs = pd.DataFrame(
                    columns=["mean_average_relative_abudances", "spec_or_gen", "bj"]
                )
                for node in G.nodes:
                    attributes = df_lookup.loc[node]
                    specOrGen, mean_average_relative_abudances, bj = (
                        identifiy_generalists_or_specialists(
                            df_relative[node].to_numpy()
                        )
                    )
                    attributes.loc["generalist_or_specialists"] = (
                        specOrGen if specOrGen is not None else "None"
                    )
                    nodes_attr[node] = attributes
                    nodes_bjs.loc[node] = [
                        mean_average_relative_abudances,
                        specOrGen if specOrGen is not None else "None",
                        bj,
                    ]

        nx.set_node_attributes(G, nodes_attr)
        return G

    @classmethod
    def plot_covariance_matrix(cls, covariance_matrix, postfix=""):
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
