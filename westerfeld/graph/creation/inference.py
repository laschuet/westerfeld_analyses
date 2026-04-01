from sklearn.discriminant_analysis import StandardScaler

from graph.settings import GLASSO_ALPHAS, GLASSO_MAX_ITER, INVERSE_VARIANCE_ZERO_THRESHOLD
from graph.utils import identifiy_generalists_or_specialists

from .utils import GraphCreationMethod
from sklearn.covariance import GraphicalLasso
from sklearn.covariance import GraphicalLassoCV

from typing import Tuple

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class GlassoGraphCreationMethod(GraphCreationMethod):
    @classmethod
    def calculate_covariance(
        cls, df_t: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cov = np.cov(df_t)
        return cov

    @classmethod
    def create_network(
        cls,
        df_t: pd.DataFrame,
        look_up_frame: pd.DataFrame,
        relative_data: pd.DataFrame,
    ) -> nx.Graph:
        # lasso method (https://scikit-learn.org/stable/modules/generated/sklearn.covariance.GraphicalLasso.html)
        df_t

        cov_est = np.cov(df_t.T.values, bias=True)
        cov_corr = np.corrcoef(df_t.T.values)
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

        cov_df = pd.DataFrame(cov, index=df_t.columns, columns=df_t.columns)

        G = nx.Graph()
        for i, taxon_i in enumerate(df_t.columns):
            for j, taxon_j in enumerate(df_t.columns):
                if i < j and abs(cov_df.loc[taxon_i, taxon_j]) != 0:
                    G.add_edge(
                        taxon_i,
                        taxon_j,
                        correlation=cov_df.loc[taxon_i, taxon_j],
                        positiv_correlation=cov_df.loc[taxon_i, taxon_j] > 0,
                    )

        nodesAttr = dict(G.nodes)
        nodes_bjs = pd.DataFrame({"mean_average_relative_abudances", "specOrGen", "bj"})
        for node in G.nodes:
            attributes = look_up_frame.loc[node]
            specOrGen, mean_average_relative_abudances, bj = (
                identifiy_generalists_or_specialists(relative_data[node].to_numpy())
            )
            attributes.loc["generalist_or_specialists"] = (
                specOrGen if specOrGen is not None else "None"
            )
            nodesAttr[node] = attributes
            nodes_bjs = pd.DataFrame(
                columns=["mean_average_relative_abudances", "specOrGen", "bj"]
            )
            for node in G.nodes:
                attributes = look_up_frame.loc[node]
                specOrGen, mean_average_relative_abudances, bj = (
                    identifiy_generalists_or_specialists(relative_data[node].to_numpy())
                )
                attributes.loc["generalist_or_specialists"] = (
                    specOrGen if specOrGen is not None else "None"
                )
                nodesAttr[node] = attributes
                nodes_bjs.loc[node] = [
                    mean_average_relative_abudances,
                    specOrGen if specOrGen is not None else "None",
                    bj,
                ]

        nx.set_node_attributes(G, nodesAttr)
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

