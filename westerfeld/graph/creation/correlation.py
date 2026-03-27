from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx

from graph.settings import CORRELATION_COEFFICIENT
from graph.utils import identifiy_generalists_or_specialists
from .utils import GraphCreationMethod
from graph.settings import CORRELATION_THRESHOLD

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


class CorrelationGraphCreationMethod(GraphCreationMethod):
    @classmethod
    def calculate_correlations(
        cls, df_t: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # calculate correlation
        if CORRELATION_COEFFICIENT == "spearman":
            corr, pval = spearmanr(df_t)
        elif CORRELATION_COEFFICIENT == "pearson":
            corr = np.ndarray(shape=(df_t.shape[1], df_t.shape[1]))
            pval = np.ndarray(shape=(df_t.shape[1], df_t.shape[1]))
            for spec1, i in zip(df_t, range(df_t.shape[1])):
                for spec2, j in zip(df_t, range(df_t.shape[1])):
                    if i == j:
                        corr[i][j] = 1
                        pval[i][j] = 0
                    elif j > i:
                        res = pearsonr(df_t[spec1], df_t[spec2])
                        corr[i][j] = res.statistic
                        corr[j][i] = res.statistic
                        pval[i][j] = res.pvalue
                        pval[j][i] = res.pvalue
        else:
            raise Exception("no valid `CORRELATION_COEFFICIENT`!")

        corr_df = pd.DataFrame(corr, index=df_t.columns, columns=df_t.columns)
        pval_df = pd.DataFrame(pval, index=df_t.columns, columns=df_t.columns)
        return corr_df, pval_df

    @classmethod
    def create_network(
        cls,
        df_t: pd.DataFrame,
        look_up_frame: pd.DataFrame,
        relative_data: pd.DataFrame,
    ) -> nx.Graph:
        # calculate correlations
        corr_df, pval_df = cls.calculate_correlations(df_t)

        # create network
        G = nx.Graph()
        for i, taxon_i in enumerate(df_t.columns):
            for j, taxon_j in enumerate(df_t.columns):
                if (
                    i < j
                    and abs(corr_df.loc[taxon_i, taxon_j]) >= CORRELATION_THRESHOLD
                    and pval_df.loc[taxon_i, taxon_j] <= 0.05
                ):
                    G.add_edge(
                        taxon_i,
                        taxon_j,
                        correlation=corr_df.loc[taxon_i, taxon_j],
                        positiv_correlation=corr_df.loc[taxon_i, taxon_j] > 0,
                    )

        nodesAttr = dict(G.nodes)
        nodes_bjs = pd.DataFrame({"mean_average_relative_abudances", "specOrGen", "bj"})
        for node in G.nodes:
            attributes = look_up_frame.loc[node]
            specOrGen, mean_average_relative_abudances, bj = (
                identifiy_generalists_or_specialists(relative_data[node].to_numpy())
            )
            attributes.loc["generalist_or_specialists"] = (
                specOrGen if specOrGen != None else "None"
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

        nodes_bjs.plot(
            kind="scatter", x="mean_average_relative_abudances", logx=True, y="bj"
        )
        plt.savefig("out/scatter_bj_mean_average.svg")
        nx.set_node_attributes(G, nodesAttr)
        return G

