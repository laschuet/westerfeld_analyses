import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from typing import Tuple

from .utils import GraphCreationMethod
from graph.settings import CORRELATION_COEFFICIENT, CORRELATION_THRESHOLD
from graph.utils import identifiy_generalists_or_specialists


class CorrelationGraphCreationMethod(GraphCreationMethod):
    @classmethod
    def calculate_correlations(
        cls, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if CORRELATION_COEFFICIENT == "spearman":
            corr, pval = spearmanr(df)
        elif CORRELATION_COEFFICIENT == "pearson":
            corr = np.ndarray(shape=(df.shape[1], df.shape[1]))
            pval = np.ndarray(shape=(df.shape[1], df.shape[1]))
            for spec1, i in zip(df, range(df.shape[1])):
                for spec2, j in zip(df, range(df.shape[1])):
                    if i == j:
                        corr[i][j] = 1
                        pval[i][j] = 0
                    elif j > i:
                        res = pearsonr(df[spec1], df[spec2])
                        corr[i][j] = res.statistic
                        corr[j][i] = res.statistic
                        pval[i][j] = res.pvalue
                        pval[j][i] = res.pvalue
        else:
            raise Exception("no valid `CORRELATION_COEFFICIENT`!")

        corr_df = pd.DataFrame(corr, index=df.columns, columns=df.columns)
        pval_df = pd.DataFrame(pval, index=df.columns, columns=df.columns)
        return corr_df, pval_df

    @classmethod
    def create_network(
        cls,
        df: pd.DataFrame,
        df_lookup: pd.DataFrame | None = None,
        df_relative: pd.DataFrame | None = None,
    ) -> nx.Graph:
        corr_df, pval_df = cls.calculate_correlations(df)

        G = nx.Graph()
        for i, taxon_i in enumerate(df.columns):
            for j, taxon_j in enumerate(df.columns):
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
            nodes_bjs.plot(
                kind="scatter", x="mean_average_relative_abudances", logx=True, y="bj"
            )
            plt.savefig("out/scatter_bj_mean_average.svg")

        nx.set_node_attributes(G, nodes_attr)
        return G
