import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr

from .utils import GraphCreationMethod
from graph.utils import identifiy_generalists_or_specialists


class CorrelationGraph(GraphCreationMethod):
    def __init__(self, coefficient="spearman", threshold=0.68):
        self.coefficient = coefficient
        self.threshold = threshold

    def calculate_correlations(self, df):
        if self.coefficient == "spearman":
            corr, pval = spearmanr(df)
        elif self.coefficient == "pearson":
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
                        correlation=corr_df.loc[taxon_i, taxon_j],
                        positiv_correlation=corr_df.loc[taxon_i, taxon_j] > 0,
                    )

        nodes_attr = dict(G.nodes)
        if df_lookup is not None and df_relative is not None:
            nodes_bjs = pd.DataFrame(
                columns=["mean_average_relative_abudances", "spec_or_gen", "bj"]
            )
            for node in G.nodes:
                attributes = df_lookup.loc[node]
                spec_or_gen, mean_average_relative_abudances, bj = (
                    identifiy_generalists_or_specialists(df_relative[node].to_numpy())
                )
                spec_or_gen = spec_or_gen if spec_or_gen is not None else "None"
                attributes.loc["generalist_or_specialists"] = spec_or_gen
                nodes_attr[node] = attributes
                nodes_bjs.loc[node] = [
                    mean_average_relative_abudances,
                    spec_or_gen,
                    bj,
                ]
            nodes_bjs.plot(
                kind="scatter", x="mean_average_relative_abudances", logx=True, y="bj"
            )
            plt.savefig("out/scatter_bj_mean_average.svg")

        nx.set_node_attributes(G, nodes_attr)
        return G
