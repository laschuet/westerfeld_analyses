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

class GraphCreationMethod(ABC):
    @abstractmethod
    def create_network(
        self,
        df: pd.DataFrame,
        df_lookup: pd.DataFrame | None = None,
        df_relative: pd.DataFrame | None = None,
    ) -> nx.Graph:
        pass

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