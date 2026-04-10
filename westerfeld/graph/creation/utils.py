import networkx as nx
import pandas as pd

from abc import ABC, abstractmethod


class GraphCreationMethod(ABC):
    @classmethod
    @abstractmethod
    def create_network(
        cls,
        df: pd.DataFrame,
        df_lookup: pd.DataFrame | None = None,
        df_relative: pd.DataFrame | None = None,
    ) -> nx.Graph:
        pass
