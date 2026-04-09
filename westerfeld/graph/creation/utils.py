import networkx as nx
import pandas as pd

from abc import ABC, abstractmethod


class GraphCreationMethod(ABC):
    @classmethod
    @abstractmethod
    def create_network(
        cls,
        df: pd.DataFrame,
        look_up_frame: pd.DataFrame | None = None,
        relative_data: pd.DataFrame | None = None,
    ) -> nx.Graph:
        pass
