import pandas as pd
import networkx as nx
from abc import ABC, abstractmethod

class GraphCreationMethod(ABC):
    @classmethod
    @abstractmethod
    def create_network(
        cls,
        df_t: pd.DataFrame,
        look_up_frame: pd.DataFrame,
        relative_data: pd.DataFrame,
    ) -> nx.Graph:
        pass

