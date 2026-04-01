from __future__ import annotations
from graph.settings import GRAPH_CREATOR_NAME

def get_graph_creator():
    if GRAPH_CREATOR_NAME == "correlation":
        from .correlation import CorrelationGraphCreationMethod

        return CorrelationGraphCreationMethod

    if GRAPH_CREATOR_NAME == "inference_glasso":
        from .inference import GlassoGraphCreationMethod

        return GlassoGraphCreationMethod

    raise ValueError(f"Unknown GRAPH_CREATOR_NAME: {GRAPH_CREATOR_NAME}")
