import numpy as np

from typing import Literal, Optional, Tuple

from graph.settings import (
    B_VALUE_GENERALIST_THRESHOLD,
    B_VALUE_SPECIALIST_THRESHOLD,
    MEAN_RELATIVE_ABUNDANCES_LOWER_THRESHOLD,
)


def identifiy_generalists_or_specialists(
    Pj: np.ndarray,
) -> Tuple[Optional[Literal["Specialist", "Generalist"]], np.ndarray, np.ndarray]:
    """
    Implementation of the niche breadth approach as described in https://doi.org/10.1093/femsec/fiw174
    """
    Pj = Pj / Pj.sum()
    mean_average_relative_abudances = Pj.mean()
    if mean_average_relative_abudances < MEAN_RELATIVE_ABUNDANCES_LOWER_THRESHOLD:
        return None, np.array([]), np.array([])
    Bj = 1 / (Pj**2).sum()
    if Bj > B_VALUE_GENERALIST_THRESHOLD:
        return "Generalist", mean_average_relative_abudances, Bj
    elif Bj < B_VALUE_SPECIALIST_THRESHOLD:
        return "Specialist", mean_average_relative_abudances, Bj
    return None, mean_average_relative_abudances, Bj
