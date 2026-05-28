import numpy as np

from typing import Literal, Optional, Tuple


# Niche-breadth thresholds (https://doi.org/10.1093/femsec/fiw174).
# A taxon below the lower mean-relative-abundance cutoff is ignored; otherwise
# Bj below the specialist threshold marks a specialist and Bj above the
# generalist threshold marks a generalist.
MEAN_RELATIVE_ABUNDANCE_LOWER_THRESHOLD: float = 2e-5
B_VALUE_SPECIALIST_THRESHOLD: float = 1.5
B_VALUE_GENERALIST_THRESHOLD: float = 6.0


def identify_generalists_or_specialists(
    Pj: np.ndarray,
) -> Tuple[Optional[Literal["Specialist", "Generalist"]], np.ndarray, np.ndarray]:
    """
    Niche-breadth approach as described in https://doi.org/10.1093/femsec/fiw174.

    Returns the classification (or ``None``), the mean relative abundance, and
    the niche-breadth value Bj.
    """
    Pj = Pj / Pj.sum()
    mean_relative_abundance = Pj.mean()
    if mean_relative_abundance < MEAN_RELATIVE_ABUNDANCE_LOWER_THRESHOLD:
        return None, np.array([]), np.array([])
    Bj = 1 / (Pj**2).sum()
    if Bj > B_VALUE_GENERALIST_THRESHOLD:
        return "Generalist", mean_relative_abundance, Bj
    elif Bj < B_VALUE_SPECIALIST_THRESHOLD:
        return "Specialist", mean_relative_abundance, Bj
    return None, mean_relative_abundance, Bj
