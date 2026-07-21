import numpy as np
from typing import Literal, Optional, Tuple


# Niche-breadth thresholds (https://doi.org/10.1093/femsec/fiw174).
# A taxon below the lower mean-relative-abundance cutoff is ignored; otherwise
# Bj below the specialist threshold marks a specialist and Bj above the
# generalist threshold marks a generalist.
HABITAT_THRESHOLDS = {
    "Field_Soil": {
        "mean_rel_abundance": 2e-5,
        "specialist": 1.5,
        "generalist": 27.0,  
    },
    "Rhizosphere": {
        "mean_rel_abundance": 2e-5,
        "specialist": 1.5,
        "generalist": 25.0, 
    }
}


def identify_generalists_or_specialists(
    Pj: np.ndarray, habitat_type: str
) -> Tuple[Optional[Literal["Specialist", "Generalist"]], np.ndarray, np.ndarray]:
    """
        Niche-breadth approach as described in https://doi.org/10.1093/femsec/fiw174.

        Returns the classification (or ``None``), the mean relative abundance, and
        the niche-breadth value Bj.
    """
    if habitat_type not in HABITAT_THRESHOLDS:
        raise ValueError(f"Unbekannter Habitat-Typ: {habitat_type}. Bitte 'FS' oder 'RH' verwenden.")

    # 2. Thresholds laden
    thresholds = HABITAT_THRESHOLDS[habitat_type]
    gen_thresh = thresholds["generalist"]
    spec_thresh = thresholds["specialist"]
    mean_thresh = thresholds["mean_rel_abundance"]

    Pj = Pj / Pj.sum()
    mean_relative_abundance = Pj.mean()
    if mean_relative_abundance < mean_thresh:
        return None, np.array([]), np.array([])
    Bj = 1 / (Pj**2).sum()
    if Bj > gen_thresh:
        return "Generalist", mean_relative_abundance, Bj
    elif Bj < spec_thresh:
        return "Specialist", mean_relative_abundance, Bj
    return None, mean_relative_abundance, Bj
