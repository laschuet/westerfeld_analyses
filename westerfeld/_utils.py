import numpy as np


def calc_iou(d1, d2):
    """Jaccard index (intersection over union) of two collections."""
    union = set(d1 + d2)
    intersect = np.intersect1d(d1, d2)
    return len(intersect) / len(union)
