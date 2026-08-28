# -*- coding: utf-8 -*-
"""
Distribution entropy of a 1-D series (Li et al. 2015).
"""

import numpy as np
from scipy.spatial.distance import pdist
from typing import Optional, Union

from pysdkit.entropy._coarse_grain import as_1d, embed, shannon


def distribution_entropy(
    y: np.ndarray,
    m: Optional[int] = 2,
    tau: int = 1,
    bins: Union[int, str] = "sturges",
    normalize: bool = True,
) -> float:
    """
    Distribution entropy (DistEn) of a 1-D series.

    All pairwise Chebyshev distances of the delay-embedded vectors are
    histogrammed; DistEn is the Shannon entropy of that histogram.  The
    measure is often more stable than sample entropy on short records.

    Li, P., Liu, C., Li, K., Zheng, D., Liu, C., & Hou, Y. (2015).
    Assessing the complexity of short-term heartbeat interval series by
    distribution entropy. Medical & Biological Engineering & Computing,
    53(1), 77-87.

    :param y: 1-D input series.
    :param m: Embedding dimension.
    :param tau: Time delay between embedding coordinates.
    :param bins: Histogram bin count, or ``'sturges'`` / ``'sqrt'`` /
                 ``'rice'``.
    :param normalize: If True, divide by ``log(n_bins)``.
    :return: Distribution entropy.
    """
    y = as_1d(y)
    n = len(y)
    if m < 1 or m >= n:
        raise ValueError("Embedding dimension must satisfy 1 <= m < N.")
    if tau < 1:
        raise ValueError("tau must be a positive integer.")

    templates = embed(y, m, tau)
    if templates.shape[0] < 2:
        raise ValueError("Not enough embedding vectors to form distances.")
    distances = pdist(templates, metric="chebyshev")
    n_bins = _n_bins(len(distances), bins)
    counts, _ = np.histogram(distances, bins=n_bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    value = shannon(counts / total)
    if normalize and n_bins > 1:
        value /= np.log(n_bins)
    return float(value)


def _n_bins(n_dist: int, bins: Union[int, str]) -> int:
    if isinstance(bins, int):
        if bins < 2:
            raise ValueError("bins must be an integer > 1.")
        return bins
    name = str(bins).lower()
    if name == "sturges":
        n = int(np.ceil(np.log2(n_dist) + 1))
    elif name == "sqrt":
        n = int(np.ceil(np.sqrt(n_dist)))
    elif name == "rice":
        n = int(np.ceil(2.0 * n_dist ** (1.0 / 3.0)))
    else:
        raise ValueError("bins must be an int or 'sturges' / 'sqrt' / 'rice'.")
    return max(n, 2)
