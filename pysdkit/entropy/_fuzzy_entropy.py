# -*- coding: utf-8 -*-
"""
Fuzzy entropy of a 1-D series (Chen et al.).
"""

import numpy as np
from scipy.spatial.distance import pdist
from typing import Optional

from pysdkit.entropy._coarse_grain import as_1d, embed, coarse_grain


def fuzzy_entropy(
    y: np.ndarray,
    m: Optional[int] = 2,
    r: Optional[float] = 0.2,
    n: Optional[float] = 2.0,
    tau: int = 1,
) -> float:
    """
    Fuzzy entropy (FuzzEn) of a 1-D series.

    Embedding vectors are mean-centred.  Pairwise Chebyshev distances use a
    Gaussian-like membership ``exp(-(d / (r * std))^n)`` instead of a hard
    radius, and self-matches are excluded.

    Chen, W., Wang, Z., Xie, H., & Yu, W. (2007). Characterization of
    surface EMG signal based on fuzzy entropy. IEEE Transactions on Neural
    Systems and Rehabilitation Engineering, 15(2), 266-272.

    :param y: 1-D input series.
    :param m: Embedding dimension.
    :param r: Fuzzy width as a fraction of the series standard deviation.
    :param n: Exponent of the fuzzy membership (default 2).
    :param tau: Time delay between embedding coordinates.
    :return: ``ln(phi(m)) - ln(phi(m + 1))``.
    """
    y = as_1d(y)
    n_samples = len(y)
    if n_samples < m + 2:
        raise ValueError("Signal is too short for the requested embedding.")
    if m < 1 or m >= n_samples:
        raise ValueError("Embedding dimension must satisfy 1 <= m < N.")
    if tau < 1:
        raise ValueError("tau must be a positive integer.")

    sigma = np.std(y)
    width = r * sigma if sigma > 0 else 0.0
    phi_m = _fuzzy_phi(y, m, width, n, tau)
    phi_m1 = _fuzzy_phi(y, m + 1, width, n, tau)
    if phi_m <= 0.0 or phi_m1 <= 0.0 or not np.isfinite(phi_m * phi_m1):
        return float("nan")
    return float(np.log(phi_m) - np.log(phi_m1))


def _fuzzy_phi(
    y: np.ndarray, m: int, width: float, n: float, tau: int
) -> float:
    templates = embed(y, m, tau)
    templates = templates - templates.mean(axis=1, keepdims=True)
    n_vec = templates.shape[0]
    if n_vec < 2:
        return float("nan")
    dist = pdist(templates, metric="chebyshev")
    if width <= 0.0:
        membership = (dist == 0.0).astype(float)
    else:
        membership = np.exp(-np.power(dist / width, n))
    return float(2.0 * np.sum(membership) / (n_vec * (n_vec - 1)))


def multiscale_fuzzy_entropy(
    y: np.ndarray,
    m: Optional[int] = 2,
    r: Optional[float] = 0.2,
    n: Optional[float] = 2.0,
    scale: Optional[int] = 3,
    tau: int = 1,
) -> np.ndarray:
    """
    Costa coarse-graining followed by fuzzy entropy at each scale.

    :param y: 1-D input series.
    :param m: Embedding dimension.
    :param r: Fuzzy width as a fraction of the original-scale std.
    :param n: Fuzzy exponent.
    :param scale: Highest scale factor (returns length ``scale``).
    :param tau: Embedding delay.
    :return: Fuzzy entropy at scales ``1 .. scale``.
    """
    y = as_1d(y)
    if scale < 1:
        raise ValueError("scale must be an integer >= 1.")
    values = []
    for s in range(1, scale + 1):
        grain = coarse_grain(y, s)
        values.append(fuzzy_entropy(grain, m=m, r=r, n=n, tau=tau))
    return np.asarray(values, dtype=float)
