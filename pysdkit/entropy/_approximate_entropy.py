# -*- coding: utf-8 -*-
"""
Approximate entropy of a 1-D series (Pincus 1991).
"""

import numpy as np
from typing import Optional

from pysdkit.entropy._coarse_grain import as_1d, embed


def approximate_entropy(
    y: np.ndarray,
    m: Optional[int] = 2,
    r: Optional[float] = 0.2,
    tau: int = 1,
) -> float:
    """
    Approximate entropy (ApEn) of a 1-D series.

    Unlike sample entropy, each template is compared with **all** templates
    of the same length, including itself.  The radius is ``r * std(y)``.

    Pincus, S. M. (1991). Approximate entropy as a measure of system
    complexity. Proceedings of the National Academy of Sciences, 88(6),
    2297-2301.

    :param y: 1-D input series.
    :param m: Embedding dimension (must be < ``len(y)``).
    :param r: Tolerance as a fraction of the series standard deviation.
    :param tau: Time delay between embedding coordinates.
    :return: ApEn = Phi(m) - Phi(m + 1).
    """
    y = as_1d(y)
    n = len(y)
    if n < 2:
        raise ValueError("Signal must have at least two points.")
    if m < 1 or m >= n:
        raise ValueError("Embedding dimension must satisfy 1 <= m < N.")
    if tau < 1:
        raise ValueError("tau must be a positive integer.")

    sigma = np.std(y)
    radius = r * sigma if sigma > 0 else 0.0
    phi_m = _phi(y, m, radius, tau)
    phi_m1 = _phi(y, m + 1, radius, tau)
    return float(phi_m - phi_m1)


def _phi(y: np.ndarray, m: int, radius: float, tau: int) -> float:
    """Mean log of the fraction of Chebyshev matches, including self-matches."""
    templates = embed(y, m, tau)
    n_vec = templates.shape[0]
    delta = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
    counts = np.sum(delta <= radius, axis=1)
    frac = counts / n_vec
    frac = np.maximum(frac, np.finfo(float).tiny)
    return float(np.mean(np.log(frac)))
