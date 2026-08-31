# -*- coding: utf-8 -*-
"""
Slope entropy of a 1-D series (Cuesta-Frau 2019).
"""

import numpy as np
from typing import Optional, Sequence

from pysdkit.entropy._coarse_grain import as_1d, embed, shannon


def slope_entropy(
    y: np.ndarray,
    m: Optional[int] = 2,
    tau: int = 1,
    levels: Sequence[float] = (5.0, 45.0),
    normalize: bool = True,
) -> float:
    """
    Slope entropy (SlopEn) of a 1-D series.

    Consecutive samples (delay ``tau``) are turned into angles
    ``atan(x[t+tau] - x[t])`` in degrees and quantized with two thresholds
    (default 5 deg and 45 deg) into symbols in ``{-2, -1, 0, 1, 2}``.
    SlopEn is the Shannon entropy of ``m - 1`` consecutive slope symbols.

    Cuesta-Frau, D. (2019). Slope entropy: A new time series complexity
    estimator based on both symbolic patterns and amplitude information.
    Entropy, 21(12), 1167.

    :param y: 1-D input series.
    :param m: Embedding dimension (pattern uses ``m - 1`` slopes).
    :param tau: Delay used to form each slope.
    :param levels: Two increasing thresholds in ``(0, 90]`` degrees.
    :param normalize: If True, divide by ``log`` of the number of observed
                      patterns (Cuesta-Frau's default).
    :return: Slope entropy.
    """
    y = as_1d(y)
    if m < 2:
        raise ValueError("m must be an integer > 1.")
    if tau < 1:
        raise ValueError("tau must be a positive integer.")
    if len(y) < m * tau:
        raise ValueError("Signal is too short for the requested embedding.")

    gamma, delta = (float(levels[0]), float(levels[1]))
    if not (0.0 < gamma < delta <= 90.0):
        raise ValueError("levels must satisfy 0 < levels[0] < levels[1] <= 90.")

    angles = np.degrees(np.arctan(y[tau:] - y[:-tau]))
    symbols = np.zeros(angles.shape[0], dtype=int)
    symbols[(angles > gamma) & (angles <= delta)] = 1
    symbols[(angles >= -delta) & (angles < -gamma)] = -1
    symbols[angles > delta] = 2
    symbols[angles < -delta] = -2

    n_slopes = m - 1
    patterns = embed(symbols.astype(float), n_slopes, 1).astype(int)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    value = shannon(counts / counts.sum())
    if normalize:
        n_symbols = 2 * len(levels) + 1
        value /= np.log(n_symbols**n_slopes)
    return float(value)
