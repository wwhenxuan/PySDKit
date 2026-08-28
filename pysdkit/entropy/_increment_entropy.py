# -*- coding: utf-8 -*-
"""
Increment entropy of a 1-D series (Liu et al. 2016).
"""

import numpy as np
from typing import Optional

from pysdkit.entropy._coarse_grain import as_1d, embed, shannon


def increment_entropy(
    y: np.ndarray,
    m: Optional[int] = 2,
    tau: int = 1,
    r: Optional[int] = 4,
    normalize: bool = False,
) -> float:
    """
    Increment entropy (IncrEn) of a 1-D series.

    First differences are coded by sign and a quantized magnitude (resolution
    ``r``), then delay-embedded.  IncrEn is the Shannon entropy of those
    increment words; it is sensitive to local jumps.

    Liu, X., Jiang, A., Xu, N., & Xue, J. (2016). Increment entropy as a
    measure of complexity for time series. Entropy, 18(1), 22.

    :param y: 1-D input series.
    :param m: Word length (number of consecutive increments).
    :param tau: Delay between increments in a word.
    :param r: Magnitude quantization levels (positive integer).
    :param normalize: If True, divide by ``m - 1``.
    :return: Increment entropy (nats unless normalized).
    """
    y = as_1d(y)
    if len(y) < 3:
        raise ValueError("Signal must have at least three points.")
    if m < 1:
        raise ValueError("m must be a positive integer.")
    if tau < 1:
        raise ValueError("tau must be a positive integer.")
    if r < 1:
        raise ValueError("r must be a positive integer.")

    increments = np.diff(y)
    sigma = np.std(increments)
    signs = np.sign(increments)
    if sigma == 0.0:
        quant = np.zeros_like(increments)
    else:
        quant = np.minimum(r, np.floor(np.abs(increments) * r / sigma))
    quant[increments == 0.0] = 0.0
    words = embed(signs * quant, m, tau)
    _, counts = np.unique(words, axis=0, return_counts=True)
    value = shannon(counts / counts.sum())
    if normalize and m > 1:
        value /= m - 1
    return float(value)
