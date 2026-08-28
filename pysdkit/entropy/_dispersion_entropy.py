# -*- coding: utf-8 -*-
"""
Dispersion entropy of a 1-D series (Rostaghi and Azami 2016).
"""

import numpy as np
from scipy.special import ndtr
from typing import Optional

from pysdkit.entropy._coarse_grain import as_1d, embed, coarse_grain, shannon


def dispersion_entropy(
    y: np.ndarray,
    m: Optional[int] = 2,
    c: Optional[int] = 3,
    tau: int = 1,
    normalize: bool = False,
) -> float:
    """
    Dispersion entropy (DispEn) of a 1-D series.

    The series is mapped through a normal CDF, quantized to ``c`` classes,
    delay-embedded, and the Shannon entropy of the pattern frequencies is
    returned.  Amplitude information is kept, unlike permutation entropy.

    Rostaghi, M., & Azami, H. (2016). Dispersion entropy: A measure for
    time-series analysis. IEEE Signal Processing Letters, 23(5), 610-614.

    :param y: 1-D input series.
    :param m: Embedding dimension.
    :param c: Number of classes (symbols), integer > 1.
    :param tau: Time delay between embedding coordinates.
    :param normalize: If True, divide by ``log(c ** m)``.
    :return: Dispersion entropy (nats, unless normalized).
    """
    y = as_1d(y)
    n = len(y)
    if c < 2:
        raise ValueError("c must be an integer > 1.")
    if m < 1 or m >= n:
        raise ValueError("Embedding dimension must satisfy 1 <= m < N.")
    if tau < 1:
        raise ValueError("tau must be a positive integer.")

    sigma = np.std(y)
    if sigma == 0.0:
        mapped = np.full(n, 0.5)
    else:
        mapped = ndtr((y - np.mean(y)) / sigma)
    symbols = np.clip(np.floor(mapped * c).astype(int), 0, c - 1)
    patterns = embed(symbols.astype(float), m, tau).astype(int)
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    prob = counts / counts.sum()
    value = shannon(prob)
    if normalize:
        value /= np.log(c**m)
    return float(value)


def multiscale_dispersion_entropy(
    y: np.ndarray,
    m: Optional[int] = 2,
    c: Optional[int] = 3,
    scale: Optional[int] = 3,
    tau: int = 1,
    normalize: bool = False,
) -> np.ndarray:
    """
    Costa coarse-graining followed by dispersion entropy at each scale.

    :param y: 1-D input series.
    :param m: Embedding dimension.
    :param c: Number of classes.
    :param scale: Highest scale factor (returns length ``scale``).
    :param tau: Embedding delay.
    :param normalize: Passed to ``dispersion_entropy``.
    :return: Dispersion entropy at scales ``1 .. scale``.
    """
    y = as_1d(y)
    if scale < 1:
        raise ValueError("scale must be an integer >= 1.")
    values = []
    for s in range(1, scale + 1):
        grain = coarse_grain(y, s)
        values.append(
            dispersion_entropy(grain, m=m, c=c, tau=tau, normalize=normalize)
        )
    return np.asarray(values, dtype=float)
