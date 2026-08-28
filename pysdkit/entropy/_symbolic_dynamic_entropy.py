# -*- coding: utf-8 -*-
"""
Symbolic dynamic entropy of a 1-D series (Li et al. 2017).
"""

import numpy as np
from typing import Optional

from pysdkit.entropy._coarse_grain import as_1d, embed, shannon


def symbolic_dynamic_entropy(
    y: np.ndarray,
    m: Optional[int] = 2,
    c: Optional[int] = 3,
    tau: int = 1,
    normalize: bool = True,
) -> float:
    """
    Symbolic dynamic entropy (SyDyEn) of a 1-D series.

    The series is quantized into ``c`` **equal-width** bins, delay-embedded
    to words of length ``m``, and the entropy of the next symbol given the
    current word is returned.  That Markov entropy is the usual rotating-
    machinery feature (Li et al., MSSP 2017).

    Li, Y., Yang, Y., Li, G., Xu, M., & Huang, W. (2017). A fault diagnosis
    scheme for planetary gearboxes using modified multi-scale symbolic
    dynamic entropy and mRMR feature selection. Mechanical Systems and
    Signal Processing, 91, 295-312.

    :param y: 1-D input series.
    :param m: Word length.
    :param c: Number of equal-width symbols, integer > 1.
    :param tau: Time delay between word coordinates.
    :param normalize: If True, divide by ``log(c)``.
    :return: Conditional entropy of the next symbol given the current word.
    """
    y = as_1d(y)
    n = len(y)
    if c < 2:
        raise ValueError("c must be an integer > 1.")
    if m < 1 or m >= n:
        raise ValueError("Embedding dimension must satisfy 1 <= m < N.")
    if tau < 1:
        raise ValueError("tau must be a positive integer.")

    span = np.ptp(y)
    if span == 0.0:
        symbols = np.zeros(n, dtype=int)
    else:
        edges = np.linspace(y.min(), y.max(), c + 1)
        symbols = np.digitize(y, edges[1:-1], right=False)
        symbols = np.clip(symbols, 0, c - 1)

    n_words = n - m * tau
    if n_words < 1:
        raise ValueError("Not enough samples for a next-symbol after each word.")
    words = embed(symbols.astype(float), m, tau)[:n_words].astype(int)
    nxt = symbols[m * tau : m * tau + n_words]

    packed = np.concatenate([words, nxt[:, None]], axis=1)
    unique_ext, counts_ext = np.unique(packed, axis=0, return_counts=True)
    unique_w, inverse = np.unique(words, axis=0, return_inverse=True)
    p_w = np.bincount(inverse).astype(float) / n_words

    cond = 0.0
    for i, word in enumerate(unique_w):
        mask = np.all(unique_ext[:, :m] == word, axis=1)
        p_next = counts_ext[mask].astype(float)
        p_next /= p_next.sum()
        cond += p_w[i] * shannon(p_next)

    if normalize:
        cond /= np.log(c)
    return float(cond)
