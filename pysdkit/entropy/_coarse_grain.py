# -*- coding: utf-8 -*-
"""
Costa coarse-graining for multiscale entropy.
"""

import numpy as np


def as_1d(y: np.ndarray) -> np.ndarray:
    """Flatten ``y`` to a 1-D float array."""
    return np.asarray(y, dtype=float).ravel()


def shannon(prob: np.ndarray) -> float:
    """Natural-log Shannon entropy of a probability vector (zeros ignored)."""
    p = np.asarray(prob, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def embed(y: np.ndarray, m: int, tau: int = 1) -> np.ndarray:
    """Delay-embed ``y`` into shape ``(n_vectors, m)``."""
    y = as_1d(y)
    n_vec = len(y) - (m - 1) * tau
    if n_vec < 1:
        raise ValueError(
            "Not enough samples for embedding dimension m=%d and delay tau=%d."
            % (m, tau)
        )
    return np.column_stack([y[i * tau : i * tau + n_vec] for i in range(m)])


def coarse_grain(x: np.ndarray, scale: int, offset: int = 0) -> np.ndarray:
    """
    Non-overlapping mean blocks (Costa coarse-graining).

    :param x: 1-D series.
    :param scale: Block length (scale factor), integer >= 1.
    :param offset: Start index; used for composite / phase-shifted grains.
    :return: Coarse-grained series of length ``(len(x) - offset) // scale``.
    """
    x = as_1d(x)
    if scale < 1:
        raise ValueError("scale must be an integer >= 1.")
    if offset < 0:
        raise ValueError("offset must be >= 0.")
    x = x[offset:]
    n_blocks = len(x) // scale
    if n_blocks < 1:
        return np.empty(0, dtype=float)
    return x[: n_blocks * scale].reshape(n_blocks, scale).mean(axis=1)
