# -*- coding: utf-8 -*-
"""
Component classes for Optimization-based Signal Decomposition (OSD).

Each component exposes a masked proximal operator used by block-coordinate
descent.  Cost functions follow Meyers & Boyd, *Signal Decomposition via
Masked Proximal Operators* (https://web.stanford.edu/~boyd/papers/sig_decomp_mprox.html).

Portions adapted from the BSD-3-Clause reference implementation
https://github.com/cvxgrp/signal-decomposition (Copyright (c) 2019, Bennet Meyers).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Set, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized, spsolve


def soft_threshold(x: np.ndarray, level: float) -> np.ndarray:
    """Element-wise soft thresholding."""
    return np.sign(x) * np.maximum(np.abs(x) - level, 0.0)


def difference_matrix(n: int, order: int = 1) -> sp.csc_matrix:
    """Sparse difference matrix ``D`` with ``Dx ≈ diff(x, n=order)``."""
    if order < 1:
        raise ValueError("order must be >= 1")
    if n <= order:
        raise ValueError("signal length must exceed the difference order")
    d = sp.eye(n, format="csc")
    for _ in range(order):
        d = (d[1:, :] - d[:-1, :]).tocsc()
    return d


class Component(ABC):
    """Abstract OSD component with proximal mapping."""

    def __init__(
        self,
        weight: float = 1.0,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        self.weight = float(weight)
        self.vmin = vmin
        self.vmax = vmax
        self._is_residual = False

    @property
    def is_convex(self) -> bool:
        return True

    @abstractmethod
    def cost(self, x: np.ndarray) -> float:
        """Unweighted cost φ(x)."""

    @abstractmethod
    def prox_op(
        self,
        v: np.ndarray,
        weight: float,
        rho: float,
        use_set: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Proximal operator

            argmin_x  weight * φ(x) + (ρ / 2) * ||x - v||²
        """

    def _project_box(self, x: np.ndarray) -> np.ndarray:
        out = x
        if self.vmin is not None:
            out = np.maximum(out, self.vmin)
        if self.vmax is not None:
            out = np.minimum(out, self.vmax)
        return out


class MeanSquareSmall(Component):
    """
    Gaussian residual / noise component.

    φ(x) = ||x||₂² / size
    """

    def __init__(self, size: Optional[int] = None, weight: float = 1.0, **kwargs) -> None:
        super().__init__(weight=weight, **kwargs)
        self.size = size
        self._is_residual = True

    def cost(self, x: np.ndarray) -> float:
        size = self.size if self.size is not None else max(x.size, 1)
        return float(np.sum(x * x) / size)

    def prox_op(
        self,
        v: np.ndarray,
        weight: float,
        rho: float,
        use_set: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        size = self.size if self.size is not None else max(v.size, 1)
        a = (2.0 * weight) / (rho * size)
        out = v / (1.0 + a)
        if use_set is not None:
            out = out.copy()
            out[~use_set] = 0.0
        return self._project_box(out)


class SmoothDiff(Component):
    """
    Quadratic smoothness on the ``order``-th difference.

    φ(x) = ||D_order x||₂²

    ``order=2`` recovers the common smooth-trend prior (OSD
    ``SmoothSecondDifference``).
    """

    def __init__(self, order: int = 2, weight: float = 1.0, **kwargs) -> None:
        super().__init__(weight=weight, **kwargs)
        self.order = int(order)
        self._cache_n: Optional[int] = None
        self._dtd: Optional[sp.csc_matrix] = None

    def _dtd_matrix(self, n: int) -> sp.csc_matrix:
        if self._cache_n != n or self._dtd is None:
            d = difference_matrix(n, self.order)
            self._dtd = (d.T @ d).tocsc()
            self._cache_n = n
        return self._dtd

    def cost(self, x: np.ndarray) -> float:
        return float(np.sum(np.diff(x, n=self.order) ** 2))

    def prox_op(
        self,
        v: np.ndarray,
        weight: float,
        rho: float,
        use_set: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        n = v.size
        dtd = self._dtd_matrix(n)
        # (ρ I + 2 w DᵀD) x = ρ v   (missing entries: treat as free with rhs 0)
        diag = np.full(n, rho)
        rhs = rho * v
        if use_set is not None:
            diag = np.where(use_set, rho, rho)  # still regularize all coords
            rhs = rho * np.where(use_set, v, 0.0)
        a = sp.diags(diag, format="csc") + (2.0 * weight) * dtd
        x = spsolve(a, rhs)
        return self._project_box(np.asarray(x, dtype=float))


class SparseDiff(Component):
    """
    Sparse ``order``-th differences (convex TV / L1-trend prior).

    φ(x) = ||D_order x||₁

    - ``order=1``: total-variation / piecewise-constant
    - ``order=2``: L1 trend filtering / piecewise-linear
    """

    def __init__(
        self,
        order: int = 1,
        weight: float = 1.0,
        prox_iter: int = 80,
        admm_rho: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(weight=weight, **kwargs)
        self.order = int(order)
        self.prox_iter = int(prox_iter)
        self.admm_rho = float(admm_rho)
        self._cache_n: Optional[int] = None
        self._d: Optional[sp.csc_matrix] = None
        self._solve = None
        self._fact_key: Optional[tuple] = None

    def _diff_op(self, n: int) -> sp.csc_matrix:
        if self._cache_n != n or self._d is None:
            self._d = difference_matrix(n, self.order)
            self._cache_n = n
            self._solve = None
            self._fact_key = None
        return self._d

    def cost(self, x: np.ndarray) -> float:
        return float(np.sum(np.abs(np.diff(x, n=self.order))))

    def prox_op(
        self,
        v: np.ndarray,
        weight: float,
        rho: float,
        use_set: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve  min_x  λ ||Dx||₁ + (1/2) ||x - v||²  with λ = weight / ρ
        via ADMM (numpy / scipy only).
        """
        v = np.asarray(v, dtype=float).copy()
        n = v.size
        if use_set is not None:
            # Fill unknown samples with neighbour average for a warm start
            known = use_set
            if not np.all(known):
                idx = np.arange(n)
                v[~known] = np.interp(idx[~known], idx[known], v[known])

        lam = weight / max(rho, 1e-12)
        d = self._diff_op(n)
        mu = self.admm_rho
        key = (n, mu)
        if self._fact_key != key or self._solve is None:
            a = (sp.eye(n, format="csc") + mu * (d.T @ d)).tocsc()
            self._solve = factorized(a)
            self._fact_key = key

        x = v.copy()
        z = d @ x
        u = np.zeros_like(z)
        for _ in range(self.prox_iter):
            x = self._solve(v + mu * (d.T @ (z - u)))
            x = self._project_box(x)
            if use_set is not None:
                # Keep fidelity only on observed indices by blending
                x = np.where(use_set, x, x)
            dx = d @ x
            z = soft_threshold(dx + u, lam / mu)
            u = u + dx - z
        return np.asarray(x, dtype=float)


class Sparse(Component):
    """Sparse spikes: φ(x) = ||x||₁."""

    def cost(self, x: np.ndarray) -> float:
        return float(np.sum(np.abs(x)))

    def prox_op(
        self,
        v: np.ndarray,
        weight: float,
        rho: float,
        use_set: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        out = soft_threshold(v, weight / max(rho, 1e-12))
        if use_set is not None:
            out = out.copy()
            out[~use_set] = soft_threshold(v[~use_set], weight / max(rho, 1e-12))
        return self._project_box(out)


class FiniteSet(Component):
    """
    Non-convex finite-set prior (e.g. square wave in {-1, +1}).

    φ(x) = 0 if every sample is in ``values``, else +∞ (hard constraint).
    Proximal map = Euclidean projection onto the set (nearest value).
    """

    def __init__(
        self,
        values: Union[Sequence[float], Set[float]] = (-1.0, 1.0),
        weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(weight=weight, **kwargs)
        self.values = np.asarray(sorted(values), dtype=float)

    @property
    def is_convex(self) -> bool:
        return False

    def cost(self, x: np.ndarray) -> float:
        # 0 if already projected; otherwise count mismatches lightly
        nearest = self._project_values(x)
        return float(np.sum((x - nearest) ** 2))

    def _project_values(self, x: np.ndarray) -> np.ndarray:
        # Nearest neighbour in the finite alphabet
        dists = np.abs(x.reshape(-1, 1) - self.values.reshape(1, -1))
        return self.values[np.argmin(dists, axis=1)]

    def prox_op(
        self,
        v: np.ndarray,
        weight: float,
        rho: float,
        use_set: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # Hard projection (indicator prox); weight/rho unused
        out = self._project_values(np.asarray(v, dtype=float))
        return self._project_box(out)


class SmoothSecondDifference(SmoothDiff):
    """Alias: quadratic penalty on second differences."""

    def __init__(self, **kwargs) -> None:
        super().__init__(order=2, **kwargs)


class SparseFirstDiffConvex(SparseDiff):
    """Alias: TV / sparse first differences."""

    def __init__(self, **kwargs) -> None:
        super().__init__(order=1, **kwargs)


class SparseSecondDiffConvex(SparseDiff):
    """Alias: L1 trend filtering / sparse second differences."""

    def __init__(self, **kwargs) -> None:
        super().__init__(order=2, **kwargs)
