# -*- coding: utf-8 -*-
"""
Optimization-based Signal Decomposition (OSD).

Decompose a (possibly incomplete) 1-D signal into a sum of structured
components by solving

    minimize   Σ_k  w_k φ_k(x^k)
    subject to Σ_k x^k  =  y   on observed samples.

The default solver is block-coordinate descent (BCD) with masked proximal
operators — the algorithm analysed by Meyers & Boyd.  Only NumPy and SciPy
are required (no CVXPY / QSS).

Reference
---------
Bennet Meyers, Stephen Boyd.
"Signal Decomposition via Masked Proximal Operators."
https://web.stanford.edu/~boyd/papers/sig_decomp_mprox.html

Reference software (BSD-3-Clause): https://github.com/cvxgrp/signal-decomposition
Copyright (c) 2019, Bennet Meyers.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from pysdkit._osd.components import (
    Component,
    FiniteSet,
    MeanSquareSmall,
    SmoothDiff,
    SmoothSecondDifference,
    Sparse,
    SparseDiff,
    SparseFirstDiffConvex,
    SparseSecondDiffConvex,
)


def _make_estimate(
    y: np.ndarray,
    X: np.ndarray,
    use_ix: np.ndarray,
    residual_term: int = 0,
) -> np.ndarray:
    """Force feasibility: residual component absorbs y − Σ_{k≠r} x^k."""
    X_tilde = np.array(X, copy=True, dtype=float)
    others = [i for i in range(X.shape[0]) if i != residual_term]
    X_tilde[residual_term, use_ix] = y[use_ix] - np.sum(X[others][:, use_ix], axis=0)
    X_tilde[residual_term, ~use_ix] = 0.0
    return X_tilde


def _objective(
    y: np.ndarray,
    X: np.ndarray,
    components: Sequence[Component],
    use_ix: np.ndarray,
    residual_term: int = 0,
) -> float:
    X_tilde = _make_estimate(y, X, use_ix, residual_term)
    total = 0.0
    for k, comp in enumerate(components):
        total += comp.weight * float(comp.cost(X_tilde[k]))
    return total


class OSD(object):
    """
    Optimization-based Signal Decomposition.

    Parameters
    ----------
    components :
        List of :class:`Component` instances.  By convention index ``0`` is the
        residual / noise term (``MeanSquareSmall``).
    residual_term :
        Index of the residual component used to enforce Σ x^k = y.
    max_iter :
        Maximum BCD iterations.
    rho :
        Augmented-Lagrangian / prox step size.  ``None`` → ``2 / N``.
    abs_tol, rel_tol :
        Stopping tolerances on the dual residual.
    """

    def __init__(
        self,
        components: Optional[Sequence[Component]] = None,
        residual_term: int = 0,
        max_iter: int = 200,
        rho: Optional[float] = None,
        abs_tol: float = 1e-5,
        rel_tol: float = 1e-5,
        verbose: bool = False,
    ) -> None:
        self.components: List[Component] = (
            list(components) if components is not None else []
        )
        self.residual_term = int(residual_term)
        self.max_iter = int(max_iter)
        self.rho = rho
        self.abs_tol = float(abs_tol)
        self.rel_tol = float(rel_tol)
        self.verbose = bool(verbose)

        self.imfs: Optional[np.ndarray] = None
        self.objective_value: Optional[float] = None
        self.history: Optional[dict] = None

    def __call__(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        return self.fit_transform(signal, **kwargs)

    def __str__(self) -> str:
        return "Optimization-based Signal Decomposition (OSD)"

    @staticmethod
    def preset(
        name: str,
        length: int,
        **overrides,
    ) -> "OSD":
        """
        Build a ready-made component list for common demos.

        Parameters
        ----------
        name :
            ``"convex_demo"`` — residual + smooth trend + TV jumps
            (sine + square-wave notebook).
            ``"l1_trend"`` — residual + L1 trend filtering.
            ``"nonconvex_square"`` — residual + smooth + finite-set square wave.
        length :
            Signal length ``T`` (used to scale default weights).
        """
        t = max(int(length), 1)
        name = name.lower().strip()
        if name in ("convex_demo", "convex", "sine_square"):
            comps = [
                MeanSquareSmall(size=t),
                SmoothSecondDifference(weight=overrides.get("smooth_weight", 1e3 / t)),
                SparseFirstDiffConvex(
                    weight=overrides.get("tv_weight", 2.0 / t),
                    vmin=overrides.get("vmin", -1.0),
                    vmax=overrides.get("vmax", 1.0),
                ),
            ]
        elif name in ("l1_trend", "l1tf", "trend"):
            comps = [
                MeanSquareSmall(size=t, weight=overrides.get("residual_weight", 1.0 / t)),
                SparseSecondDiffConvex(weight=overrides.get("trend_weight", 1.0)),
            ]
        elif name in ("nonconvex_square", "nonconvex", "finite_set"):
            comps = [
                MeanSquareSmall(size=t, weight=overrides.get("residual_weight", 1.0 / t)),
                SmoothSecondDifference(weight=overrides.get("smooth_weight", 1.0)),
                FiniteSet(
                    values=overrides.get("values", (-1.0, 1.0)),
                    weight=overrides.get("set_weight", 1.0),
                ),
            ]
        else:
            raise ValueError(
                "Unknown preset {!r}. Choose from "
                "'convex_demo', 'l1_trend', 'nonconvex_square'.".format(name)
            )
        max_iter = overrides.get("max_iter", 200)
        return OSD(components=comps, max_iter=max_iter, verbose=overrides.get("verbose", False))

    def fit_transform(
        self,
        signal: np.ndarray,
        components: Optional[Sequence[Component]] = None,
        use_set: Optional[np.ndarray] = None,
        max_iter: Optional[int] = None,
        rho: Optional[float] = None,
        X_init: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Decompose ``signal`` into additive components.

        :param signal: 1-D array (``NaN`` marks missing samples)
        :param components: optional override of constructor components
        :param use_set: boolean mask of observed samples (default: non-NaN)
        :param max_iter: optional override of iteration budget
        :param rho: optional override of prox step size
        :param X_init: optional warm start of shape ``(K, N)``
        :return: array of shape ``(K, N)`` with ``sum(axis=0) ≈ signal``
                 on observed indices
        """
        y = np.asarray(signal, dtype=float).ravel()
        if y.ndim != 1:
            raise ValueError("OSD currently supports 1-D signals only")
        n = y.size
        if n < 3:
            raise ValueError("signal length must be >= 3")

        comps = list(components) if components is not None else list(self.components)
        if not comps:
            raise ValueError(
                "No components provided. Pass components=... or use OSD.preset(...)."
            )
        # Ensure residual size default
        for c in comps:
            if isinstance(c, MeanSquareSmall) and c.size is None:
                c.size = n

        known = ~np.isnan(y) if use_set is None else np.asarray(use_set, dtype=bool)
        if known.shape != y.shape:
            raise ValueError("use_set must match signal shape")
        known = np.logical_and(known, ~np.isnan(y))
        if not np.any(known):
            raise ValueError("signal has no observed samples")

        k = len(comps)
        max_iter = self.max_iter if max_iter is None else int(max_iter)
        rho_val = self.rho if rho is None else rho
        if rho_val is None:
            rho_val = 2.0 / float(n)

        if X_init is None:
            X = np.zeros((k, n), dtype=float)
            X[self.residual_term, known] = y[known]
        else:
            X = np.array(X_init, dtype=float, copy=True)
            if X.shape != (k, n):
                raise ValueError("X_init must have shape (K, N)")

        obj_hist: List[float] = []
        res_hist: List[float] = []
        indices = np.arange(k)

        for it in range(max_iter):
            for j in range(k):
                if j == self.residual_term:
                    continue
                others = (indices != self.residual_term) & (indices != j)
                rhs = np.sum(X[others], axis=0)
                vin = np.zeros(n, dtype=float)
                vin[known] = y[known] - rhs[known]
                X[j] = comps[j].prox_op(vin, comps[j].weight, rho_val, use_set=known)

            X = _make_estimate(y, X, known, residual_term=self.residual_term)

            # Dual residual diagnostic (same spirit as reference BCD)
            grads = np.zeros_like(X)
            for j in range(k):
                if j == self.residual_term:
                    continue
                others = (indices != self.residual_term) & (indices != j)
                rhs = np.sum(X[others], axis=0)
                vin = np.zeros(n)
                vin[known] = y[known] - rhs[known]
                grads[j, known] = rho_val * (vin[known] - X[j, known])
            # residual component gradient ~ 2 x0 / N for MeanSquareSmall
            size0 = n
            if isinstance(comps[self.residual_term], MeanSquareSmall):
                size0 = comps[self.residual_term].size or n
            grads[self.residual_term] = X[self.residual_term] * (
                2.0 * comps[self.residual_term].weight / size0
            )

            active = indices != self.residual_term
            if np.any(active):
                r = float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (grads[active] - grads[self.residual_term]) ** 2,
                                axis=1,
                            )
                        )
                    )
                )
            else:
                r = 0.0

            obj = _objective(y, X, comps, known, self.residual_term)
            obj_hist.append(obj)
            res_hist.append(r)
            stop_tol = self.abs_tol + self.rel_tol * (
                np.linalg.norm(grads[self.residual_term]) + 1e-12
            )
            if self.verbose and (it % 20 == 0 or it == max_iter - 1):
                print(
                    "OSD BCD it={:4d}  obj={:.4e}  r={:.3e}  tol={:.3e}".format(
                        it, obj, r, stop_tol
                    )
                )
            if r <= stop_tol and it >= 1:
                break

        self.imfs = X
        self.objective_value = obj_hist[-1] if obj_hist else None
        self.history = {"obj_vals": obj_hist, "residual": res_hist}
        return X

    def reconstruct(self, imfs: Optional[np.ndarray] = None) -> np.ndarray:
        """Sum of components (exact on observed samples after ``fit_transform``)."""
        X = self.imfs if imfs is None else imfs
        if X is None:
            raise ValueError("No decomposition available; run fit_transform first.")
        return np.sum(X, axis=0)


__all__ = [
    "OSD",
    "Component",
    "MeanSquareSmall",
    "SmoothDiff",
    "SmoothSecondDifference",
    "SparseDiff",
    "SparseFirstDiffConvex",
    "SparseSecondDiffConvex",
    "Sparse",
    "FiniteSet",
]
