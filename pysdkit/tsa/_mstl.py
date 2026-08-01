# -*- coding: utf-8 -*-
"""
Created on 2025/02/12 00:17:59
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

from __future__ import annotations

import operator
import warnings
from typing import Optional, Sequence, Union, Dict, Any, Tuple

import numpy as np
from scipy.stats import boxcox

from ._stl import STL, STLResult


class MSTLResult(object):
    """Stores MSTL decomposition results (multiple seasonal components)."""

    def __init__(
        self,
        observed: np.ndarray,
        seasonal: np.ndarray,
        trend: np.ndarray,
        resid: np.ndarray,
        periods: Tuple[int, ...],
        weights: Optional[np.ndarray] = None,
        lmbda: Optional[float] = None,
    ) -> None:
        self.observed = np.asarray(observed)
        self.seasonal = np.asarray(seasonal)
        self.trend = np.asarray(trend)
        self.resid = np.asarray(resid)
        self.periods = tuple(periods)
        self.weights = (
            np.asarray(weights)
            if weights is not None
            else np.ones_like(self.observed, dtype=float)
        )
        self.lmbda = lmbda

    @property
    def seasonals(self) -> np.ndarray:
        """Alias for ``seasonal`` with shape ``(n_seasons, n_obs)``."""
        seas = np.asarray(self.seasonal)
        if seas.ndim == 1:
            return seas.reshape(1, -1)
        return seas


class MSTL(object):
    """
    Multiple Seasonal-Trend decomposition using LOESS (MSTL)

    Bandara, K., Hyndman, R. J., and Bergmeir, C. (2021).
    MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with
    Multiple Seasonal Patterns. arXiv:2107.13462.

    MSTL repeatedly applies :class:`STL` to extract several seasonal
    components (shortest period first), then returns the trend from the
    final STL fit and the residual ``deseasonalized - trend``.

    Notes
    -----
    This is *multiple-seasonality* STL (one univariate series, several
    periods), not a multivariate / multi-channel decomposer.  Missing values
    must be handled before calling :meth:`fit_transform`.  The implementation
    assumes at least one seasonal period.
    """

    def __init__(
        self,
        periods: Union[int, Sequence[int]],
        windows: Optional[Union[int, Sequence[int]]] = None,
        lmbda: Optional[Union[float, str]] = None,
        iterate: int = 2,
        stl_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param periods: Period of each seasonal component, e.g. ``(24, 168)``
            for hourly data with daily and weekly seasonality.
        :param windows: Odd seasonal smoother lengths for each period.  If
            ``None``, uses the paper default ``7 + 4 * i`` for
            ``i = 1 .. n_seasons`` (e.g. ``(11, 15)`` for two seasons).
        :param lmbda: Box-Cox transform applied before decomposition.
            ``None`` means no transform; ``"auto"`` estimates λ by MLE;
            a float applies that fixed λ (requires strictly positive data).
        :param iterate: Number of outer refinement iterations when there are
            two or more seasonal periods (forced to 1 for a single period).
        :param stl_kwargs: Extra keyword arguments forwarded to :class:`STL`
            (e.g. ``robust=True``, ``trend_deg=0``).  Keys ``period`` and
            ``seasonal`` are reserved and ignored if present.
        """
        self.periods = self._as_period_tuple(periods)
        if len(self.periods) == 0:
            raise ValueError("periods must contain at least one seasonal period")

        self.windows = (
            None if windows is None else self._as_window_tuple(windows, len(self.periods))
        )
        if self.windows is not None and len(self.windows) != len(self.periods):
            raise ValueError("periods and windows must have the same length")

        if iterate < 1:
            raise ValueError("iterate must be a positive integer")
        self.iterate = int(iterate)

        if lmbda is not None and lmbda != "auto":
            try:
                lmbda = float(lmbda)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "lmbda must be None, 'auto', or a floating-point value"
                ) from exc
        self.lmbda = lmbda

        stl_kwargs = dict(stl_kwargs or {})
        for key in ("period", "seasonal", "endog"):
            stl_kwargs.pop(key, None)
        self.stl_kwargs = stl_kwargs

        self.est_lmbda: Optional[float] = None

    def __str__(self) -> str:
        return (
            "Multiple Seasonal-Trend decomposition using LOESS (MSTL)"
            f"(periods={self.periods}, windows={self.windows}, "
            f"lmbda={self.lmbda}, iterate={self.iterate})"
        )

    def __call__(self, endog: np.ndarray) -> MSTLResult:
        """Allow instances to be called like functions."""
        return self.fit_transform(endog)

    def fit_transform(self, endog: np.ndarray) -> MSTLResult:
        """
        Decompose a univariate series into trend, multiple seasonals, and residual.

        :param endog: 1-D input series
        :return: :class:`MSTLResult` with ``seasonal`` shaped
            ``(n_seasons, n_obs)`` (or ``(n_obs,)`` when ``n_seasons == 1``)
        """
        y_raw = np.asarray(endog, dtype=float).ravel()
        if y_raw.ndim != 1 or y_raw.size < 2:
            raise ValueError("endog must be a 1-D array with at least 2 samples")
        if not np.all(np.isfinite(y_raw)):
            raise ValueError(
                "endog contains non-finite values; impute missing data before MSTL"
            )

        nobs = y_raw.size
        periods, windows = self._prepare_periods_windows(nobs)
        if len(periods) == 0:
            raise ValueError(
                "No valid seasonal periods remain after removing those >= n_obs / 2"
            )

        # Box-Cox optional preprocessing (same convention as the reference paper)
        y, used_lmbda = self._maybe_boxcox(y_raw)
        self.est_lmbda = used_lmbda

        n_seasons = len(periods)
        iterate = 1 if n_seasons == 1 else self.iterate

        seasonal = np.zeros((n_seasons, nobs), dtype=float)
        deseas = y.copy()
        last_fit: Optional[STLResult] = None

        # Extract / refine each seasonal component (shortest period first)
        for _ in range(iterate):
            for i, (period, window) in enumerate(zip(periods, windows)):
                deseas = deseas + seasonal[i]
                stl = STL(period=period, seasonal=window, **self.stl_kwargs)
                last_fit = stl.fit_transform(deseas)
                seasonal[i] = last_fit.seasonal
                deseas = deseas - seasonal[i]

        assert last_fit is not None
        trend = np.asarray(last_fit.trend, dtype=float)
        resid = deseas - trend
        weights = np.asarray(last_fit.weights, dtype=float)

        seasonal_out: np.ndarray
        if n_seasons == 1:
            seasonal_out = seasonal[0]
        else:
            seasonal_out = seasonal

        return MSTLResult(
            observed=y,
            seasonal=seasonal_out,
            trend=trend,
            resid=resid,
            periods=periods,
            weights=weights,
            lmbda=used_lmbda,
        )

    def _prepare_periods_windows(
        self, nobs: int
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        periods = list(self.periods)
        if self.windows is None:
            windows = list(self._default_seasonal_windows(len(periods)))
            # Sort periods ascending; keep default windows aligned to sorted order
            periods = sorted(periods)
        else:
            periods, windows = map(
                list, self._sort_periods_and_windows(periods, list(self.windows))
            )

        # Drop periods that are too long for the sample (paper / statsmodels)
        keep = [p < nobs / 2 for p in periods]
        if not all(keep):
            warnings.warn(
                "A period(s) is larger than half the length of the time series. "
                "Removing these period(s).",
                UserWarning,
                stacklevel=2,
            )
            periods = [p for p, k in zip(periods, keep) if k]
            windows = [w for w, k in zip(windows, keep) if k]

        return tuple(periods), tuple(windows)

    def _maybe_boxcox(self, y: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        if self.lmbda is None:
            return y.copy(), None

        if np.any(y <= 0):
            raise ValueError("Box-Cox transform requires strictly positive data")

        if self.lmbda == "auto":
            transformed, est = boxcox(y, lmbda=None)
            return np.asarray(transformed, dtype=float), float(est)

        transformed = boxcox(y, lmbda=float(self.lmbda))
        return np.asarray(transformed, dtype=float), float(self.lmbda)

    @staticmethod
    def _default_seasonal_windows(n: int) -> Tuple[int, ...]:
        # Bandara et al. Appendix A: s_window_i = 7 + 4 * i, i = 1..n
        return tuple(7 + 4 * i for i in range(1, n + 1))

    @staticmethod
    def _as_period_tuple(periods: Union[int, Sequence[int]]) -> Tuple[int, ...]:
        if isinstance(periods, (str, bytes)):
            raise TypeError("periods must be an int or a sequence of ints")
        if isinstance(periods, (int, np.integer)):
            values = [operator.index(periods)]
        else:
            values = [operator.index(p) for p in periods]
        if any(p < 2 for p in values):
            raise ValueError("each period must be an integer >= 2")
        return tuple(values)

    @staticmethod
    def _as_window_tuple(
        windows: Union[int, Sequence[int]], n_seasons: int
    ) -> Tuple[int, ...]:
        if isinstance(windows, (str, bytes)):
            raise TypeError("windows must be an int or a sequence of ints")
        if isinstance(windows, (int, np.integer)):
            values = [operator.index(windows)]
        else:
            values = [operator.index(w) for w in windows]

        if len(values) == 1 and n_seasons > 1:
            raise ValueError("windows must have the same length as periods")
        for w in values:
            if w < 3 or w % 2 == 0:
                raise ValueError("each window must be an odd integer >= 3")
        return tuple(values)

    @staticmethod
    def _sort_periods_and_windows(
        periods: Sequence[int], windows: Sequence[int]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        if len(periods) != len(windows):
            raise ValueError("periods and windows must have the same length")
        paired = sorted(zip(periods, windows), key=lambda pw: pw[0])
        periods_s, windows_s = zip(*paired)
        return tuple(periods_s), tuple(windows_s)
