# -*- coding: utf-8 -*-
"""
Created on 2025/02/12 00:14:21
@author: Ruizhe Wang
@email: changewam@stu.xidian.edu.cn
"""

import operator
from typing import Optional, Literal

import numpy as np
from scipy.linalg import lstsq


class STLResult(object):
    """Stores STL decomposition results"""

    def __init__(
        self,
        observed: np.ndarray,
        seasonal: np.ndarray,
        trend: np.ndarray,
        resid: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> None:
        self.observed = observed
        self.seasonal = seasonal
        self.trend = trend
        self.resid = resid
        self.weights = weights if weights is not None else np.ones_like(observed)


class STL(object):
    """
    Seasonal-Trend decomposition using LOESS (STL)

    R. B. Cleveland, W. S. Cleveland, J. E. McRae, and I. Terpenning.
    STL: A Seasonal-Trend Decomposition Procedure Based on Loess.
    Journal of Official Statistics, 6:3-73, 1990.

    STL uses LOESS (locally estimated scatterplot smoothing) to extract
    smooth estimates of the seasonal, trend and remainder components.
    The key inputs into STL are:
    (1) seasonal - The length of the seasonal smoother. Must be odd (>= 3; >= 7 recommended).
    (2) trend - The length of the trend smoother, usually around 150% of season. Must be odd and >= period.
    (3) low_pass - The length of the low-pass estimation window, usually the smallest odd number >= period.
    """

    def __init__(
        self,
        period: int,
        seasonal: int = 7,
        trend: Optional[int] = None,
        low_pass: Optional[int] = None,
        seasonal_deg: Literal[0, 1] = 1,
        trend_deg: Literal[0, 1] = 1,
        low_pass_deg: Literal[0, 1] = 1,
        robust: bool = False,
        seasonal_jump: int = 1,
        trend_jump: int = 1,
        low_pass_jump: int = 1,
    ) -> None:
        """
        Initialize STL decomposer configuration

        :param period: Seasonal period length (must be >= 2)
        :param seasonal: Seasonal smoothing window size (odd, >= 3; >= 7 recommended)
        :param trend: Trend smoothing window size (odd, >= period)
        :param low_pass: Low-pass filter window size (odd, >= period)
        :param seasonal_deg: Seasonal LOESS degree (0 or 1)
        :param trend_deg: Trend LOESS degree (0 or 1)
        :param low_pass_deg: Low-pass filter LOESS degree (0 or 1)
        :param robust: Whether to use robust mode for outlier handling
        :param seasonal_jump: Seasonal component calculation jump step (optimization)
        :param trend_jump: Trend component calculation jump step (optimization)
        :param low_pass_jump: Low-pass filter calculation jump step (optimization)
        """

        # Validate and set period
        try:
            period = operator.index(period)
        except TypeError as exc:
            raise ValueError("period must be an integer >= 2") from exc
        if period < 2:
            raise ValueError("period must be an integer >= 2")
        self.period = period

        # Validate and set seasonal parameters
        if not self._is_odd_int(seasonal) or int(seasonal) < 3:
            raise ValueError("seasonal must be an odd integer >= 3")
        self.seasonal = int(seasonal)

        # Calculate trend window size (Cleveland et al. default)
        if trend is None:
            # Ensure denominator is positive
            denom = max(1 - 1.5 / self.seasonal, 0.01)
            trend = int(np.ceil(1.5 * period / denom))
            # Ensure it's odd
            trend = trend + 1 if trend % 2 == 0 else trend
        if not self._is_odd_int(trend) or int(trend) <= period:
            # NETLIB / statsmodels: trend > period (strict), odd
            raise ValueError("trend must be an odd integer > period length")
        self.trend = int(trend)

        # Calculate low-pass filter window size
        if low_pass is None:
            low_pass = period + 1 if period % 2 == 0 else period
            if low_pass % 2 == 0:
                low_pass += 1
        if not self._is_odd_int(low_pass) or int(low_pass) < period:
            raise ValueError("low_pass must be an odd integer >= period length")
        self.low_pass = int(low_pass)

        # Set other parameters
        if seasonal_deg not in (0, 1) or trend_deg not in (0, 1) or low_pass_deg not in (
            0,
            1,
        ):
            raise ValueError("LOESS degrees must be 0 or 1")
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = bool(robust)
        self.seasonal_jump = max(1, int(seasonal_jump))
        self.trend_jump = max(1, int(trend_jump))
        self.low_pass_jump = max(1, int(low_pass_jump))

    @staticmethod
    def _is_odd_int(x: int) -> bool:
        """Check if value is an odd positive integer (accepts numpy integers)."""
        try:
            x = operator.index(x)
        except TypeError:
            return False
        return x > 0 and x % 2 == 1

    def __str__(self) -> str:
        return "Seasonal-Trend decomposition using LOESS (STL)"

    def __call__(
        self,
        endog: np.ndarray,
        inner_iter: Optional[int] = None,
        outer_iter: Optional[int] = None,
    ) -> STLResult:
        """
        Make class callable like a function
        Equivalent to calling fit_transform method
        """
        return self.fit_transform(endog, inner_iter, outer_iter)

    def _inner_loop(self):
        """STL inner loop algorithm"""
        # Step 1: Detrend
        detrended = self.y - self.trend_arr

        # Step 2: Seasonal subseries smoothing
        seasonal_temp = self._seasonal_smoothing(detrended)

        # Step 3: Low-pass filtering (triple moving average + LOESS)
        low_pass = self._low_pass_filter(seasonal_temp)

        # Step 4: Remove low-frequency part from seasonal component
        self.seasonal_arr = (
            seasonal_temp[self.period : self.period + self.nobs] - low_pass
        )

        # Step 5: Deseasonalize
        deseasonalized = self.y - self.seasonal_arr

        # Step 6: Trend smoothing
        self.trend_arr = self._trend_smoothing(deseasonalized)

    def _seasonal_smoothing(self, detrended: np.ndarray) -> np.ndarray:
        """
        Seasonal subseries smoothing

        :param detrended: Detrended series
        :return: Temporary seasonal component (length = nobs + 2*period)
        """
        # Initialize output array (with boundary extension)
        seasonal_temp = np.zeros(self.nobs + 2 * self.period)

        # Process each seasonal position (cycle-subseries)
        for j in range(self.period):
            # Get subseries for current seasonal position
            subseries = detrended[j :: self.period]
            n_sub = len(subseries)
            x = np.arange(n_sub, dtype=float)

            # Get corresponding robust weights
            weights = self.rw[j :: self.period] if self.robust else np.ones(n_sub)

            # Apply LOESS smoothing on the observed cycle indices
            smoothed = self._loess(
                x=x,
                y=subseries,
                weights=weights,
                window_size=self.seasonal,
                degree=self.seasonal_deg,
                jump=self.seasonal_jump,
            )

            # LOESS extrapolation at the cycle boundaries (Cleveland et al.)
            left = self._loess_at(
                x,
                subseries,
                x0=-1.0,
                window_size=self.seasonal,
                degree=self.seasonal_deg,
                weights=weights,
            )
            right = self._loess_at(
                x,
                subseries,
                x0=float(n_sub),
                window_size=self.seasonal,
                degree=self.seasonal_deg,
                weights=weights,
            )

            # Store into the extended workspace.  Time t maps to index
            # ``period + t``, so the central slice
            # ``seasonal_temp[period:period+nobs]`` aligns with the series.
            for i in range(-1, n_sub + 1):
                idx = self.period + j + i * self.period
                if 0 <= idx < len(seasonal_temp):
                    if i == -1:
                        seasonal_temp[idx] = left
                    elif i == n_sub:
                        seasonal_temp[idx] = right
                    else:
                        seasonal_temp[idx] = smoothed[i]

        return seasonal_temp

    def _low_pass_filter(self, seasonal_temp: np.ndarray) -> np.ndarray:
        """
        Low-pass filtering of the extended seasonal series.

        Classic STL applies three moving averages of lengths
        (period, period, 3), which shortens the series from
        ``nobs + 2*period`` down to ``nobs``, then LOESS-smooths the result.

        :param seasonal_temp: Temporary seasonal component
        :return: Low-frequency component (length = nobs)
        """
        # First moving average (length=period)
        ma1 = self._moving_average_reduce(seasonal_temp, self.period)

        # Second moving average (length=period)
        ma2 = self._moving_average_reduce(ma1, self.period)

        # Third moving average (length=3) -> length == nobs
        ma3 = self._moving_average_reduce(ma2, 3)

        # Apply LOESS smoothing
        low_pass = self._loess(
            x=np.arange(len(ma3), dtype=float),
            y=ma3,
            window_size=self.low_pass,
            degree=self.low_pass_deg,
            jump=self.low_pass_jump,
        )

        if len(low_pass) != self.nobs:
            # Defensive trim / pad (should not trigger with classic MA lengths)
            out = np.zeros(self.nobs, dtype=float)
            n = min(self.nobs, len(low_pass))
            out[:n] = low_pass[:n]
            return out
        return low_pass

    def _trend_smoothing(self, deseasonalized: np.ndarray) -> np.ndarray:
        """
        Trend component smoothing

        :param deseasonalized: Deseasonalized series
        :return: Trend component
        """
        weights = self.rw if self.robust else None
        return self._loess(
            x=np.arange(self.nobs),
            y=deseasonalized,
            weights=weights,
            window_size=self.trend,
            degree=self.trend_deg,
            jump=self.trend_jump,
        )

    def _update_robust_weights(self):
        """Calculate robust weights"""
        # Calculate absolute residuals
        resid = self.y - self.trend_arr - self.seasonal_arr
        abs_resid = np.abs(resid)

        # Calculate median absolute deviation (MAD)
        median_abs = np.median(abs_resid)
        if median_abs < 1e-12:
            self.rw.fill(1.0)
            return

        # Calculate h and weights (bisquare function)
        h = 6 * median_abs
        c1 = 0.001 * h
        c9 = 0.999 * h

        for i in range(self.nobs):
            r = abs_resid[i]
            if r <= c1:
                self.rw[i] = 1.0
            elif r <= c9:
                t = r / h
                self.rw[i] = (1.0 - t**2) ** 2
            else:
                self.rw[i] = 0.0

    def _loess(
        self,
        x: np.ndarray,
        y: np.ndarray,
        window_size: int,
        degree: int,
        jump: int = 1,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        LOESS locally weighted regression

        :param x: Independent variable (1D array)
        :param y: Dependent variable (1D array)
        :param window_size: Smoothing window size
        :param degree: Polynomial degree (0 or 1)
        :param jump: Calculation jump step (optimization)
        :param weights: Observation weights (for robust estimation)
        :return: Smoothed series
        """
        n = len(x)
        result = np.zeros(n)

        # Handle small sample case
        if n < 2:
            return y.copy()

        # Expand weights array
        if weights is None:
            weights = np.ones(n)

        # Determine actual calculation points
        if jump > 1:
            indices = np.arange(0, n, jump)
            if indices[-1] != n - 1:
                indices = np.append(indices, n - 1)
        else:
            indices = np.arange(n)

        # Perform LOESS smoothing for each calculation point
        for i in indices:
            result[i] = self._loess_at(
                x, y, float(x[i]), window_size, degree, weights, fallback=float(y[i])
            )

        # Linear interpolation for jump points
        if jump > 1 and len(indices) > 1:
            for j in range(len(indices) - 1):
                start = indices[j]
                end = indices[j + 1]
                num_points = end - start
                if num_points > 1:
                    # Linear interpolation
                    interp = np.linspace(result[start], result[end], num_points + 1)
                    result[start : end + 1] = interp

        return result

    @staticmethod
    def _loess_at(
        x: np.ndarray,
        y: np.ndarray,
        x0: float,
        window_size: int,
        degree: int,
        weights: np.ndarray,
        fallback: Optional[float] = None,
    ) -> float:
        """
        Evaluate LOESS at an arbitrary abscissa ``x0``.

        :param x: Independent variable
        :param y: Dependent variable
        :param x0: Target abscissa (may lie outside ``x`` for extrapolation)
        :param window_size: Neighborhood size
        :param degree: Polynomial degree (0 or 1)
        :param weights: Observation weights
        :param fallback: Value used if the local fit is singular
        :return: Fitted value at ``x0``
        """
        if fallback is None:
            # Nearest observation as a safe fallback
            fallback = float(y[int(np.argmin(np.abs(x - x0)))])

        n = len(x)
        distances = np.abs(x - x0)

        if window_size >= n:
            max_distance = float(np.max(distances)) * window_size / max(n, 1)
        else:
            max_distance = float(np.partition(distances, window_size - 1)[window_size - 1])

        if max_distance <= 0:
            return fallback

        # Tricube neighborhood weights
        d_scaled = distances / max_distance
        weights_tricube = np.where(d_scaled < 1.0, (1.0 - d_scaled**3) ** 3, 0.0)
        weights_total = weights * weights_tricube

        if degree == 0:  # Constant fit
            weight_sum = np.sum(weights_total)
            if weight_sum < 1e-12:
                return fallback
            return float(np.sum(weights_total * y) / weight_sum)

        if degree == 1:  # Linear fit centered at x0
            X = np.column_stack((np.ones(n, dtype=float), x - x0))
            sqrt_w = np.sqrt(np.maximum(weights_total, 0.0))
            X_weighted = X * sqrt_w[:, np.newaxis]
            y_weighted = y * sqrt_w
            try:
                beta, _, _, _ = lstsq(X_weighted, y_weighted, lapack_driver="gelsy")
                return float(beta[0])
            except np.linalg.LinAlgError:
                return fallback

        raise ValueError("Only degrees 0 or 1 are supported")

    @staticmethod
    def _moving_average_reduce(data: np.ndarray, window: int) -> np.ndarray:
        """
        Non-centered moving average that shortens the series by ``window - 1``.

        This matches the NETLIB / Cleveland STL low-pass filter stage.

        :param data: Input data
        :param window: Window size
        :return: Moving average of length ``len(data) - window + 1``
        """
        data = np.asarray(data, dtype=float)
        n = len(data)
        if window <= 1:
            return data.copy()
        if window > n:
            return np.array([float(np.mean(data))])

        cumsum = np.cumsum(np.insert(data, 0, 0.0))
        return (cumsum[window:] - cumsum[:-window]) / window

    def fit_transform(
        self,
        endog: np.ndarray,
        inner_iter: Optional[int] = None,
        outer_iter: Optional[int] = None,
    ) -> STLResult:
        """
        Perform STL decomposition and return results

        :param endog: Time series to decompose (1D numpy array)
        :param inner_iter: Number of inner loop iterations (None for default)
        :param outer_iter: Number of outer loop iterations (None for default)
        :return: STLResult object
        """
        # Validate input data
        self.y = np.asarray(endog, dtype=float).flatten()
        self.nobs = len(self.y)
        if self.nobs < 2 * self.period:
            raise ValueError("Data length must cover at least 2 full periods")

        # Initialize component arrays
        self.trend_arr = np.zeros(self.nobs)
        self.seasonal_arr = np.zeros(self.nobs)
        self.rw = np.ones(self.nobs)  # Robust weights

        # Working array
        self.work = np.zeros((5, self.nobs + 2 * self.period))

        # Set default iteration counts
        if inner_iter is None:
            inner_iter = 1 if self.robust else 2  # 1 inner for robust, 2 for non-robust
        if outer_iter is None:
            outer_iter = (
                10 if self.robust else 0
            )  # 10 outer for robust, 0 for non-robust

        # Initialize components
        self.trend_arr.fill(0)
        self.seasonal_arr.fill(0)
        self.rw.fill(1)

        # Outer loop (robust iterations)
        for k in range(outer_iter + 1):
            # Inner loop (update seasonal and trend components)
            for _ in range(inner_iter):
                self._inner_loop()

            # Update robust weights (except on last iteration)
            if k < outer_iter:
                self._update_robust_weights()

        # Calculate residuals
        resid = self.y - self.trend_arr - self.seasonal_arr

        return STLResult(self.y, self.seasonal_arr, self.trend_arr, resid, self.rw)
