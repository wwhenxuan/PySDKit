# -*- coding: utf-8 -*-
"""
Created on 2025/02/12 00:14:21
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from typing import Optional, Literal
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

    RB C. STL: A seasonal-trend decomposition procedure based on loess[J]. J Off Stat, 1990, 6: 3-73.

    STL uses LOESS (locally estimated scatterplot smoothing) to extract smooths estimates of the three components.
    The key inputs into STL are:
    (1) season - The length of the seasonal smoother. Must be odd.
    (2) trend - The length of the trend smoother, usually around 150% of season. Must be odd and larger than season.
    (3) low_pass - The length of the low-pass estimation window, usually the smallest odd number larger than the periodicity of the data.
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
        :param seasonal: Seasonal smoothing window size (odd, >=7)
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
        if not isinstance(period, int) or period < 2:
            raise ValueError("period must be an integer >= 2")
        self.period = period

        # Validate and set seasonal parameters
        if not self._is_odd_int(seasonal) or seasonal < 3:
            raise ValueError("seasonal must be an odd integer >= 3")
        self.seasonal = seasonal

        # Calculate trend window size
        if trend is None:
            # Ensure denominator is positive
            denom = max(1 - 1.5 / seasonal, 0.01)
            trend = int(np.ceil(1.5 * period / denom))
            # Ensure it's odd
            trend = trend + 1 if trend % 2 == 0 else trend
        if not self._is_odd_int(trend) or trend < period:
            raise ValueError("trend must be an odd integer >= period length")
        self.trend = trend

        # Calculate low-pass filter window size
        if low_pass is None:
            low_pass = period + (1 if period % 2 == 0 else 0)
            low_pass = low_pass + 1 if low_pass % 2 == 0 else low_pass
        if not self._is_odd_int(low_pass) or low_pass < period:
            raise ValueError("low_pass must be an odd integer >= period length")
        self.low_pass = low_pass

        # Set other parameters
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
        self.seasonal_jump = max(1, seasonal_jump)
        self.trend_jump = max(1, trend_jump)
        self.low_pass_jump = max(1, low_pass_jump)

    @staticmethod
    def _is_odd_int(x: int) -> bool:
        """Check if value is an odd integer"""
        return isinstance(x, int) and x > 0 and x % 2 == 1

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

        # Process each seasonal position
        for j in range(self.period):
            # Get subseries for current seasonal position
            subseries = detrended[j :: self.period]
            n_sub = len(subseries)

            # Get corresponding robust weights
            weights = self.rw[j :: self.period] if self.robust else np.ones(n_sub)

            # Apply LOESS smoothing
            smoothed = self._loess(
                x=np.arange(n_sub),
                y=subseries,
                weights=weights,
                window_size=self.seasonal,
                degree=self.seasonal_deg,
                jump=self.seasonal_jump,
            )

            # Store results (including boundary points)
            start_idx = j
            for i in range(
                -1, n_sub + 1
            ):  # Include left boundary(-1) and right boundary(n_sub)
                idx = start_idx + i * self.period
                if 0 <= idx < len(seasonal_temp):
                    # Boundary points use extrapolated values
                    if i == -1:
                        seasonal_temp[idx] = smoothed[0]  # Left boundary
                    elif i == n_sub:
                        seasonal_temp[idx] = smoothed[-1]  # Right boundary
                    else:
                        seasonal_temp[idx] = smoothed[i]

        return seasonal_temp

    def _low_pass_filter(self, seasonal_temp: np.ndarray) -> np.ndarray:
        """
        Low-pass filtering

        :param seasonal_temp: Temporary seasonal component
        :return: Low-frequency component (length = nobs)
        """
        n = self.nobs + 2 * self.period

        # First moving average (length=period)
        ma1 = self._moving_average(seasonal_temp, self.period)

        # Second moving average (length=period)
        ma2 = self._moving_average(ma1, self.period)

        # Third moving average (length=3)
        ma3 = self._moving_average(ma2, 3)

        # Apply LOESS smoothing
        low_pass = self._loess(
            x=np.arange(len(ma3)),
            y=ma3,
            window_size=self.low_pass,
            degree=self.low_pass_deg,
            jump=self.low_pass_jump,
        )

        # Trim to original data length (remove boundary effects)
        return low_pass[: self.nobs]

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
            result[i] = self._loess_point(x, y, i, window_size, degree, weights)

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
    def _loess_point(
        x: np.ndarray,
        y: np.ndarray,
        idx: int,
        window_size: int,
        degree: int,
        weights: np.ndarray,
    ) -> float:
        """
        Single-point LOESS calculation

        :param x: Independent variable
        :param y: Dependent variable
        :param idx: Target point index
        :param window_size: Window size
        :param degree: Polynomial degree
        :param weights: Observation weights
        :return: Fitted value
        """
        # 1. Calculate distances and determine window
        distances = np.abs(x - x[idx])
        max_distance = np.partition(distances, window_size - 1)[window_size - 1]

        # Handle case where window is larger than data length
        if window_size > len(x):
            max_distance = np.max(distances) * window_size / len(x)

        # 2. Calculate tricube weights
        d_scaled = distances / (max_distance + 1e-12)
        weights_tricube = np.where(d_scaled < 1, (1 - d_scaled**3) ** 3, 0)

        # Combine observation weights
        weights_total = weights * weights_tricube

        # 3. Weighted least squares fitting
        if degree == 0:  # Constant fit
            weight_sum = np.sum(weights_total)
            if weight_sum < 1e-12:
                return y[idx]  # Fallback to original value
            return np.sum(weights_total * y) / weight_sum

        elif degree == 1:  # Linear fit
            # Construct design matrix [1, x]
            X = np.column_stack((np.ones_like(x), x - x[idx]))

            # Weighted least squares
            X_weighted = X * np.sqrt(weights_total)[:, np.newaxis]
            y_weighted = y * np.sqrt(weights_total)

            # Solve
            try:
                beta, _, _, _ = lstsq(X_weighted, y_weighted, lapack_driver="gelsy")
                return beta[0]  # Constant term is the fitted value
            except:
                return y[idx]  # Fallback if solution fails

        else:
            raise ValueError("Only degrees 0 or 1 are supported")

    @staticmethod
    def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """
        Moving average

        :param data: Input data
        :param window: Window size
        :return: Moving average result
        """
        n = len(data)
        if window > n:
            return np.full(n, np.mean(data))

        # Calculate cumulative sum
        cumsum = np.cumsum(np.insert(data, 0, 0))
        result = (cumsum[window:] - cumsum[:-window]) / window

        # Boundary handling (padding ends)
        pad = window - 1
        return np.pad(result, (pad // 2, pad - pad // 2), "edge")

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
