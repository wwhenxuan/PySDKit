# -*- coding: utf-8 -*-
"""
Created on 2025/02/10 13:14:41
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from matplotlib import pyplot as plt

from pysdkit.utils import (
    simple_moving_average,
    weighted_moving_average,
    gaussian_smoothing,
    savgol_smoothing,
    exponential_smoothing,
)

from typing import Optional, Tuple, List


class Moving_Decomp(object):
    """
    Moving Average decomposition.

    The 1D signal is decomposed into two parts, trend and cycle, by sliding average.
    This method is very simple and very suitable for processing non-stationary time series data.
    """

    def __init__(
        self,
        window_size: int = 5,
        method: str = "simple",
        sigma: int = 2,
        poly_order: int = 2,
        alpha: float = 0.4,
    ) -> None:
        """
        The input signal is decomposed by sliding average to obtain the trend and cycle parts.

        :param window_size: The window size of the sliding average decomposition is preferably an odd number
        :param method: Sliding decomposition method, optional ["simple", "weighted", "gaussian", "savgol", "exponential"]
        :param sigma: Standard deviation for Gaussian kernel (default is 2)
        :param poly_order: Order of the polynomial used to fit the samples (default is 2)
        :param alpha: Smoothing factor, range from 0 to 1 (default is 0.4)
        """
        self.window_size = window_size
        self.method = method

        # Specific parameter settings for various methods
        self.sigma = sigma
        self.poly_order = poly_order
        self.alpha = alpha

        # A list of all methods of moving average decomposition
        self.methods_list = ["simple", "weighted", "gaussian", "savgol", "exponential"]
        if self.method not in self.methods_list:
            # Wrong smoothing method used
            raise ValueError("method must be one of {}".format(self.methods_list))

    def __call__(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Moving Average decomposition (Moving_Decomp)"

    def _decomposition(
        self, signal: np.ndarray, method: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute a sliding average decomposition algorithm.
        The input signal must be a univariate signal.

        For multivariate signals, multiple calls are required to decompose

        :param signal: the input univariate signal of 1D numpy ndarray
        :return: the trend and seasonality of the input signal
        """
        # Use a specific moving average decomposition method
        if method == "simple":
            trend = simple_moving_average(signal=signal, window_size=self.window_size)
        elif method == "weighted":
            trend = weighted_moving_average(signal=signal, window_size=self.window_size)
        elif method == "gaussian":
            trend = gaussian_smoothing(signal=signal, sigma=self.sigma)
        elif method == "savgol":
            trend = savgol_smoothing(
                signal=signal,
                window_length=self.window_size,
                poly_order=self.poly_order,
            )
        elif method == "exponential":
            trend = exponential_smoothing(signal=signal, alpha=self.alpha)
        else:
            raise ValueError(
                "method must be 'simple' or 'weighted' or 'gaussian' or 'savgol' or 'exponential'"
            )

        # Subtract the trend component from the original input signal
        seasonality = signal - trend

        # Returns both trend and seasonal components
        return trend, seasonality

    def fit_transform(
        self, signal: np.ndarray, methods_list: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute the moving average decomposition algorithm

        :param signal: the input univariate or multivariate signal of 1D numpy ndarray
        :param methods_list: If it is a multivariate signal or a multivariate time series,
                             a different sliding average method needs to be used for the signal of each channel.
                             This can be done by passing a string that matches the number of input signal channels to the variable.
        :return: The trend and seasonality of the input signal.
        """
        # Verify the dimensionality of the input signal
        shape = signal.shape

        if len(shape) == 1:
            # The input is a one-dimensional univariate signal
            # Choose a specific decomposition method
            method = methods_list[0] if methods_list is not None else self.method
            trend, seasonality = self._decomposition(signal=signal, method=method)

        elif len(shape) == 2:
            # The input is a one-dimensional multivariate signal
            # Get the number of input signals
            n_vars, seq_len = shape

            # Initialize the decomposed array
            trend, seasonality = np.zeros(shape=(n_vars, seq_len)), np.zeros(
                shape=(n_vars, seq_len)
            )

            # The submitted decomposition list must be greater than or equal to the number of channels of the input signal
            if methods_list is not None:
                if len(methods_list) < n_vars:
                    methods_list = methods_list + (
                        [None] * (n_vars - len(methods_list))
                    )

                for n, method in enumerate(methods_list, 0):
                    # Traverse each signal and perform sliding average decomposition
                    # Further determine the specific method to be used
                    if method is None or method not in self.methods_list:
                        method = self.method

                    trend[n, :], seasonality[n, :] = self._decomposition(
                        signal=signal[n, :], method=method
                    )
            else:
                for n in range(n_vars):
                    # Traverse each signal and perform sliding average decomposition
                    trend[n, :], seasonality[n, :] = self._decomposition(
                        signal=signal[n, :], method=self.method
                    )
        else:
            raise ValueError(
                "The input must be 1D univariate or multivariate signal with shape [seq_len] or [n_vars, seq_len]"
            )

        # Return the decomposed result
        return trend, seasonality

    @staticmethod
    def plot_decomposition(
        signal: np.ndarray,
        trend: np.ndarray,
        seasonality: np.ndarray,
        colors: List[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Visualize the decomposition results of the input signal

        :param signal: The input 1D signal of numpy ndarray
        :param trend: The trend of the input 1D signal decomposed by `Moving_Decomp`
        :param seasonality: The seasonality of the input 1D signal decomposed by `Moving_Decomp`
        :param colors: The colors for plotting signal, trend and seasonality
        :return: the figure of matplotlib for plotting
        """
        # Determine the dimensionality of the input data by its shape
        if colors is None:
            colors = ["royalblue", "royalblue", "royalblue"]
        shape = signal.shape

        # If the inputs is univariate signal
        if len(shape) == 1:
            # Creating a drawing object
            fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 5), sharex=True)
            # Start drawing the image
            ax[0].plot(signal, color=colors[0])
            ax[1].plot(trend, color=colors[1])
            ax[2].plot(seasonality, color=colors[2])

            ax[0].set_ylabel("input signal")
            ax[1].set_ylabel("trend")
            ax[2].set_ylabel("seasonality")

        elif len(shape) == 2:
            # Get the number of input signals
            n_vars, seq_len = shape
            # Creating a drawing object
            fig, ax = plt.subplots(
                nrows=3, ncols=n_vars, figsize=(4 * n_vars, 5), sharex=True
            )
            # Iterate over all dimensions of variables and draw images
            for n in range(n_vars):
                ax[0, n].plot(signal[n, :], color=colors[n])
                ax[1, n].plot(trend[n, :], color=colors[n])
                ax[2, n].plot(seasonality[n, :], color=colors[n])

                ax[0, n].set_title(f"Input channels {n}")

            ax[0, 0].set_ylabel("input signal")
            ax[1, 0].set_ylabel("trend")
            ax[2, 0].set_ylabel("seasonality")

        else:
            raise ValueError(
                "The input must be 1D univariate or multivariate signal with shape [seq_len] or [n_vars, seq_len]"
            )

        return fig


if __name__ == "__main__":
    from pysdkit.data import generate_time_series

    # univariate time series
    time_series = generate_time_series()

    moving_decomp = Moving_Decomp(window_size=5)

    trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

    moving_decomp.plot_decomposition(
        signal=time_series, trend=trends, seasonality=seasonalities
    )
    plt.show()

    # multivariate time series
    time_series = np.vstack([time_series] * 3)

    trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

    moving_decomp.plot_decomposition(
        signal=time_series, trend=trends, seasonality=seasonalities
    )
    plt.show()
