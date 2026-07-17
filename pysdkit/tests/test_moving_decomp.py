# -*- coding: utf-8 -*-
"""
Created on 2025/02/15 23:21:53
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np
from matplotlib import pyplot as plt
from numpy.ma.core import shape

from pysdkit import Moving_Decomp
from pysdkit.data import generate_time_series


class MDTest(unittest.TestCase):
    """Moving_Decomp test"""

    def test_univariate_fit_transform(self):
        """Verify decomposition of univariate signals or time series"""
        # Create a decomposition instance
        moving_decomp = Moving_Decomp(window_size=5)

        # univariate time series
        time_series = generate_time_series()
        trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

        # Verify univariate time series output
        self.assertEqual(
            first=len(time_series),
            second=len(trends),
            msg="Input and output signal lengths do not match",
        )
        self.assertEqual(
            first=len(time_series),
            second=len(seasonalities),
            msg="Input and output signal lengths do not match",
        )

        # Further verify numerical reconstruction
        diff = np.allclose(time_series, trends + seasonalities)
        self.assertTrue(
            expr=diff,
            msg="Decomposed trend and seasonality cannot reconstruct the original signal",
        )

    def test_multivariate_fit_transform(self):
        """Verify decomposition of multivariate signals or time series"""
        # Create a decomposition instance
        moving_decomp = Moving_Decomp(window_size=5)

        # multivariate time series
        time_series = np.vstack(
            [
                generate_time_series(periodicities=np.array(p_list))
                for p_list in [[10, 20, 30], [5, 20, 50], [40, 60, 10]]
            ]
        )

        trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

        # Verify that input and output shapes match
        self.assertEqual(
            first=trends.shape,
            second=time_series.shape,
            msg="Trend shape should match the input series",
        )
        self.assertEqual(
            first=seasonalities.shape,
            second=time_series.shape,
            msg="Seasonality shape should match the input series",
        )

        # Further verify numerical reconstruction
        for index in range(3):
            diff = np.allclose(time_series[index], trends[index] + seasonalities[index])
            self.assertTrue(
                expr=diff,
                msg="Decomposed trend and seasonality cannot reconstruct the original signal",
            )

    def test_default_call(self) -> None:
        """Verify that the call method can run normally"""
        # Create a decomposition instance
        moving_decomp = Moving_Decomp(window_size=5)

        # univariate time series
        time_series = generate_time_series()
        trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

        # Verify univariate time series output
        self.assertEqual(
            first=len(time_series),
            second=len(trends),
            msg="Input and output signal lengths do not match",
        )
        self.assertEqual(
            first=len(time_series),
            second=len(seasonalities),
            msg="Input and output signal lengths do not match",
        )

        # Further verify numerical reconstruction
        diff = np.allclose(time_series, trends + seasonalities)
        self.assertTrue(
            expr=diff,
            msg="Decomposed trend and seasonality cannot reconstruct the original signal",
        )

    def test_different_window_size(self) -> None:
        """Verify the effect of different moving-average window sizes"""
        # univariate time series
        time_series = generate_time_series()

        # List of different window sizes
        window_size_list = [3, 5, 7, 9, 15, 25, 35]

        for window_size in window_size_list:
            # Create a decomposition instance for each window size
            moving_decomp = Moving_Decomp(window_size=window_size)
            trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

            # Verify univariate time series output
            self.assertEqual(
                first=len(time_series),
                second=len(trends),
                msg="Input and output signal lengths do not match",
            )
            self.assertEqual(
                first=len(time_series),
                second=len(seasonalities),
                msg="Input and output signal lengths do not match",
            )

            # Further verify numerical reconstruction
            diff = np.allclose(time_series, trends + seasonalities)
            self.assertTrue(
                expr=diff,
                msg="Decomposed trend and seasonality cannot reconstruct the original signal",
            )

    def test_different_method(self) -> None:
        """Verify different moving-average decomposition methods"""
        # univariate time series
        time_series = generate_time_series()

        # List of different methods
        methods = ["simple", "weighted", "gaussian", "savgol", "exponential"]

        for method in methods:
            # Create a decomposition instance for each method
            moving_decomp = Moving_Decomp(method=method)
            trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

            # Verify univariate time series output
            self.assertEqual(
                first=len(time_series),
                second=len(trends),
                msg="Input and output signal lengths do not match",
            )
            self.assertEqual(
                first=len(time_series),
                second=len(seasonalities),
                msg="Input and output signal lengths do not match",
            )

            # Further verify numerical reconstruction
            diff = np.allclose(time_series, trends + seasonalities)
            self.assertTrue(
                expr=diff,
                msg="Decomposed trend and seasonality cannot reconstruct the original signal",
            )

    def test_wrong_method(self) -> None:
        """Verify invalid decomposition methods"""
        method = "wrong"
        # univariate time series
        time_series = generate_time_series()

        with self.assertRaises(ValueError):
            moving_decomp = Moving_Decomp(method=method)
            moving_decomp.fit_transform(signal=time_series)

    def test_methods_list(self) -> None:
        """Verify passing a list of decomposition methods to fit_transform"""
        moving_decomp = Moving_Decomp()

        # univariate time series
        time_series = generate_time_series()
        # multivariate time series
        time_series = np.vstack([time_series] * 3)

        trends, seasonalities = moving_decomp.fit_transform(
            signal=time_series, methods_list=["simple", "weighted", "gaussian"]
        )

        # Verify that input and output shapes match
        self.assertEqual(
            first=trends.shape,
            second=time_series.shape,
            msg="Trend shape should match the input series",
        )
        self.assertEqual(
            first=seasonalities.shape,
            second=time_series.shape,
            msg="Seasonality shape should match the input series",
        )

        # Further verify numerical reconstruction
        for index in range(3):
            diff = np.allclose(time_series[index], trends[index] + seasonalities[index])
            self.assertTrue(
                expr=diff,
                msg="Decomposed trend and seasonality cannot reconstruct the original signal",
            )

    def test_univariate_plotting(self) -> None:
        """Verify the decomposition plotting function"""
        # Randomly generate a signal
        time = np.linspace(0, 1, 100)
        time_series = 2 * time + np.cos(time * 2 * np.pi)

        # Decompose the time series
        moving_decomp = Moving_Decomp()
        trends, seasonalities = moving_decomp.fit_transform(signal=time_series)

        # Plot the decomposition
        fig = moving_decomp.plot_decomposition(
            signal=time_series, trend=trends, seasonality=seasonalities
        )
        self.assertTrue(expr=isinstance(fig, plt.Figure))

    def test_multivariate_plotting(self) -> None:
        """Verify plotting for multivariate time series"""
        # Randomly generate a multivariate series
        time_series = np.random.rand(3, 100)

        # Decompose the time series
        moving_decomp = Moving_Decomp()

        # Plot the decomposition
        fig = moving_decomp.plot_decomposition(
            signal=time_series, trend=time_series, seasonality=time_series
        )
        self.assertTrue(expr=isinstance(fig, plt.Figure))

    def test_wrong_plotting(self) -> None:
        """Verify plotting with incorrectly shaped inputs"""
        wrong_inputs = np.random.rand(3, 3, 3)

        # Decompose the time series
        moving_decomp = Moving_Decomp()

        # Plot the decomposition
        with self.assertRaises(ValueError):
            moving_decomp.plot_decomposition(
                signal=wrong_inputs, trend=wrong_inputs, seasonality=wrong_inputs
            )


if __name__ == "__main__":
    unittest.main()
