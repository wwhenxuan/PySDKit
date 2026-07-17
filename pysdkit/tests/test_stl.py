# -*- coding: utf-8 -*-
import unittest
import numpy as np
from pysdkit import STL
from pysdkit.tsa import STLResult
from pysdkit.data import generate_time_series


class TestSTL(unittest.TestCase):
    """Test whether STL decomposition runs normally"""

    def setUp(self) -> None:
        """Create time series data for testing"""
        np.random.seed(42)
        # Generate a base time series
        self.period = 12  # seasonal period
        self.data = generate_time_series(
            duration=120,
            periodicities=np.array([self.period]),
            num_harmonics=np.array([2]),
            std=np.array([0.5]),
        )
        # Add a linear trend
        self.trend = np.linspace(0, 10, len(self.data))
        self.data += self.trend

    def test_fit_transform(self) -> None:
        """Verify that time series decomposition runs normally"""
        # Create an algorithm instance object
        stl = STL(period=self.period)
        result = stl.fit_transform(self.data)

        # Verify the return type
        self.assertIsInstance(result, STLResult, "Wrong return result type")

        # Verify component relationship: observed = seasonal + trend + resid
        reconstructed = result.seasonal + result.trend + result.resid
        self.assertTrue(
            np.allclose(result.observed, reconstructed, atol=1e-10),
            "Reconstructed series after decomposition does not match",
        )

    def test_default_call(self) -> None:
        """Verify that the call method can run normally"""
        # Create an algorithm instance object
        stl = STL(period=self.period)
        result = stl(self.data)

        # Verify basic output
        self.assertIsInstance(result, STLResult, "Wrong return result type")
        self.assertEqual(len(result.observed), len(self.data), "Data length mismatch")

    def test_robust_mode(self) -> None:
        """Verify that robust and non-robust modes both run and reconstruct."""
        data_with_outliers = self.data.copy()
        outlier_indices = [10, 30, 50, 70, 90]
        data_with_outliers[outlier_indices] += 10.0

        for robust in (False, True):
            stl = STL(period=self.period, robust=robust)
            result = stl(data_with_outliers)
            reconstructed = result.seasonal + result.trend + result.resid
            self.assertTrue(
                np.allclose(result.observed, reconstructed, atol=1e-10),
                f"Reconstruction failed for robust={robust}",
            )
            self.assertEqual(len(result.resid), len(data_with_outliers))
            # Outlier residuals should remain finite in both modes
            self.assertTrue(np.all(np.isfinite(result.resid[outlier_indices])))

    def test_seasonal_component(self) -> None:
        """Verify periodicity of the seasonal component"""
        stl = STL(period=self.period, seasonal=7)
        result = stl(self.data)

        # Verify seasonal component periodicity
        seasonal = result.seasonal
        autocorr = np.correlate(seasonal, seasonal, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        # Peaks should appear at period points
        self.assertGreater(
            autocorr[0], autocorr[1], "Wrong autocorrelation peak in seasonal component"
        )
        self.assertGreater(
            autocorr[self.period],
            autocorr[self.period - 1],
            "Seasonal component periodicity is not evident",
        )

    def test_trend_component(self) -> None:
        """Verify smoothness of the trend component"""
        stl = STL(period=self.period, trend=25)
        result = stl(self.data)

        # Compute first differences of the trend component
        trend_diff = np.diff(result.trend)

        # Trend component should be smoother than the raw data
        data_diff = np.diff(self.data)
        self.assertLess(
            np.std(trend_diff),
            np.std(data_diff),
            "Trend component is not smooth enough",
        )

        # Trend component should correlate with the added linear trend
        corr = np.corrcoef(result.trend, self.trend)[0, 1]
        self.assertGreater(
            corr, 0.9, "Trend component correlation with expected trend is insufficient"
        )

    def test_parameter_validation(self) -> None:
        """Verify parameter validation logic"""
        # Invalid period
        with self.assertRaises(ValueError):
            STL(period=1)

        # Invalid seasonal parameter
        with self.assertRaises(ValueError):
            STL(period=12, seasonal=4)  # must be >= 7 and odd

        # Insufficient data length
        stl = STL(period=12)
        with self.assertRaises(ValueError):
            stl.fit_transform(np.random.rand(10))  # 10 < 2*12

    def test_different_iterations(self) -> None:
        """Verify that different iteration counts still yield valid decompositions."""
        configs = [
            STL(period=self.period),
            STL(period=self.period),
            STL(period=self.period, robust=True),
        ]
        calls = [
            lambda stl: stl(self.data),
            lambda stl: stl.fit_transform(self.data, inner_iter=5),
            lambda stl: stl.fit_transform(self.data, outer_iter=15),
        ]

        for stl, call in zip(configs, calls):
            result = call(stl)
            reconstructed = result.seasonal + result.trend + result.resid
            self.assertTrue(
                np.allclose(result.observed, reconstructed, atol=1e-10),
                "Reconstructed series after decomposition does not match",
            )
            self.assertEqual(len(result.observed), len(self.data))
            self.assertTrue(np.all(np.isfinite(result.resid)))


if __name__ == "__main__":
    unittest.main()
