# -*- coding: utf-8 -*-
"""
Unit tests for Seasonal-Trend decomposition using LOESS (STL).
"""

import unittest

import numpy as np

from pysdkit import STL
from pysdkit.tsa import STLResult
from pysdkit.data import generate_time_series


class TestSTL(unittest.TestCase):
    """Test whether STL decomposition runs normally."""

    def setUp(self) -> None:
        """Create time series data for testing."""
        np.random.seed(42)
        self.period = 12
        self.data = generate_time_series(
            duration=240,
            periodicities=np.array([self.period]),
            num_harmonics=np.array([2]),
            std=np.array([0.5]),
            seed=42,
        )
        self.trend = np.linspace(0, 10, len(self.data))
        self.data = self.data + self.trend

    def test_fit_transform(self) -> None:
        """Verify reconstruction identity observed = seasonal + trend + resid."""
        stl = STL(period=self.period)
        result = stl.fit_transform(self.data)

        self.assertIsInstance(result, STLResult)
        reconstructed = result.seasonal + result.trend + result.resid
        self.assertTrue(
            np.allclose(result.observed, reconstructed, atol=1e-10),
            "Reconstructed series after decomposition does not match",
        )
        self.assertEqual(len(result.observed), len(self.data))
        self.assertTrue(np.all(np.isfinite(result.seasonal)))
        self.assertTrue(np.all(np.isfinite(result.trend)))
        self.assertTrue(np.all(np.isfinite(result.resid)))

    def test_default_call(self) -> None:
        """Verify that __call__ matches fit_transform."""
        stl = STL(period=self.period)
        a = stl(self.data)
        b = stl.fit_transform(self.data)

        self.assertIsInstance(a, STLResult)
        self.assertEqual(len(a.observed), len(self.data))
        self.assertTrue(np.allclose(a.seasonal, b.seasonal))
        self.assertTrue(np.allclose(a.trend, b.trend))
        self.assertTrue(np.allclose(a.resid, b.resid))

    def test_str(self) -> None:
        """Verify the human-readable algorithm name."""
        self.assertIn("STL", str(STL(period=12)))

    def test_robust_mode(self) -> None:
        """Verify that robust and non-robust modes both reconstruct."""
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
            self.assertTrue(np.all(np.isfinite(result.resid[outlier_indices])))
            self.assertEqual(result.weights.shape, result.observed.shape)

    def test_seasonal_component(self) -> None:
        """Verify that the seasonal component shows the expected periodicity."""
        stl = STL(period=self.period, seasonal=7)
        result = stl(self.data)

        seasonal = result.seasonal
        autocorr = np.correlate(seasonal, seasonal, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / (autocorr[0] + 1e-12)

        self.assertGreater(autocorr[0], autocorr[1])
        self.assertGreater(
            autocorr[self.period],
            autocorr[self.period // 2],
            "Seasonal component periodicity is not evident",
        )

    def test_trend_component(self) -> None:
        """Verify smoothness / correlation of the recovered trend."""
        stl = STL(period=self.period, trend=25)
        result = stl(self.data)

        trend_diff = np.diff(result.trend)
        data_diff = np.diff(self.data)
        self.assertLess(
            np.std(trend_diff),
            np.std(data_diff),
            "Trend component is not smooth enough",
        )

        corr = np.corrcoef(result.trend, self.trend)[0, 1]
        self.assertGreater(
            corr,
            0.9,
            "Trend component correlation with expected trend is insufficient",
        )

    def test_parameter_validation(self) -> None:
        """Verify parameter validation logic."""
        with self.assertRaises(ValueError):
            STL(period=1)

        # seasonal must be odd and >= 3
        with self.assertRaises(ValueError):
            STL(period=12, seasonal=4)
        with self.assertRaises(ValueError):
            STL(period=12, seasonal=2)

        # trend must be odd and > period
        with self.assertRaises(ValueError):
            STL(period=12, trend=12)
        with self.assertRaises(ValueError):
            STL(period=12, trend=11)

        # insufficient length
        stl = STL(period=12)
        with self.assertRaises(ValueError):
            stl.fit_transform(np.random.rand(10))

    def test_different_iterations(self) -> None:
        """Verify that different iteration counts still yield valid decompositions."""
        configs = [
            (STL(period=self.period), dict()),
            (STL(period=self.period), dict(inner_iter=5)),
            (STL(period=self.period, robust=True), dict(outer_iter=5)),
        ]
        for stl, kwargs in configs:
            result = stl.fit_transform(self.data, **kwargs)
            reconstructed = result.seasonal + result.trend + result.resid
            self.assertTrue(np.allclose(result.observed, reconstructed, atol=1e-10))
            self.assertTrue(np.all(np.isfinite(result.resid)))

    def test_mean_seasonal_near_zero(self) -> None:
        """Seasonal component should be approximately zero-mean after low-pass removal."""
        result = STL(period=self.period, seasonal=13).fit_transform(self.data)
        self.assertLess(abs(np.mean(result.seasonal)), 1.0)


if __name__ == "__main__":
    unittest.main()
