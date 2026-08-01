# -*- coding: utf-8 -*-
"""
Unit tests for Multiple Seasonal-Trend decomposition using LOESS (MSTL).
"""

import unittest
import warnings

import numpy as np

from pysdkit import MSTL
from pysdkit.tsa import MSTLResult, STL


def _multi_seasonal_series(
    n: int = 500,
    periods=(24, 168),
    seed: int = 0,
) -> tuple:
    """Synthetic series with trend + two seasonal sine waves + noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    trend = 0.0001 * t**2 + 50.0
    seas = [amp * np.sin(2 * np.pi * t / p) for p, amp in zip(periods, (5.0, 10.0))]
    noise = 0.3 * rng.standard_normal(n)
    y = trend + sum(seas) + noise
    return y, trend, seas


class TestMSTL(unittest.TestCase):
    """Automated tests for MSTL."""

    def setUp(self) -> None:
        self.periods = (24, 24 * 7)
        self.y, self.true_trend, self.true_seas = _multi_seasonal_series(
            n=500, periods=self.periods, seed=0
        )

    def test_fit_transform_shape(self) -> None:
        """Verify output shapes for two seasonal periods."""
        mstl = MSTL(periods=self.periods, iterate=2)
        result = mstl.fit_transform(self.y)

        self.assertIsInstance(result, MSTLResult)
        self.assertEqual(result.seasonal.shape, (2, self.y.size))
        self.assertEqual(result.trend.shape, self.y.shape)
        self.assertEqual(result.resid.shape, self.y.shape)
        self.assertEqual(result.periods, tuple(sorted(self.periods)))

    def test_reconstruction(self) -> None:
        """observed ≈ sum(seasonals) + trend + resid on the transformed scale."""
        result = MSTL(periods=self.periods).fit_transform(self.y)
        reconstructed = result.seasonal.sum(axis=0) + result.trend + result.resid
        self.assertTrue(np.allclose(result.observed, reconstructed, atol=1e-8))

    def test_default_call(self) -> None:
        """__call__ should match fit_transform."""
        mstl = MSTL(periods=self.periods)
        a = mstl(self.y)
        b = mstl.fit_transform(self.y)
        self.assertTrue(np.allclose(a.trend, b.trend))
        self.assertTrue(np.allclose(a.seasonal, b.seasonal))
        self.assertTrue(np.allclose(a.resid, b.resid))

    def test_str(self) -> None:
        self.assertIn("MSTL", str(MSTL(periods=24)))

    def test_single_period_matches_stl_identity(self) -> None:
        """With one period, MSTL should still reconstruct exactly."""
        period = 24
        y, _, _ = _multi_seasonal_series(n=240, periods=(period, 48), seed=1)
        # Keep only daily-like oscillation for a cleaner single-period case
        t = np.arange(y.size, dtype=float)
        y = 0.01 * t + 3 * np.sin(2 * np.pi * t / period) + 0.1 * np.random.default_rng(0).standard_normal(y.size)

        result = MSTL(periods=period, iterate=5).fit_transform(y)
        self.assertEqual(result.seasonal.ndim, 1)
        self.assertEqual(result.seasonal.shape, y.shape)
        reconstructed = result.seasonal + result.trend + result.resid
        self.assertTrue(np.allclose(result.observed, reconstructed, atol=1e-8))

    def test_default_windows(self) -> None:
        """Paper default windows are 7 + 4*i."""
        mstl = MSTL(periods=(24, 168))
        _, windows = mstl._prepare_periods_windows(nobs=500)
        self.assertEqual(windows, (11, 15))

    def test_periods_sorted_ascending(self) -> None:
        """Longer periods should not be extracted before shorter ones."""
        result = MSTL(periods=(168, 24)).fit_transform(self.y)
        self.assertEqual(result.periods, (24, 168))

    def test_custom_windows(self) -> None:
        result = MSTL(periods=(24, 168), windows=(11, 15)).fit_transform(self.y)
        reconstructed = result.seasonal.sum(0) + result.trend + result.resid
        self.assertTrue(np.allclose(result.observed, reconstructed, atol=1e-8))

    def test_recovers_seasonal_energy(self) -> None:
        """Extracted seasonals should correlate with the planted components."""
        result = MSTL(periods=self.periods, iterate=2, stl_kwargs={"seasonal_deg": 0}).fit_transform(
            self.y
        )
        # periods are sorted ascending -> seasonal[0] ~ daily, seasonal[1] ~ weekly
        corr_daily = np.corrcoef(result.seasonal[0], self.true_seas[0])[0, 1]
        corr_weekly = np.corrcoef(result.seasonal[1], self.true_seas[1])[0, 1]
        self.assertGreater(corr_daily, 0.8, msg=f"daily corr={corr_daily}")
        self.assertGreater(corr_weekly, 0.8, msg=f"weekly corr={corr_weekly}")

        corr_trend = np.corrcoef(result.trend, self.true_trend)[0, 1]
        self.assertGreater(corr_trend, 0.9, msg=f"trend corr={corr_trend}")

    def test_boxcox_auto(self) -> None:
        """Box-Cox('auto') should run on positive data and reconstruct."""
        y = self.y - self.y.min() + 1.0
        result = MSTL(periods=self.periods, lmbda="auto", iterate=1).fit_transform(y)
        self.assertIsNotNone(result.lmbda)
        reconstructed = result.seasonal.sum(0) + result.trend + result.resid
        self.assertTrue(np.allclose(result.observed, reconstructed, atol=1e-8))

    def test_boxcox_rejects_nonpositive(self) -> None:
        y = self.y.copy()
        y[0] = -1.0
        with self.assertRaises(ValueError):
            MSTL(periods=24, lmbda=0.0).fit_transform(y)

    def test_invalid_periods(self) -> None:
        with self.assertRaises(ValueError):
            MSTL(periods=1)
        with self.assertRaises(ValueError):
            MSTL(periods=(24, 24), windows=(11,))

    def test_invalid_windows(self) -> None:
        with self.assertRaises(ValueError):
            MSTL(periods=(24, 168), windows=(10, 15))  # even window

    def test_long_period_warning(self) -> None:
        y = np.random.default_rng(0).standard_normal(50)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(ValueError):
                # both periods >= n/2 -> nothing left
                MSTL(periods=(40, 45)).fit_transform(y)
            self.assertTrue(any(issubclass(w.category, UserWarning) for w in caught))

    def test_stl_kwargs_robust(self) -> None:
        y = self.y.copy()
        y[100] += 20.0
        result = MSTL(
            periods=self.periods, iterate=1, stl_kwargs={"robust": True}
        ).fit_transform(y)
        self.assertTrue(np.all(np.isfinite(result.resid)))
        self.assertEqual(result.weights.shape, y.shape)

    def test_uses_pysdkit_stl(self) -> None:
        """MSTL must be built on the local STL class (no statsmodels dependency)."""
        self.assertIs(MSTL.__init__.__globals__.get("STL", STL), STL)


if __name__ == "__main__":
    unittest.main()
