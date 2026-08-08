# -*- coding: utf-8 -*-
"""
Unit tests for Optimization-based Signal Decomposition (OSD).
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit import OSD
from pysdkit._osd import (
    FiniteSet,
    MeanSquareSmall,
    SmoothSecondDifference,
    Sparse,
    SparseFirstDiffConvex,
    SparseSecondDiffConvex,
)
from pysdkit._osd.components import difference_matrix, soft_threshold


def _sine_square(n: int = 400, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    sine = np.sin(2.0 * np.pi * 3.0 * t)
    square = np.sign(np.sin(2.0 * np.pi * 1.5 * t))
    noise = 0.05 * rng.standard_normal(n)
    return t, sine + square + noise, sine, square


class HelperTest(unittest.TestCase):
    def test_soft_threshold(self) -> None:
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        y = soft_threshold(x, 1.0)
        self.assertTrue(np.allclose(y, np.array([-1.0, 0.0, 0.0, 0.0, 1.0])))

    def test_difference_matrix_shapes(self) -> None:
        d1 = difference_matrix(10, 1)
        d2 = difference_matrix(10, 2)
        self.assertEqual(d1.shape, (9, 10))
        self.assertEqual(d2.shape, (8, 10))
        x = np.arange(10.0)
        self.assertTrue(np.allclose(d1 @ x, np.diff(x)))
        self.assertTrue(np.allclose(d2 @ x, np.diff(x, n=2)))


class ComponentProxTest(unittest.TestCase):
    def test_mean_square_shrinks(self) -> None:
        c = MeanSquareSmall(size=100, weight=1.0)
        v = np.ones(100)
        out = c.prox_op(v, weight=1.0, rho=2.0)
        self.assertTrue(np.all(np.abs(out) < np.abs(v) + 1e-12))
        self.assertTrue(np.allclose(out, out[0]))

    def test_smooth_diff_reduces_roughness(self) -> None:
        rng = np.random.default_rng(0)
        v = rng.standard_normal(128)
        c = SmoothSecondDifference(weight=10.0)
        out = c.prox_op(v, weight=10.0, rho=1.0)
        self.assertLess(c.cost(out), c.cost(v))

    def test_sparse_first_diff_piecewise(self) -> None:
        v = np.r_[np.zeros(40), np.ones(40), np.zeros(40)]
        v = v + 0.05 * np.random.default_rng(1).standard_normal(v.size)
        c = SparseFirstDiffConvex(weight=0.5, vmin=-0.1, vmax=1.1)
        out = c.prox_op(v, weight=0.5, rho=1.0)
        self.assertEqual(out.shape, v.shape)
        self.assertTrue(np.all(out >= -0.1 - 1e-8))
        self.assertTrue(np.all(out <= 1.1 + 1e-8))

    def test_sparse_second_diff_finite(self) -> None:
        v = np.linspace(0, 1, 100) + 0.1 * np.sin(np.linspace(0, 12, 100))
        c = SparseSecondDiffConvex(weight=1.0)
        out = c.prox_op(v, weight=1.0, rho=1.0)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_sparse_spikes(self) -> None:
        v = np.zeros(50)
        v[[10, 25, 40]] = [3.0, -2.5, 4.0]
        c = Sparse(weight=1.0)
        out = c.prox_op(v, weight=1.0, rho=1.0)
        self.assertLess(np.count_nonzero(out), np.count_nonzero(v) + 1)

    def test_finite_set_projection(self) -> None:
        c = FiniteSet(values=(-1.0, 1.0))
        v = np.array([-0.8, 0.1, 0.9, -0.2])
        out = c.prox_op(v, 1.0, 1.0)
        self.assertTrue(np.all(np.isin(out, [-1.0, 1.0])))
        self.assertFalse(c.is_convex)


class OSDTest(unittest.TestCase):
    def test_str(self) -> None:
        self.assertIn("OSD", str(OSD.preset("l1_trend", 50)))

    def test_preset_convex_demo_shape(self) -> None:
        _, y, _, _ = _sine_square(n=256)
        osd = OSD.preset("convex_demo", length=y.size, max_iter=40)
        X = osd.fit_transform(y)
        self.assertEqual(X.shape, (3, y.size))
        self.assertTrue(np.allclose(X.sum(0), y, atol=1e-8))

    def test_default_call(self) -> None:
        _, y, _, _ = _sine_square(n=200)
        osd = OSD.preset("convex_demo", length=y.size, max_iter=30)
        a = osd(y)
        b = osd.fit_transform(y)
        self.assertEqual(a.shape, b.shape)

    def test_l1_trend_reconstruction(self) -> None:
        n = 300
        t = np.linspace(0, 1, n)
        # piecewise-linear trend + noise
        trend = np.piecewise(
            t,
            [t < 0.4, (t >= 0.4) & (t < 0.7), t >= 0.7],
            [lambda u: 2 * u, lambda u: 0.8 - 0.5 * u, lambda u: 1.5 * u - 1.0],
        )
        y = trend + 0.05 * np.random.default_rng(2).standard_normal(n)
        osd = OSD.preset("l1_trend", length=n, max_iter=60)
        X = osd.fit_transform(y)
        self.assertEqual(X.shape[0], 2)
        self.assertTrue(np.allclose(X.sum(0), y, atol=1e-8))
        # recovered trend should be closer to truth than raw y
        err_raw = np.linalg.norm(y - trend)
        err_hat = np.linalg.norm(X[1] - trend)
        self.assertLess(err_hat, err_raw)

    def test_nonconvex_preset(self) -> None:
        _, y, _, square = _sine_square(n=220)
        osd = OSD.preset("nonconvex_square", length=y.size, max_iter=50)
        X = osd.fit_transform(y)
        self.assertEqual(X.shape[0], 3)
        self.assertTrue(np.allclose(X.sum(0), y, atol=1e-8))
        # finite-set component should be nearly binary
        uniq = np.unique(np.round(X[2], 5))
        self.assertTrue(set(uniq).issubset({-1.0, 1.0}))

    def test_missing_entries(self) -> None:
        _, y, _, _ = _sine_square(n=180)
        y_miss = y.copy()
        y_miss[20:35] = np.nan
        osd = OSD.preset("l1_trend", length=y.size, max_iter=40)
        X = osd.fit_transform(y_miss)
        known = ~np.isnan(y_miss)
        self.assertTrue(np.allclose(X.sum(0)[known], y[known], atol=1e-8))

    def test_custom_components(self) -> None:
        n = 120
        y = np.sin(np.linspace(0, 6, n))
        comps = [
            MeanSquareSmall(size=n),
            SmoothSecondDifference(weight=5.0),
            Sparse(weight=0.1),
        ]
        X = OSD(components=comps, max_iter=25).fit_transform(y)
        self.assertEqual(X.shape, (3, n))
        self.assertTrue(np.allclose(X.sum(0), y, atol=1e-8))

    def test_no_components_raises(self) -> None:
        with self.assertRaises(ValueError):
            OSD().fit_transform(np.ones(20))

    def test_unknown_preset_raises(self) -> None:
        with self.assertRaises(ValueError):
            OSD.preset("not_a_real_preset", 10)

    def test_short_signal_raises(self) -> None:
        osd = OSD.preset("l1_trend", 10, max_iter=5)
        with self.assertRaises(ValueError):
            osd.fit_transform(np.ones(2))

    def test_reconstruct(self) -> None:
        _, y, _, _ = _sine_square(n=100)
        osd = OSD.preset("l1_trend", y.size, max_iter=20)
        X = osd.fit_transform(y)
        self.assertTrue(np.allclose(osd.reconstruct(), X.sum(0)))

    def test_objective_recorded(self) -> None:
        _, y, _, _ = _sine_square(n=100)
        osd = OSD.preset("l1_trend", y.size, max_iter=15)
        osd.fit_transform(y)
        self.assertIsNotNone(osd.objective_value)
        self.assertTrue(len(osd.history["obj_vals"]) >= 1)

    def test_warm_start(self) -> None:
        _, y, _, _ = _sine_square(n=120)
        osd = OSD.preset("convex_demo", y.size, max_iter=15)
        X0 = osd.fit_transform(y)
        X1 = osd.fit_transform(y, X_init=X0, max_iter=5)
        self.assertEqual(X1.shape, X0.shape)
        self.assertTrue(np.allclose(X1.sum(0), y, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
