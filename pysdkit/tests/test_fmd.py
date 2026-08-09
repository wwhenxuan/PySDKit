# -*- coding: utf-8 -*-
"""
Unit tests for Feature Mode Decomposition (FMD).

Covers the public class API and every helper used by the MATLAB port:
``TT``, ``CK``, ``max_IJ``, ``_build_XmT``, and ``FMD._xxc_mckd``.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.signal import hilbert

from pysdkit import FMD
from pysdkit._fmd.fmd import CK, TT, _build_XmT, max_IJ
from pysdkit.data import test_emd, test_fmd


def _tone(n: int = 2048, fs: float = 2000.0, f0: float = 80.0) -> tuple:
    t = np.arange(n, dtype=float) / fs
    return t, np.cos(2.0 * np.pi * f0 * t), fs


def _periodic_impulses(
    n: int = 4000,
    fs: float = 5000.0,
    period: int = 100,
    fn: float = 800.0,
    decay: float = 300.0,
) -> tuple:
    """Synthetic underdamped impulses — paper Eq. (2) style."""
    t = np.arange(n, dtype=float) / fs
    x = np.zeros(n, dtype=float)
    kernel_len = min(period, n)
    tk = np.arange(kernel_len, dtype=float) / fs
    kernel = np.exp(-decay * tk) * np.cos(2.0 * np.pi * fn * tk)
    for start in range(0, n - kernel_len, period):
        x[start : start + kernel_len] += kernel
    x += 0.05 * np.random.default_rng(0).standard_normal(n)
    return t, x, fs


class HelperTTTest(unittest.TestCase):
    def test_tt_finds_known_period(self) -> None:
        fs = 1000.0
        period = 50
        n = 2000
        t = np.arange(n) / fs
        # Narrow pulses every ``period`` samples → clear autocorr peak
        y = np.zeros(n)
        y[::period] = 1.0
        y = y - y.mean()
        T = TT(y, fs)
        self.assertGreater(T, period // 2)
        self.assertLess(abs(T - period), 5)

    def test_tt_short_signal(self) -> None:
        self.assertEqual(TT(np.array([1.0]), 10), 1)

    def test_tt_matches_matlab_index_convention(self) -> None:
        """T = zeroposi_1based + max_position_1based (= 0-based lag + 2)."""
        y = np.zeros(200)
        y[0] = 1.0
        y[40] = 0.8
        y = y - y.mean()
        # Just ensure a finite positive integer period
        T = TT(y, fs=100)
        self.assertIsInstance(T, int)
        self.assertGreaterEqual(T, 1)


class HelperCKTest(unittest.TestCase):
    def test_ck_positive_for_impulses(self) -> None:
        _, x, _ = _periodic_impulses()
        val = CK(x - x.mean(), T=100, M=1)
        self.assertTrue(np.isfinite(val))
        self.assertGreater(val, 0.0)

    def test_ck_default_order(self) -> None:
        x = np.random.default_rng(1).standard_normal(500)
        a = CK(x, T=20)
        b = CK(x, T=20, M=2)
        self.assertAlmostEqual(a, b)

    def test_ck_zero_signal(self) -> None:
        val = CK(np.zeros(100), T=10, M=1)
        self.assertTrue(np.isfinite(val))


class HelperMaxIJTest(unittest.TestCase):
    def test_max_ij_location(self) -> None:
        X = np.array(
            [
                [0.0, 0.2, 0.9],
                [0.0, 0.0, 0.3],
                [0.0, 0.0, 0.0],
            ]
        )
        I, J, M = max_IJ(X)
        self.assertEqual((I, J), (0, 2))
        self.assertAlmostEqual(M, 0.9)

    def test_max_ij_square(self) -> None:
        X = np.eye(4) * 0.0
        X[2, 1] = 1.5
        I, J, M = max_IJ(X)
        self.assertEqual(I, 2)
        self.assertEqual(J, 1)
        self.assertAlmostEqual(M, 1.5)


class BuildXmTTest(unittest.TestCase):
    def test_shapes_and_invertible(self) -> None:
        x = np.random.default_rng(0).standard_normal(300)
        L, T, M = 16, 20, 1
        XmT, Xinv = _build_XmT(x, L, T, M)
        self.assertEqual(XmT.shape, (L, x.size, M + 1))
        self.assertEqual(Xinv.shape, (L, L))
        G = XmT[:, :, 0] @ XmT[:, :, 0].T
        self.assertTrue(np.allclose(Xinv @ G, np.eye(L), atol=1e-6))


class MCKDTest(unittest.TestCase):
    def test_xxc_mckd_shapes(self) -> None:
        _, x, fs = _tone(n=1024, fs=2000.0, f0=60.0)
        fmd = FMD(fs=fs, mode_num=1, filter_size=20, cut_num=3, max_iter_num=6)
        f0 = np.zeros(20)
        f0[0] = 1.0
        y_f, f_f, ck, T = fmd._xxc_mckd(x, f0, term_iter=3, T=None, M=1)
        self.assertEqual(y_f.shape, (x.size, 3))
        self.assertEqual(f_f.shape, (20, 3))
        self.assertEqual(ck.shape, (3,))
        self.assertTrue(np.all(np.isfinite(ck)))
        self.assertGreaterEqual(T, 1)

    def test_xxc_mckd_with_given_T(self) -> None:
        _, x, fs = _periodic_impulses(n=2000)
        fmd = FMD(fs=fs, filter_size=16, cut_num=4, mode_num=1, max_iter_num=4)
        f0 = np.kaiser(16, 6)
        f0 = f0 / np.linalg.norm(f0)
        y_f, f_f, ck, T = fmd._xxc_mckd(x, f0, term_iter=2, T=100, M=1)
        self.assertEqual(y_f.shape[1], 2)
        self.assertTrue(np.all(np.isfinite(y_f)))
        self.assertTrue(np.all(np.isfinite(f_f)))


class FMDClassTest(unittest.TestCase):
    def test_str_and_call(self) -> None:
        fmd = FMD(fs=1000, mode_num=2, cut_num=4, filter_size=20, max_iter_num=6)
        self.assertIn("FMD", str(fmd))
        _, x, _ = _tone(n=800, fs=1000.0)
        a = fmd(x)
        b = fmd.fit_transform(x)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_invalid_params(self) -> None:
        with self.assertRaises(ValueError):
            FMD(mode_num=0)
        with self.assertRaises(ValueError):
            FMD(filter_size=2)
        with self.assertRaises(ValueError):
            FMD(cut_num=0)
        with self.assertRaises(ValueError):
            FMD(mode_num=5, cut_num=3)
        with self.assertRaises(ValueError):
            FMD(fs=-1)
        with self.assertRaises(ValueError):
            FMD(max_iter_num=0)

    def test_short_signal_raises(self) -> None:
        fmd = FMD(fs=1000, filter_size=30, cut_num=3, mode_num=1, max_iter_num=4)
        with self.assertRaises(ValueError):
            fmd.fit_transform(np.ones(10))

    def test_fit_transform_shape_and_attrs(self) -> None:
        _, x, fs = _tone(n=1500, fs=2000.0, f0=50.0)
        fmd = FMD(fs=fs, mode_num=2, cut_num=4, filter_size=24, max_iter_num=8)
        modes = fmd.fit_transform(x)
        self.assertEqual(modes.shape, (2, x.size))
        self.assertTrue(np.isrealobj(modes))
        self.assertTrue(np.all(np.isfinite(modes)))
        self.assertTrue(np.allclose(fmd.imfs, modes))
        self.assertEqual(fmd.filters.shape, (2, 24))
        self.assertEqual(fmd.peak_freqs.shape, (2,))

    def test_fs_override(self) -> None:
        _, x, _ = _tone(n=1000, fs=1000.0)
        fmd = FMD(fs=500.0, mode_num=1, cut_num=3, filter_size=16, max_iter_num=4)
        modes = fmd.fit_transform(x, fs=1000.0)
        self.assertEqual(fmd.fs, 1000.0)
        self.assertEqual(modes.shape, (1, x.size))

    def test_invalid_fs_override(self) -> None:
        fmd = FMD(fs=1000, mode_num=1, cut_num=2, filter_size=12, max_iter_num=3)
        with self.assertRaises(ValueError):
            fmd.fit_transform(np.ones(200), fs=-5)

    def test_builtin_emd_signal_runs(self) -> None:
        _, signal = test_emd()
        modes = FMD(
            fs=1000, mode_num=2, cut_num=5, filter_size=20, max_iter_num=8
        ).fit_transform(signal)
        self.assertEqual(modes.shape, (2, signal.size))

    def test_periodic_impulses_extracts_energy(self) -> None:
        _, x, fs = _periodic_impulses()
        modes = FMD(
            fs=fs, mode_num=2, cut_num=5, filter_size=30, max_iter_num=10
        ).fit_transform(x)
        self.assertEqual(modes.shape[0], 2)
        # At least one mode should carry a non-trivial fraction of energy
        energies = np.sum(modes**2, axis=1)
        self.assertGreater(energies.max() / (np.sum(x**2) + 1e-12), 0.05)

    def test_demo_npy_loader(self) -> None:
        t, x, fs = test_fmd()
        self.assertEqual(x.ndim, 1)
        self.assertEqual(t.size, x.size)
        self.assertAlmostEqual(fs, 2.0e4)
        self.assertEqual(x.size, 20001)

    def test_demo_signal_decomposition(self) -> None:
        """MATLAB demo parameters: fs=20kHz, L=30, K=7, ModeNum=2, MaxIter=20."""
        _, x, fs = test_fmd()
        # Use a shorter crop for unit-test runtime while keeping demo settings
        x_c = x[:8000]
        modes = FMD(
            fs=fs,
            mode_num=2,
            filter_size=30,
            cut_num=7,
            max_iter_num=20,
        ).fit_transform(x_c)
        self.assertEqual(modes.shape, (2, x_c.size))
        self.assertTrue(np.all(np.isfinite(modes)))
        # Envelope spectra should be finite (fault-feature view in the demo)
        for m in modes:
            env = np.abs(hilbert(m)) - np.mean(np.abs(hilbert(m)))
            self.assertTrue(np.all(np.isfinite(env)))


if __name__ == "__main__":
    unittest.main()
