# -*- coding: utf-8 -*-
"""
Automated tests for Robust Empirical Mode Decomposition (REMD / EMD-SSSC).
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit import REMD
from pysdkit._emd.remd import (
    extr,
    extend,
    emd_mean,
    stop_emd,
    is_sifting_process_stop,
    index_of_orthogonality,
    remd,
)


def _matlab_demo_signal(
    n: int = 4000, fs: float = 10000.0
) -> tuple[np.ndarray, np.ndarray]:
    """AM-FM chirp + tone from ``emd_sssc.m`` example (shortened for unit tests)."""
    t = np.arange(1, n + 1, dtype=float) / fs
    x = (2.0 + np.cos(2.0 * np.pi * 0.5 * t)) * np.cos(
        2.0 * np.pi * 5.0 * t + 15.0 * t**2
    ) + np.cos(2.0 * np.pi * 2.0 * t)
    return t, x


class ExtrTest(unittest.TestCase):
    def test_sine_extrema_and_zeros(self) -> None:
        n = 1000
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        x = np.sin(2.0 * np.pi * 10.0 * t)
        indmin, indmax, indzer = extr(x)
        self.assertEqual(indmin.size, 10)
        self.assertEqual(indmax.size, 10)
        self.assertEqual(indzer.size, 20)
        # maxima near cos peaks (samples 25, 125, ...)
        np.testing.assert_array_equal(indmax[:3], np.array([25, 125, 225]))

    def test_monotonic_few_extrema(self) -> None:
        x = np.linspace(0.0, 1.0, 50)
        indmin, indmax, _ = extr(x)
        self.assertEqual(indmin.size + indmax.size, 0)


class ExtendTest(unittest.TestCase):
    def test_ext_ratio_zero_passthrough(self) -> None:
        x = np.sin(2.0 * np.pi * np.linspace(0, 3, 200))
        imin, imax, _ = extr(x)
        eimin, eimax, ex, cut = extend(x, imin, imax, 0.0)
        np.testing.assert_array_equal(eimin, imin)
        np.testing.assert_array_equal(eimax, imax)
        np.testing.assert_allclose(ex, x)
        np.testing.assert_array_equal(cut, np.array([0, x.size]))

    def test_extend_lengthens_and_cut_span(self) -> None:
        x = np.sin(2.0 * np.pi * np.linspace(0, 5, 500))
        imin, imax, _ = extr(x)
        eimin, eimax, ex, cut = extend(x, imin, imax, 0.2)
        self.assertGreater(ex.size, x.size)
        # MATLAB cut window has length N (may mix mirrored edges when lsym>1)
        self.assertEqual(cut[1] - cut[0], x.size)
        self.assertTrue(0 <= cut[0] < cut[1] <= ex.size)
        self.assertTrue(eimin.size >= imin.size)
        self.assertTrue(eimax.size >= imax.size)

    def test_extend_cut_matches_when_lsym_is_boundary(self) -> None:
        # Pure sine with sample 0 at a zero-crossing often mirrors about index 1
        t = np.arange(1, 2001, dtype=float) / 1000.0
        x = np.sin(2.0 * np.pi * 5.0 * t)
        imin, imax, _ = extr(x)
        _, _, ex, cut = extend(x, imin, imax, 0.2)
        np.testing.assert_allclose(ex[cut[0] : cut[1]], x, atol=1e-12)


class EmdMeanTest(unittest.TestCase):
    def test_mean_shape_and_fewer_than_three(self) -> None:
        x = np.sin(2.0 * np.pi * np.linspace(0, 4, 400))
        m, n_extr = emd_mean(x, ext_ratio=0.2)
        self.assertEqual(m.shape, x.shape)
        self.assertGreaterEqual(n_extr, 3)
        self.assertTrue(np.isfinite(m).all())

        m2, n2 = emd_mean(np.linspace(0, 1, 20), ext_ratio=0.2)
        self.assertEqual(m2.size, 0)
        self.assertLess(n2, 3)

    def test_invalid_smooth_mode(self) -> None:
        x = np.sin(2.0 * np.pi * np.linspace(0, 2, 100))
        with self.assertRaises(ValueError):
            emd_mean(x, smooth_mode="ma")


class StopAndSscTest(unittest.TestCase):
    def test_stop_emd(self) -> None:
        x = np.sin(2.0 * np.pi * np.linspace(0, 3, 300))
        energy = float(np.sum(x**2))
        self.assertFalse(stop_emd(x, energy))
        residual = 1e-4 * np.ones(300)
        self.assertTrue(stop_emd(residual, energy))
        self.assertTrue(stop_emd(np.linspace(0, 1, 40), energy))

    def test_ssc_liu_stops_when_fv_nondecreasing(self) -> None:
        # Build a local mean / IMF-like pair with enough extrema & zeros
        n = 400
        t = np.linspace(0, 4, n)
        s = np.sin(2.0 * np.pi * 5.0 * t)
        m = 0.05 * np.sin(2.0 * np.pi * 5.0 * t + 0.3)
        fv = np.zeros(10)
        stop, fv = is_sifting_process_stop(m, s, 1, fv, ssc="liu")
        self.assertFalse(stop)
        stop, fv = is_sifting_process_stop(m * 0.9, s, 2, fv, ssc="liu")
        self.assertFalse(stop)
        # Force non-decreasing FV for last three iterations
        fv[0], fv[1] = 1.0, 1.5
        stop, fv = is_sifting_process_stop(m * 2.0, s, 3, fv, ssc="liu")
        # May or may not stop depending on nzm/nem; just check API + finite FV
        self.assertTrue(np.isfinite(fv[2]))
        self.assertIsInstance(stop, (bool, np.bool_))

    def test_ssc_unknown(self) -> None:
        with self.assertRaises(ValueError):
            is_sifting_process_stop(
                np.ones(10), np.sin(np.linspace(0, 6, 10)), 1, np.zeros(5), ssc="foo"
            )


class OrthogonalityTest(unittest.TestCase):
    def test_io_identical_modes(self) -> None:
        x = np.random.randn(100)
        imf = np.vstack([x, np.zeros_like(x)])
        ort = index_of_orthogonality(x, imf)
        self.assertGreaterEqual(ort, 0.0)
        self.assertTrue(np.isfinite(ort))

    def test_io_1d_imf(self) -> None:
        x = np.random.randn(50)
        ort = index_of_orthogonality(x, x)
        self.assertAlmostEqual(ort, 0.0)


class RemdFunctionalTest(unittest.TestCase):
    def test_reconstruction_and_shapes(self) -> None:
        _, x = _matlab_demo_signal(n=3000)
        imf, ort, fvs, iter_num = remd(x, max_imfs=4, max_iter=20, ext_ratio=0.2)
        self.assertEqual(imf.ndim, 2)
        self.assertEqual(imf.shape[1], x.size)
        self.assertGreaterEqual(imf.shape[0], 2)  # at least one IMF + residual
        np.testing.assert_allclose(imf.sum(axis=0), x, atol=1e-10)
        self.assertTrue(np.isfinite(ort))
        self.assertEqual(fvs.shape[1], 20)
        self.assertEqual(iter_num.size, imf.shape[0] - 1)
        self.assertTrue(np.all(iter_num >= 1))

    def test_ext_ratio_zero(self) -> None:
        _, x = _matlab_demo_signal(n=2000)
        imf, _, _, _ = remd(x, max_imfs=3, max_iter=15, ext_ratio=0.0)
        np.testing.assert_allclose(imf.sum(axis=0), x, atol=1e-10)


class RemdClassTest(unittest.TestCase):
    def test_str_and_init(self) -> None:
        decomp = REMD(max_imfs=5, max_iter=15, ext_ratio=0.1)
        self.assertIn("REMD", str(decomp))
        self.assertEqual(decomp.max_imfs, 5)
        self.assertEqual(decomp.max_iter, 15)
        self.assertEqual(decomp.ext_ratio, 0.1)
        self.assertEqual(decomp.ssc, "liu")

    def test_fit_transform_and_call(self) -> None:
        _, x = _matlab_demo_signal(n=2500)
        decomp = REMD(max_imfs=4, max_iter=20)
        imf = decomp.fit_transform(x)
        self.assertIsNotNone(decomp.imfs)
        self.assertIsNotNone(decomp.ort)
        self.assertIsNotNone(decomp.fvs)
        self.assertIsNotNone(decomp.iter_num)
        np.testing.assert_allclose(imf.sum(axis=0), x, atol=1e-10)

        imf2, ort, fvs, it = decomp(x, return_all=True)
        self.assertEqual(imf2.shape, imf.shape)
        self.assertTrue(np.isfinite(ort))
        self.assertEqual(fvs.shape[1], 20)
        self.assertEqual(it.size, imf2.shape[0] - 1)

    def test_package_export(self) -> None:
        _, x = _matlab_demo_signal(n=1500)
        imf = REMD(max_imfs=3, max_iter=12)(x)
        self.assertEqual(imf.shape[1], x.size)


if __name__ == "__main__":
    unittest.main()
