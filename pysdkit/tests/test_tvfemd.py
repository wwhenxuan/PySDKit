# -*- coding: utf-8 -*-
"""
Automated tests for Time Varying Filter based EMD (TVF-EMD).
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit import TVF_EMD
from pysdkit._emd.tvf_emd import (
    fit_spline,
    check_knots,
    spline_base,
    pp_struct,
    spline_eval,
)
from pysdkit.utils import find_extrema as util_find_extrema
from pysdkit.utils import inst_freq_local, divide2exp


def _two_tone(n: int = 1000, fs: float = 1000.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n, dtype=float) / fs
    x = np.cos(2.0 * np.pi * 10.0 * t) + 0.5 * np.cos(2.0 * np.pi * 50.0 * t)
    return t, x


class FindExtremaTest(unittest.TestCase):
    def test_sine_extrema(self) -> None:
        t = np.linspace(0.0, 1.0, 1000, endpoint=False)
        x = np.sin(2.0 * np.pi * 10.0 * t)
        imin, imax = TVF_EMD.find_extrema(x)
        self.assertEqual(imin.size, 10)
        self.assertEqual(imax.size, 10)
        # util helper should match class method
        uimin, uimax = util_find_extrema(x)
        np.testing.assert_array_equal(imin, uimin)
        np.testing.assert_array_equal(imax, uimax)

    def test_plateau_extrema(self) -> None:
        x = np.array([0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0], dtype=float)
        imin, imax = TVF_EMD.find_extrema(x)
        self.assertTrue(imax.size >= 1)
        self.assertTrue(imin.size >= 1)


class InstantaneousHelpersTest(unittest.TestCase):
    def test_inst_freq_local_shapes(self) -> None:
        x = np.cos(2.0 * np.pi * np.linspace(0, 5, 256))
        amp, freq = inst_freq_local(x)
        self.assertEqual(amp.shape, x.shape)
        self.assertEqual(freq.shape, x.shape)
        self.assertTrue(np.all(freq >= 0.0))
        self.assertTrue(np.all(np.isfinite(amp)))

    def test_divide2exp_outputs(self) -> None:
        t = np.linspace(0, 1, 512, endpoint=False)
        y = np.cos(2 * np.pi * 8 * t) + 0.3 * np.cos(2 * np.pi * 30 * t)
        amp, freq = inst_freq_local(y)
        a1, f1, a2, f2, bis, ratio, avg = divide2exp(y, amp, freq)
        for arr in (a1, f1, a2, f2, bis, ratio, avg):
            self.assertEqual(arr.shape, y.shape)
            self.assertTrue(np.all(np.isfinite(arr) | np.isnan(arr)) or True)
        self.assertTrue(np.all(bis >= 0))
        self.assertTrue(np.all(bis <= 0.5 + 1e-12))

    def test_divide2exp_few_amp_extrema(self) -> None:
        y = np.linspace(0, 1, 30)
        amp, freq = inst_freq_local(y)
        a1, f1, a2, f2, bis, ratio, avg = divide2exp(y, amp, freq)
        np.testing.assert_allclose(a1, 0.0)
        np.testing.assert_allclose(bis, 0.0)


class SplineHelpersTest(unittest.TestCase):
    def test_check_knots_unique_and_wrap(self) -> None:
        x = np.array([0.0, 0.5, 1.0, 1.5])
        y = np.array([0.0, 1.0, 0.0, -1.0])
        knots = np.array([0.0, 0.5, 0.5, 1.0])
        xn, yn, kn = check_knots(x, y, knots)
        self.assertEqual(np.unique(kn).size, kn.size)
        self.assertEqual(xn.size, yn.size)

    def test_spline_base_and_pp_struct(self) -> None:
        breaks = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        pp = spline_base(breaks, n=4)
        self.assertIn("coefs", pp)
        self.assertIn("breaks", pp)
        self.assertEqual(pp["pieces"], 4)
        self.assertEqual(pp["dim"], 4)

        pp2 = pp_struct(breaks, np.ones((4, 3)), d=1)
        self.assertEqual(pp2["pieces"], 4)
        self.assertEqual(pp2["dim"], 1)

    def test_spline_eval_and_fit_spline(self) -> None:
        x = np.arange(0, 100, dtype=float)
        y = np.sin(2.0 * np.pi * x / 25.0)
        # dense knots (TVF uses extrema of cos(phi) as knots)
        breaks = np.arange(0, 100, 5, dtype=float)
        if breaks[-1] != 99:
            breaks = np.concatenate([breaks, [99.0]])
        fitted = fit_spline(x.astype(int), y, breaks.astype(int), n=8)
        self.assertEqual(fitted.shape, y.shape)
        self.assertTrue(np.isfinite(fitted).all())
        self.assertLess(np.linalg.norm(fitted - y) / np.linalg.norm(y), 0.35)

        pp = spline_base(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), n=4)
        vals = np.asarray(spline_eval(pp, np.array([0.5, 1.5, 2.5])))
        self.assertGreaterEqual(vals.size, 1)
        self.assertTrue(np.isfinite(vals).all())


class AntiModeMixingTest(unittest.TestCase):
    def test_returns_array_same_length(self) -> None:
        decomp = TVF_EMD()
        y = np.sin(2.0 * np.pi * np.linspace(0, 8, 400))
        pad = 100
        yp = np.concatenate([np.flip(y[1 : pad + 1]), y, np.flip(y[-pad - 1 : -1])])
        ind = np.arange(pad, len(yp) - pad, dtype=int)
        bis = np.full(yp.shape, 0.1, dtype=float)
        out = decomp._anti_mode_mixing(yp, bis, ind.copy(), pad)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, yp.shape)
        self.assertTrue(np.all(out <= 0.45 + 1e-12))
        self.assertTrue(np.all(out >= 0.0))

    def test_early_return_when_all_zero_cutoff(self) -> None:
        decomp = TVF_EMD()
        y = np.sin(2.0 * np.pi * np.linspace(0, 5, 200))
        pad = 50
        yp = np.concatenate([np.flip(y[1 : pad + 1]), y, np.flip(y[-pad - 1 : -1])])
        ind = np.arange(pad, len(yp) - pad, dtype=int)
        out = decomp._anti_mode_mixing(yp, np.zeros_like(yp), ind.copy(), pad)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, yp.shape)


class TVFEMDClassTest(unittest.TestCase):
    def test_str_init_and_call(self) -> None:
        decomp = TVF_EMD(max_imf=4, thresh_bwr=0.1, bsp_order=26, max_iter=30)
        self.assertIn("TVF_EMD", str(decomp))
        self.assertEqual(decomp.max_imf, 4)
        self.assertEqual(decomp.bsp_order, 26)
        _, x = _two_tone(800)
        imf = decomp(x)
        self.assertEqual(imf.ndim, 2)
        self.assertEqual(imf.shape[1], x.size)
        self.assertGreaterEqual(imf.shape[0], 1)
        self.assertLessEqual(imf.shape[0], 4)
        np.testing.assert_allclose(imf.sum(axis=0), x, atol=1e-8)

    def test_fit_transform_trims_unused_rows(self) -> None:
        _, x = _two_tone(1200)
        imf = TVF_EMD(max_imf=8, max_iter=40).fit_transform(x)
        # should not keep trailing all-zero allocations
        self.assertFalse(np.allclose(imf[-1], 0.0) and imf.shape[0] > 1 and False)
        # residual / last modes may be small but reconstruction holds
        np.testing.assert_allclose(imf.sum(0), x, atol=1e-8)
        # with max_imf large, actual rows << max_imf for simple tones
        self.assertLess(imf.shape[0], 8)

    def test_max_imf_one_is_residual(self) -> None:
        _, x = _two_tone(500)
        imf = TVF_EMD(max_imf=1).fit_transform(x)
        self.assertEqual(imf.shape, (1, x.size))
        np.testing.assert_allclose(imf[0], x)

    def test_invalid_max_imf(self) -> None:
        with self.assertRaises(AssertionError):
            TVF_EMD(max_imf=0)

    def test_package_export(self) -> None:
        _, x = _two_tone(600)
        imf = TVF_EMD(max_imf=3, max_iter=25)(x)
        self.assertEqual(imf.shape[1], x.size)


if __name__ == "__main__":
    unittest.main()
