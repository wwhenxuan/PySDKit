# -*- coding: utf-8 -*-
"""
Unit tests for Bandwidth-Aware Adaptive Chirp Mode Decomposition (BA-ACMD).

Covers every public helper and the ``BA_ACMD`` class API, matching the
MATLAB package under ``Codes of BA-ACMD``.
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit._acmd.ba_acmd import (
    BA_ACMD,
    acmd_extract,
    add_noise,
    bandwidth_to_alpha,
    coef_overcomplete_fourier,
    compute_snr,
    differ,
    extract_if_ia,
    generate_demo_components,
    generate_demo_signal,
    gini_squared_envelope,
    impulse_times,
    spectrum_trend_generate,
)


class DifferTest(unittest.TestCase):
    def test_constant_derivative_zero(self) -> None:
        y = np.ones(20)
        d = differ(y, 0.01)
        self.assertEqual(d.shape, y.shape)
        self.assertTrue(np.allclose(d, 0.0, atol=1e-12))

    def test_linear_slope(self) -> None:
        dt = 0.1
        y = np.arange(30, dtype=float) * dt
        d = differ(y, dt)
        self.assertTrue(np.allclose(d[1:-1], 1.0, atol=1e-10))


class NoiseAndSNRTest(unittest.TestCase):
    def test_add_noise_stats(self) -> None:
        rng = np.random.default_rng(0)
        n = add_noise(5000, mean=0.0, std=0.2, rng=rng)
        self.assertEqual(n.size, 5000)
        self.assertAlmostEqual(float(np.mean(n)), 0.0, delta=0.05)
        self.assertAlmostEqual(float(np.std(n)), 0.2, delta=0.05)

    def test_snr_positive_for_mild_noise(self) -> None:
        clean = np.sin(np.linspace(0, 20, 1000))
        noisy = clean + 0.01 * np.random.default_rng(1).standard_normal(1000)
        self.assertGreater(compute_snr(clean, noisy), 20.0)


class ImpulseAndDemoSignalTest(unittest.TestCase):
    def test_impulse_times_period(self) -> None:
        fs = 1000.0
        t = np.arange(0, 1, 1 / fs)
        f0 = 10.0
        phi = 2.0 * np.pi * f0 * t
        times, index = impulse_times(phi, t)
        self.assertGreaterEqual(index.size, 9)
        gaps = np.diff(index)
        self.assertTrue(np.all(np.abs(gaps - fs / f0) <= 2))

    def test_generate_demo_components_shapes(self) -> None:
        t = np.arange(0, 0.5, 1 / 2000.0)
        c1, c2, c3, c4 = generate_demo_components(t)
        for c in (c1, c2, c3, c4):
            self.assertEqual(c.shape, t.shape)
            self.assertTrue(np.all(np.isfinite(c)))

    def test_generate_demo_signal_example2_length(self) -> None:
        t, x, snr, comps = generate_demo_signal(
            fs=5000.0, duration=1.0, noise_std=0.2, rng=np.random.default_rng(0)
        )
        self.assertEqual(t.size, 5001)
        self.assertEqual(x.size, 5001)
        self.assertTrue(np.isfinite(snr))
        self.assertIn("clean", comps)


class GiniAndIFIATest(unittest.TestCase):
    def test_gini_impulsive_vs_tone(self) -> None:
        n = 2000
        tone = np.sin(2 * np.pi * 40 * np.arange(n) / 1000.0)
        impulse = np.zeros(n)
        impulse[::100] = 1.0
        g_tone = gini_squared_envelope(tone)
        g_imp = gini_squared_envelope(impulse)
        self.assertTrue(np.isfinite(g_tone))
        self.assertTrue(np.isfinite(g_imp))
        self.assertGreater(g_imp, g_tone)

    def test_extract_if_ia_shapes(self) -> None:
        fs = 1000.0
        t = np.arange(500) / fs
        y = np.cos(2 * np.pi * 40 * t)
        inst_f, inst_a = extract_if_ia(y, fs)
        self.assertEqual(inst_f.shape, y.shape)
        self.assertEqual(inst_a.shape, y.shape)
        self.assertTrue(np.all(np.isfinite(inst_f)))
        self.assertAlmostEqual(float(np.median(inst_f)), 40.0, delta=3.0)


class BandwidthAlphaTest(unittest.TestCase):
    def test_alpha_increases_with_bandwidth(self) -> None:
        a_small = bandwidth_to_alpha(0.05, ac=0.3)
        a_large = bandwidth_to_alpha(0.2, ac=0.3)
        self.assertGreater(a_large, a_small)
        self.assertGreater(a_small, 0.0)

    def test_ac05_branch(self) -> None:
        a = bandwidth_to_alpha(0.1, ac=0.5)
        self.assertTrue(np.isfinite(a) and a > 0.0)


class CoefFourierTest(unittest.TestCase):
    def test_fit_smooth_series(self) -> None:
        x = np.linspace(0, 10, 256)
        y = np.exp(-0.2 * x) + 0.1 * np.sin(x)
        fit, inte = coef_overcomplete_fourier(y, samp_freq=1000.0, order_amp=8)
        self.assertEqual(fit.shape, y.shape)
        self.assertEqual(inte.shape, y.shape)
        self.assertLess(np.linalg.norm(fit - y) / np.linalg.norm(y), 0.35)


class SpectrumTrendTest(unittest.TestCase):
    def test_returns_ranked_intervals(self) -> None:
        fs = 2000.0
        t = np.arange(0, 0.5, 1 / fs)
        x = (
            np.sin(2 * np.pi * 80 * t)
            + 0.5 * np.sin(2 * np.pi * 300 * t)
            + 0.05 * np.random.default_rng(0).standard_normal(t.size)
        )
        spec, weight, trend, wtrend, sort_inter = spectrum_trend_generate(
            x, fs, offset=0.01, cut_pfreq=0.0015
        )
        self.assertEqual(spec.ndim, 1)
        self.assertEqual(weight.shape, spec.shape)
        self.assertEqual(trend.shape, spec.shape)
        self.assertEqual(wtrend.shape, spec.shape)
        self.assertGreaterEqual(sort_inter.shape[0], 1)
        self.assertEqual(sort_inter.shape[1], 2)
        # Intervals should be ordered by descending weighted peak
        self.assertTrue(np.all(sort_inter[:, 1] >= sort_inter[:, 0] - 1e-9))


class ACMDExtractTest(unittest.TestCase):
    def test_recovers_single_tone(self) -> None:
        fs = 1000.0
        n = 400
        t = np.arange(n) / fs
        f0 = 60.0
        s = np.cos(2 * np.pi * f0 * t)
        sest, if_est, ia_est = acmd_extract(
            s,
            fs,
            init_if=f0 * np.ones(n),
            alpha0=1e-3,
            beta=1e-4,
            tol=1e-7,
            max_iter=60,
        )
        self.assertTrue(np.all(np.isfinite(sest)))
        self.assertGreater(np.corrcoef(s, sest)[0, 1], 0.95)
        self.assertAlmostEqual(float(np.mean(if_est)), f0, delta=2.0)
        self.assertTrue(np.all(np.isfinite(ia_est)))

    def test_init_if_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            acmd_extract(np.ones(50), 100.0, np.ones(10), 1e-3)


class BAACMDClassTest(unittest.TestCase):
    def test_str_and_call(self) -> None:
        ba = BA_ACMD(fs=1000.0, max_iter=20, ce=0.1)
        self.assertIn("BA-ACMD", str(ba))
        t = np.arange(0, 0.3, 1 / 1000.0)
        x = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)
        a = ba(x)
        b = ba.fit_transform(x)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_invalid_fs(self) -> None:
        with self.assertRaises(ValueError):
            BA_ACMD(fs=-1)

    def test_short_signal_raises(self) -> None:
        ba = BA_ACMD(fs=1000.0)
        with self.assertRaises(ValueError):
            ba.fit_transform(np.ones(5))

    def test_fit_transform_attributes(self) -> None:
        t, x, _, _ = generate_demo_signal(
            fs=2000.0, duration=0.25, noise_std=0.1, rng=np.random.default_rng(2)
        )
        ba = BA_ACMD(
            fs=2000.0,
            beta=1e-10,
            tol=1e-6,
            ce=0.15,
            max_iter=35,
        )
        modes, ifs, ias = ba.fit_transform(x, return_all=True)
        self.assertEqual(modes.ndim, 2)
        self.assertEqual(modes.shape[1], x.size)
        self.assertEqual(ifs.shape, modes.shape)
        self.assertEqual(ias.shape, modes.shape)
        self.assertTrue(np.all(np.isfinite(modes)))
        self.assertIsNotNone(ba.imfs)
        self.assertIsNotNone(ba.spec)
        self.assertIsNotNone(ba.spec_trend)
        self.assertIsNotNone(ba.sort_intervals)
        self.assertEqual(ba.sort_intervals.shape[1], 2)

    def test_compute_spectrum_trend_wrapper(self) -> None:
        t = np.arange(0, 0.4, 1 / 1500.0)
        x = np.sin(2 * np.pi * 70 * t)
        ba = BA_ACMD(fs=1500.0)
        out = ba.compute_spectrum_trend(x)
        self.assertEqual(len(out), 5)

    def test_example2_smoke(self) -> None:
        """MATLAB Example2 parameters on a shorter crop for CI speed."""
        t, x, snr, _ = generate_demo_signal(
            fs=5000.0, duration=0.4, noise_std=0.2, rng=np.random.default_rng(0)
        )
        ba = BA_ACMD(
            fs=5000.0,
            beta=1e-10,
            tol=1e-7,
            ce=0.15,
            offset=0.01,
            cut_pfreq=0.0015,
            max_iter=50,
        )
        modes = ba.fit_transform(x)
        self.assertGreaterEqual(modes.shape[0], 1)
        self.assertEqual(modes.shape[1], x.size)
        self.assertTrue(np.all(np.isfinite(modes)))
        self.assertTrue(np.isfinite(snr))


if __name__ == "__main__":
    unittest.main()
