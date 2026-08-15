# -*- coding: utf-8 -*-
"""
Unit tests for Data-driven Adaptive Chirp Mode Decomposition (DD-ACMD).

Covers every public helper and the ``DD_ACMD`` class API, matching the
MATLAB package under ``Code of DDACMD``.
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit import DD_ACMD
from pysdkit._acmd.dd_acmd import (
    DDACMD,
    add_noise,
    data_driven_if_init,
    find_extrema,
    generate_close_modes_demo,
    generate_nonstationary_demo,
    generate_stationary_demo,
    if_derivative_normalization,
    low_filter,
    norm_safe,
    phase_arccos,
    time_varying_lowpass,
)


class FindExtremaTest(unittest.TestCase):
    """Tests for local-maxima extraction ``find_extrema`` (MATLAB ``findev``)."""

    def test_sine_maxima_count(self) -> None:
        """A multi-period sine should yield several sorted interior maxima."""
        t = np.linspace(0, 1, 201, endpoint=False)
        y = np.sin(2 * np.pi * 5 * t)
        vals, idx, up = find_extrema(y)
        self.assertEqual(vals.shape, idx.shape)
        self.assertEqual(up.shape, y.shape)
        self.assertGreaterEqual(idx.size, 4)
        self.assertTrue(np.all(np.diff(idx) > 0))

    def test_endpoint_maximum(self) -> None:
        """Endpoint samples that dominate neighbors should be recorded as maxima."""
        y = np.array([3.0, 1.0, 0.0, 1.0, 2.5])
        _, idx, _ = find_extrema(y)
        self.assertEqual(int(idx[0]), 0)
        self.assertEqual(int(idx[-1]), 4)


class PhaseArccosTest(unittest.TestCase):
    """Tests for piecewise phase recovery via custom ``phase_arccos``."""

    def test_constant_zero(self) -> None:
        """A zero normalized waveform should map to a finite constant phase."""
        g = np.zeros(20)
        th = phase_arccos(g)
        self.assertEqual(th.shape, g.shape)
        self.assertTrue(np.all(np.isfinite(th)))

    def test_clipped_domain(self) -> None:
        """Values outside ``[-1, 1]`` should be clipped without NaN/Inf phases."""
        g = np.linspace(-1.2, 1.2, 50)
        th = phase_arccos(g)
        self.assertTrue(np.all(np.isfinite(th)))


class LowFilterAndTVLPTest(unittest.TestCase):
    """Tests for FIR low-pass filtering and time-varying low-pass (TVLP)."""

    def test_low_filter_preserves_dc(self) -> None:
        """A constant (DC) input should pass the FIR low-pass nearly unchanged."""
        fs = 200.0
        x = np.ones(400)
        y = low_filter(x, cut_freq=20.0, samp_freq=fs)
        self.assertEqual(y.shape, x.shape)
        self.assertAlmostEqual(float(np.mean(y)), 1.0, delta=0.05)

    def test_low_filter_attenuates_high(self) -> None:
        """High-frequency content above the cutoff should be strongly reduced."""
        fs = 500.0
        t = np.arange(1000) / fs
        x = np.cos(2 * np.pi * 5 * t) + np.cos(2 * np.pi * 120 * t)
        y = low_filter(x, cut_freq=30.0, samp_freq=fs)
        # residual of high-tone should shrink
        self.assertLess(float(np.std(y - np.cos(2 * np.pi * 5 * t))), 0.6)

    def test_tvlp_shape(self) -> None:
        """TVLP output must keep the input length and remain finite."""
        fs = 200.0
        t = np.arange(300) / fs
        x = np.cos(2 * np.pi * 40 * t) + 0.3 * np.cos(2 * np.pi * 10 * t)
        e_if = 40.0 * np.ones_like(t)
        out = time_varying_lowpass(x, fs, e_if, c_pass=40.0)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(out)))


class IFDNAndDDIFITest(unittest.TestCase):
    """Tests for derivative-normalization IF and data-driven IF initialization."""

    def test_if_dn_near_constant_tone(self) -> None:
        """IF-DN on a pure tone should return a finite, non-trivial IF estimate."""
        fs = 500.0
        t = np.arange(0, 1.0, 1 / fs)
        f0 = 40.0
        x = np.cos(2 * np.pi * f0 * t)
        inst = if_derivative_normalization(x, fs, beta=1e-9)
        self.assertEqual(inst.shape, x.shape)
        # median IF should be in the ballpark of f0 (DN is approximate)
        self.assertTrue(np.isfinite(np.median(inst)))
        self.assertGreater(float(np.median(np.abs(inst))), 5.0)

    def test_ddifi_returns_finite(self) -> None:
        """DDIFI should return a finite IF trajectory for a two-tone residual."""
        fs = 300.0
        t = np.arange(0, 0.8, 1 / fs)
        x = np.cos(2 * np.pi * 25 * t) + 0.5 * np.cos(2 * np.pi * 40 * t)
        ini = data_driven_if_init(x, fs, beta=1e-10, max_iter=3, tol=0.05)
        self.assertEqual(ini.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(ini)))

    def test_norm_safe_nonzero(self) -> None:
        """``norm_safe`` on a zero vector should return a positive epsilon floor."""
        self.assertGreater(norm_safe(np.zeros(5)), 0.0)


class NoiseAndDemoTest(unittest.TestCase):
    """Tests for noise helper and MATLAB-style demo signal generators."""

    def test_add_noise_stats(self) -> None:
        """``add_noise`` should approximately match the requested mean and std."""
        n = add_noise(4000, mean=0.0, std=0.25, rng=np.random.default_rng(0))
        self.assertAlmostEqual(float(np.mean(n)), 0.0, delta=0.05)
        self.assertAlmostEqual(float(np.std(n)), 0.25, delta=0.05)

    def test_stationary_demo_shapes(self) -> None:
        """Stationary demo must provide four modes aligned with the time axis."""
        demo = generate_stationary_demo(fs=200.0, duration=0.5, noise_std=0.1, rng=np.random.default_rng(1))
        self.assertEqual(demo["modes"].shape[0], 4)
        self.assertEqual(demo["signal"].size, demo["t"].size)

    def test_nonstationary_and_close_demos(self) -> None:
        """Non-stationary and close-mode demos should each expose three chirps."""
        a = generate_nonstationary_demo(fs=400.0, duration=0.5, noise_std=0.2, rng=np.random.default_rng(2))
        b = generate_close_modes_demo(fs=400.0, duration=0.5, noise_std=0.0)
        self.assertEqual(a["modes"].shape[0], 3)
        self.assertEqual(b["ifs"].shape[0], 3)


class DDACMDClassTest(unittest.TestCase):
    """Tests for the ``DD_ACMD`` / ``DDACMD`` class public API."""

    def test_str_and_alias(self) -> None:
        """String form should mention DD-ACMD; ``DDACMD`` aliases ``DD_ACMD``."""
        alg = DD_ACMD(fs=200.0, max_iter=20)
        self.assertIn("DD-ACMD", str(alg))
        self.assertIs(DDACMD, DD_ACMD)

    def test_init_rejects_nonpositive_fs(self) -> None:
        """Construction with non-positive sampling rate must raise ``ValueError``."""
        with self.assertRaises(ValueError):
            DD_ACMD(fs=0.0)

    def test_extract_trend_shapes(self) -> None:
        """Trend extraction uses a zero IF seed and returns finite arrays."""
        fs = 200.0
        t = np.arange(0, 0.6, 1 / fs)
        x = 2 * t**2 + 0.3 * np.cos(2 * np.pi * 20 * t)
        alg = DD_ACMD(fs=fs, max_iter=40, tol=1e-8)
        sest, if_est, ia_est, init_if = alg.extract_trend(x)
        self.assertEqual(sest.shape, x.shape)
        self.assertTrue(np.allclose(init_if, 0.0))
        self.assertTrue(np.all(np.isfinite(if_est)))

    def test_estimate_init_if(self) -> None:
        """``estimate_init_if`` should wrap DDIFI and preserve signal length."""
        fs = 250.0
        t = np.arange(0, 0.5, 1 / fs)
        x = np.cos(2 * np.pi * 30 * t)
        alg = DD_ACMD(fs=fs, ddifi_max_iter=2)
        ini = alg.estimate_init_if(x)
        self.assertEqual(ini.shape, x.shape)

    def test_fit_transform_stationary_short(self) -> None:
        """Short stationary mixture should yield a trend-like first mode."""
        demo = generate_stationary_demo(
            fs=200.0, duration=0.6, noise_std=0.15, rng=np.random.default_rng(0)
        )
        alg = DD_ACMD(
            fs=demo["fs"],
            k_max=6,
            max_iter=50,
            tol=1e-8,
            energy_tol=0.02,
            ddifi_max_iter=3,
        )
        imfs = alg.fit_transform(demo["signal"])
        self.assertGreaterEqual(imfs.shape[0], 2)
        self.assertEqual(imfs.shape[1], demo["signal"].size)
        self.assertIsNotNone(alg.imfs_)
        # trend-ish first mode should correlate with quadratic component
        trend = demo["modes"][0]
        corr = abs(np.corrcoef(imfs[0], trend)[0, 1])
        self.assertGreater(corr, 0.5)

    def test_return_all_and_call(self) -> None:
        """``__call__(..., return_all=True)`` returns IMFs, init IF, eIF, and eIA."""
        demo = generate_stationary_demo(
            fs=200.0, duration=0.5, noise_std=0.1, rng=np.random.default_rng(3)
        )
        alg = DD_ACMD(fs=demo["fs"], k_max=5, max_iter=40, tol=1e-8, ddifi_max_iter=2)
        imfs, ini, eif, eia = alg(demo["signal"], return_all=True)
        self.assertEqual(imfs.shape, ini.shape)
        self.assertEqual(eif.shape, eia.shape)
        self.assertEqual(imfs.shape[1], demo["signal"].size)

    def test_short_signal_raises(self) -> None:
        """Signals shorter than the minimum length must raise ``ValueError``."""
        alg = DD_ACMD(fs=100.0, max_iter=10)
        with self.assertRaises(ValueError):
            alg.fit_transform(np.ones(5))

    def test_close_modes_extracts_multiple(self) -> None:
        """Close chirps: algorithm returns several finite modes with energy."""
        demo = generate_close_modes_demo(fs=400.0, duration=0.6, noise_std=0.0)
        alg = DD_ACMD(
            fs=demo["fs"],
            k_max=5,
            max_iter=60,
            tol=1e-8,
            energy_tol=0.05,
            ddifi_max_iter=3,
        )
        imfs = alg.fit_transform(demo["signal"])
        self.assertGreaterEqual(imfs.shape[0], 2)
        self.assertTrue(np.all(np.isfinite(imfs)))
        self.assertGreater(
            float(np.sum(imfs**2)), 0.05 * float(np.sum(demo["signal"] ** 2))
        )


if __name__ == "__main__":
    unittest.main()
