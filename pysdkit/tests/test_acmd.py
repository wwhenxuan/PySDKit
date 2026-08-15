# -*- coding: utf-8 -*-
"""
Unit tests for Adaptive Chirp Mode Decomposition (ACMD).

Covers every public helper and the ``ACMD`` class API, matching the
MATLAB package under ``Codes of ACMD``.
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit import ACMD
from pysdkit._acmd.acmd import (
    add_noise,
    compute_snr,
    curve_smooth,
    differ,
    find_ridges,
    second_order_difference,
    stft,
    tf_spectrum,
)


def _test1_signal(fs: float = 1000.0):
    """MATLAB Test1.m synthetic mixture (two oscillating-IF chirps)."""
    t = np.arange(0.0, 1.0 + 1.0 / fs, 1.0 / fs)
    sig1 = np.exp(-0.3 * t) * np.cos(
        2 * np.pi * (350 * t + (1.0 / (2 * np.pi)) * np.cos(2 * np.pi * 25 * t))
    )
    if1 = 350 - 25 * np.sin(50 * np.pi * t)
    sig2 = np.exp(-0.6 * t) * np.cos(
        2 * np.pi * (250 * t + (1.0 / (2 * np.pi)) * np.cos(2 * np.pi * 20 * t))
    )
    if2 = 250 - 20 * np.sin(40 * np.pi * t)
    return t, sig1 + sig2, sig1, sig2, if1, if2


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

    def test_class_static_differ_matches_module(self) -> None:
        y = np.linspace(0, 1, 50)
        a = differ(y, 0.02)
        b = ACMD.differ(y, 0.02)
        self.assertTrue(np.allclose(a, b))


class NoiseAndSNRTest(unittest.TestCase):
    def test_add_noise_stats(self) -> None:
        rng = np.random.default_rng(0)
        n = add_noise(5000, mean=0.0, std=0.2, rng=rng)
        self.assertEqual(n.size, 5000)
        self.assertAlmostEqual(float(np.mean(n)), 0.0, delta=0.05)
        self.assertAlmostEqual(float(np.std(n)), 0.2, delta=0.05)

    def test_snr_perfect_is_inf(self) -> None:
        x = np.sin(np.linspace(0, 10, 200))
        self.assertEqual(compute_snr(x, x), float("inf"))

    def test_snr_positive_for_mild_noise(self) -> None:
        clean = np.sin(np.linspace(0, 20, 1000))
        noisy = clean + 0.01 * np.random.default_rng(1).standard_normal(1000)
        self.assertGreater(compute_snr(clean, noisy), 20.0)


class SecondOrderAndSmoothTest(unittest.TestCase):
    def test_oper_shape(self) -> None:
        oper = second_order_difference(50)
        self.assertEqual(oper.shape, (48, 50))

    def test_curve_smooth_preserves_constant(self) -> None:
        f = 40.0 * np.ones(100)
        out = curve_smooth(f, beta=1e-4)
        self.assertEqual(out.shape, f.shape)
        self.assertTrue(np.allclose(out, 40.0, atol=1e-6))

    def test_curve_smooth_2d(self) -> None:
        f = np.vstack([np.ones(80), 2 * np.ones(80)])
        out = curve_smooth(f, beta=1e-6)
        self.assertEqual(out.shape, f.shape)


class RidgeAndSTFTTest(unittest.TestCase):
    def test_find_ridges_on_diagonal(self) -> None:
        m, n = 40, 30
        spec = np.zeros((m, n))
        for j in range(n):
            spec[10 + j // 5, j] = 1.0
        idx = find_ridges(spec, delta=5)
        self.assertEqual(idx.shape, (n,))
        self.assertTrue(np.all((idx >= 0) & (idx < m)))

    def test_stft_shapes(self) -> None:
        fs = 200.0
        t = np.arange(0, 1, 1 / fs)
        x = np.cos(2 * np.pi * 40 * t)
        spec, f = stft(x, fs, n_fft=128, win_len=32)
        self.assertEqual(spec.shape, (128, t.size))
        self.assertEqual(f.size, 128)
        self.assertAlmostEqual(float(f[0]), -fs / 2.0, places=6)
        self.assertAlmostEqual(float(f[-1]), fs / 2.0, places=6)


class TFSpectrumTest(unittest.TestCase):
    def test_tf_spectrum_shapes(self) -> None:
        ifs = np.vstack([30 * np.ones(50), 60 * np.ones(50)])
        ias = np.ones_like(ifs)
        a_spec, fbin = tf_spectrum(ifs, ias, band=(0.0, 100.0), fr_num=128)
        self.assertEqual(a_spec.shape, (128, 50))
        self.assertEqual(fbin.size, 128)
        self.assertGreater(float(a_spec.sum()), 0.0)


class ACMDInitAndStrTest(unittest.TestCase):
    def test_str(self) -> None:
        self.assertIn("ACMD", str(ACMD(K=2, fs=1000)))

    def test_init_stores_params(self) -> None:
        alg = ACMD(K=3, fs=500.0, alpha0=1e-6, beta=1e-9, tol=1e-7, max_iter=50)
        self.assertEqual(alg.K, 3)
        self.assertEqual(alg.fs, 500.0)
        self.assertEqual(alg.max_iter, 50)

    def test_init_if1_near_true_tone(self) -> None:
        fs = 1000.0
        n = 1000
        t = np.arange(n) / fs
        x = np.cos(2 * np.pi * 120 * t)
        ini = ACMD.init_IF1(x, fs, n)
        self.assertEqual(ini.shape, (n,))
        self.assertAlmostEqual(float(ini[0]), 120.0, delta=2.0)


class ACMDIterAndExtractTest(unittest.TestCase):
    def test_extract_mode_shapes(self) -> None:
        fs = 500.0
        t = np.arange(0, 0.5, 1 / fs)
        x = np.cos(2 * np.pi * 40 * t)
        alg = ACMD(K=1, fs=fs, alpha0=1e-4, beta=1e-5, tol=1e-7, max_iter=80)
        sest, if_est, ia_est = alg.extract_mode(x, 40.0 * np.ones_like(x))
        self.assertEqual(sest.shape, x.shape)
        self.assertEqual(if_est.shape, x.shape)
        self.assertEqual(ia_est.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(sest)))
        self.assertTrue(np.all(np.isfinite(if_est)))

    def test_iter_recovers_constant_if_tone(self) -> None:
        fs = 1000.0
        t = np.arange(0, 0.4, 1 / fs)
        f0 = 80.0
        x = np.cos(2 * np.pi * f0 * t)
        alg = ACMD(K=1, fs=fs, alpha0=1e-3, beta=1e-4, tol=1e-8, max_iter=100)
        sest, if_est, ia_est = alg.iter(x, f0 * np.ones_like(x), len(x), fs)
        self.assertGreater(compute_snr(x, sest), 15.0)
        self.assertAlmostEqual(float(np.median(if_est)), f0, delta=3.0)
        self.assertAlmostEqual(float(np.median(ia_est)), 1.0, delta=0.15)

    def test_iter_rejects_length_mismatch(self) -> None:
        alg = ACMD(K=1, fs=100.0)
        with self.assertRaises(ValueError):
            alg.iter(np.ones(10), np.ones(8), 10, 100.0)


class ACMDFitTransformTest(unittest.TestCase):
    def test_fit_transform_k_modes(self) -> None:
        t, sig, *_ = _test1_signal(fs=500.0)
        # shorter / lower fs for speed: rebuild a compact two-tone mixture
        fs = 500.0
        t = np.arange(0, 0.6, 1 / fs)
        s1 = np.exp(-0.3 * t) * np.cos(
            2 * np.pi * (120 * t + (1.0 / (2 * np.pi)) * np.cos(2 * np.pi * 8 * t))
        )
        s2 = np.exp(-0.6 * t) * np.cos(
            2 * np.pi * (70 * t + (1.0 / (2 * np.pi)) * np.cos(2 * np.pi * 6 * t))
        )
        sig = s1 + s2
        alg = ACMD(K=2, fs=fs, alpha0=1e-3, beta=1e-4, tol=1e-7, max_iter=80)
        imfs = alg.fit_transform(sig)
        self.assertEqual(imfs.shape, (2, sig.size))
        self.assertIsNotNone(alg.imfs_)
        # residual energy should drop after two modes
        residual = sig - imfs.sum(axis=0)
        self.assertLess(np.linalg.norm(residual), 0.6 * np.linalg.norm(sig))

    def test_return_all_and_call(self) -> None:
        fs = 400.0
        t = np.arange(0, 0.5, 1 / fs)
        x = np.cos(2 * np.pi * 50 * t) + 0.7 * np.cos(2 * np.pi * 90 * t)
        alg = ACMD(K=2, fs=fs, alpha0=1e-3, beta=1e-4, tol=1e-7, max_iter=60)
        imfs, ifs, ias = alg(x, return_all=True)
        self.assertEqual(imfs.shape, (2, x.size))
        self.assertEqual(ifs.shape, imfs.shape)
        self.assertEqual(ias.shape, imfs.shape)

    def test_test1_reconstruction_quality(self) -> None:
        """Closer to MATLAB Test1: two modes, fs=1000, expect usable SNR."""
        t, sig, sig1, sig2, if1, if2 = _test1_signal(fs=1000.0)
        alg = ACMD(K=2, fs=1000.0, alpha0=1e-3, beta=1e-4, tol=1e-8, max_iter=200)
        imfs, ifs, ias = alg.fit_transform(sig, return_all=True)
        # Match modes by correlation (order follows spectral peak strength)
        snrs = []
        for true in (sig1, sig2):
            best = max(compute_snr(true, imfs[i]) for i in range(2))
            snrs.append(best)
        self.assertTrue(all(s > 5.0 for s in snrs), msg=f"SNRs={snrs}")
        # IF median should be near component centers after matching
        for true_if in (if1, if2):
            best_if_err = min(
                np.linalg.norm(ifs[i] - true_if) / np.linalg.norm(true_if)
                for i in range(2)
            )
            self.assertLess(best_if_err, 0.15)


if __name__ == "__main__":
    unittest.main()
