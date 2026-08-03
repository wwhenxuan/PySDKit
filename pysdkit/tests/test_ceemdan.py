# -*- coding: utf-8 -*-
"""
Unit tests for Complete Ensemble EMD with Adaptive Noise (CEEMDAN).

Torres, M. E., Colominas, M. A., Schlotthauer, G. and Flandrin, P. (2011).
A complete ensemble empirical mode decomposition with adaptive noise.
IEEE ICASSP.
"""

import unittest
from typing import Tuple

import numpy as np

from pysdkit import CEEMDAN, EMD
from pysdkit.data import test_emd, test_univariate_signal


def _two_tone(n: int = 320) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    signal = np.sin(2.0 * np.pi * 6.0 * t) + 0.5 * np.sin(2.0 * np.pi * 28.0 * t)
    return t, signal


class CEEMDANTest(unittest.TestCase):
    """Automated tests for CEEMDAN."""

    def test_str(self) -> None:
        self.assertIn("CEEMDAN", str(CEEMDAN()))

    def test_fit_transform_shape(self) -> None:
        ceemdan = CEEMDAN(trials=10, epsilon=0.05, max_imfs=3, random_seed=0)
        for case in range(1, 4):
            _, signal = test_univariate_signal(case=case)
            imfs = ceemdan.fit_transform(signal)
            self.assertEqual(imfs.ndim, 2)
            self.assertEqual(imfs.shape[1], signal.size)
            self.assertGreaterEqual(imfs.shape[0], 1)

    def test_default_call(self) -> None:
        """``__call__`` must match ``fit_transform`` on a fresh instance."""
        _, signal = test_emd()
        a = CEEMDAN(trials=8, max_imfs=3, random_seed=1)(signal)
        b = CEEMDAN(trials=8, max_imfs=3, random_seed=1).fit_transform(signal)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_complete_reconstruction(self) -> None:
        """
        Paper Eq. (5): x = sum_k IMF~_k + R  (exact / numerically complete).

        In this implementation the residue is appended as the last row of
        ``fit_transform``, so the modes alone must reconstruct ``x``.
        """
        _, signal = test_emd()
        ceemdan = CEEMDAN(trials=12, epsilon=0.05, max_imfs=4, random_seed=2)
        imfs = ceemdan.fit_transform(signal)
        self.assertTrue(np.allclose(np.sum(imfs, axis=0), signal, atol=1e-8))

        stored, residue = ceemdan.get_imfs_and_residue()
        self.assertTrue(np.allclose(stored, imfs))
        # Residue stored after scaling is ~0 because R is already in ``imfs``
        self.assertTrue(np.allclose(residue, 0.0, atol=1e-8))

    def test_get_imfs_and_residue_without_running(self) -> None:
        with self.assertRaises(ValueError):
            CEEMDAN().get_imfs_and_residue()

    def test_get_imfs_and_trend(self) -> None:
        t, signal = _two_tone()
        signal = signal + 3.0 * t
        ceemdan = CEEMDAN(trials=10, max_imfs=3, random_seed=3)
        imfs = ceemdan.fit_transform(signal, time=t)
        modes, trend = ceemdan.get_imfs_and_trend()
        self.assertEqual(modes.ndim, 2)
        self.assertEqual(trend.shape, signal.shape)
        self.assertTrue(np.allclose(np.sum(imfs, axis=0), signal, atol=1e-8))

    def test_different_length_inputs(self) -> None:
        ceemdan = CEEMDAN(trials=4, random_seed=0)
        with self.assertRaises(ValueError):
            ceemdan.fit_transform(signal=np.random.randn(125), time=np.arange(100))

    def test_trend_only(self) -> None:
        time = np.arange(0.0, 1.0, 0.01)
        signal = 2.0 * time
        imfs = CEEMDAN(trials=8, max_imfs=2, random_seed=4).fit_transform(
            signal, time=time
        )
        self.assertGreaterEqual(imfs.shape[0], 1)
        self.assertTrue(np.allclose(np.sum(imfs, axis=0), signal, atol=1e-6))

    def test_single_tone_plus_trend(self) -> None:
        time = np.arange(0.0, 1.0, 0.002)
        cosine = np.cos(2.0 * np.pi * 4.0 * time)
        trend = 3.0 * (time - 0.5)
        imfs = CEEMDAN(
            trials=12, epsilon=0.05, max_imfs=3, random_seed=5
        ).fit_transform(cosine + trend, time=time)
        self.assertGreaterEqual(imfs.shape[0], 2)
        self.assertTrue(np.allclose(np.sum(imfs, axis=0), cosine + trend, atol=1e-6))

    def test_end_condition_max_imf(self) -> None:
        """Stopping when the requested IMF count is reached."""
        t, signal = _two_tone(n=200)
        ceemdan = CEEMDAN(
            trials=6,
            range_thr=1e-12,
            total_power_thr=1e-12,
            random_seed=0,
        )
        # Already extracted 2 modes while max_imf=2 → stop
        dummy = np.vstack([0.3 * signal, 0.2 * signal])
        self.assertTrue(ceemdan.end_condition(signal, dummy, max_imf=2))
        # Only 1 mode so far, residue still rich → continue
        self.assertFalse(ceemdan.end_condition(signal, dummy[:1] * 0.05, max_imf=3))

    def test_end_condition_range_and_power(self) -> None:
        ceemdan = CEEMDAN(range_thr=0.5, total_power_thr=1e6, trials=4, random_seed=0)
        signal = np.ones(64)
        cimfs = np.zeros((1, 64))
        # Flat residue → range criterion
        self.assertTrue(ceemdan.end_condition(signal, cimfs, max_imf=-1))

        ceemdan_pwr = CEEMDAN(
            range_thr=1e-12, total_power_thr=1.0, trials=4, random_seed=0
        )
        # Residue = signal (sum of empty-ish cIMFs), large absolute power
        # total_power_thr=1.0 with sum(|R|) = 64 should stop
        self.assertTrue(
            ceemdan_pwr.end_condition(np.ones(64), np.zeros((1, 64)), max_imf=-1)
        )

    def test_noise_kinds(self) -> None:
        t, signal = _two_tone(n=180)
        for kind in ("normal", "uniform"):
            imfs = CEEMDAN(
                trials=6,
                noise_kind=kind,
                max_imfs=2,
                random_seed=6,
            ).fit_transform(signal, time=t)
            self.assertEqual(imfs.shape[1], signal.size)
            self.assertTrue(np.allclose(np.sum(imfs, axis=0), signal, atol=1e-7))

        bad = CEEMDAN(noise_kind="laplace", trials=2, random_seed=0)
        with self.assertRaises(ValueError):
            bad._generate_noise(1.0, 10)

    def test_update_random_seed_changes_noise(self) -> None:
        ceemdan = CEEMDAN(random_seed=1)
        a = ceemdan._generate_noise(1.0, 200)
        ceemdan.update_random_seed(2)
        b = ceemdan._generate_noise(1.0, 200)
        self.assertFalse(np.allclose(a, b))

    def test_reproducible_with_seed(self) -> None:
        t, signal = _two_tone(n=220)
        a = CEEMDAN(trials=8, max_imfs=3, random_seed=8).fit_transform(signal, time=t)
        b = CEEMDAN(trials=8, max_imfs=3, random_seed=8).fit_transform(signal, time=t)
        self.assertTrue(np.allclose(a, b))

    def test_max_imfs_cap(self) -> None:
        """``max_imfs`` limits oscillatory modes before the residue row."""
        t, signal = _two_tone(n=240)
        imfs = CEEMDAN(trials=8, max_imfs=2, random_seed=9).fit_transform(
            signal, time=t
        )
        # at most 2 oscillatory cIMFs + 1 residue
        self.assertLessEqual(imfs.shape[0], 3)
        self.assertTrue(np.allclose(np.sum(imfs, axis=0), signal, atol=1e-7))

    def test_ext_emd_injection(self) -> None:
        emd = EMD(max_imfs=2, max_iteration=200)
        ceemdan = CEEMDAN(ext_EMD=emd, trials=6, max_imfs=2, random_seed=0)
        _, signal = test_emd()
        imfs = ceemdan.fit_transform(signal[:280])
        self.assertEqual(imfs.ndim, 2)
        self.assertTrue(np.allclose(np.sum(imfs, axis=0), signal[:280], atol=1e-7))

    def test_delta_like_impulse_decomposes(self) -> None:
        """
        Torres et al. use a discrete Dirac as a stress test for noise-assisted EMD.

        We only require a finite multi-mode output that reconstructs the impulse.
        """
        n = 128
        impulse = np.zeros(n)
        impulse[n // 2] = 1.0
        imfs = CEEMDAN(
            trials=20,
            epsilon=0.05,
            noise_scale=0.02,
            max_imfs=5,
            random_seed=10,
        ).fit_transform(impulse)
        self.assertGreaterEqual(imfs.shape[0], 2)
        self.assertTrue(np.allclose(np.sum(imfs, axis=0), impulse, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
