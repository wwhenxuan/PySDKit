# -*- coding: utf-8 -*-
"""
Unit tests for Ensemble Empirical Mode Decomposition (EEMD).

Wu, Z. and Huang, N. E. (2009). Ensemble empirical mode decomposition:
a noise-assisted data analysis method. Advances in Adaptive Data Analysis.
"""

import unittest

import numpy as np

from pysdkit import EEMD, EMD
from pysdkit.data import test_emd, test_univariate_signal


def _intermittent_signal(n: int = 400, seed: int = 0) -> np.ndarray:
    """
    Classic mode-mixing style signal: low-frequency carrier + sparse HF bursts.

    This is the intermittency setting highlighted by Wu & Huang as a
    motivation for EEMD.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    carrier = np.sin(2.0 * np.pi * 3.0 * t)
    bursts = np.zeros_like(t)
    for centre in (0.2, 0.55, 0.8):
        mask = np.abs(t - centre) < 0.04
        bursts[mask] = 0.6 * np.sin(2.0 * np.pi * 45.0 * t[mask])
    return carrier + bursts + 0.02 * rng.standard_normal(n)


class EEMDTest(unittest.TestCase):
    """Automated tests for EEMD."""

    def test_str(self) -> None:
        self.assertIn("EEMD", str(EEMD()))

    def test_fit_transform_shape(self) -> None:
        """Output must be 2-D with matching temporal length."""
        eemd = EEMD(trials=12, noise_width=0.05, max_imfs=4, random_seed=0)
        for case in range(1, 4):
            _, signal = test_univariate_signal(case=case)
            imfs = eemd.fit_transform(signal)
            self.assertEqual(imfs.ndim, 2)
            self.assertEqual(imfs.shape[1], signal.size)
            self.assertGreaterEqual(imfs.shape[0], 1)

    def test_default_call(self) -> None:
        """``__call__`` must match ``fit_transform`` on a fresh instance."""
        _, signal = test_emd()
        a = EEMD(trials=10, max_imfs=3, random_seed=1)(signal)
        b = EEMD(trials=10, max_imfs=3, random_seed=1).fit_transform(signal)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_reconstruction_via_residue(self) -> None:
        """
        EEMD returns ensemble-mean IMFs; the stored residue closes the sum.

        Note (Wu & Huang / Torres et al.): unlike CEEMDAN, the ensemble mean
        alone need not reconstruct ``x`` exactly — residual ensemble noise remains.
        """
        _, signal = test_emd()
        eemd = EEMD(trials=16, max_imfs=4, random_seed=2)
        imfs = eemd.fit_transform(signal)
        imfs_stored, residue = eemd.get_imfs_and_residue()
        self.assertTrue(np.allclose(imfs, imfs_stored))
        recon = np.sum(imfs, axis=0) + residue
        self.assertTrue(np.allclose(recon, signal, atol=1e-10))

    def test_get_imfs_and_residue_without_running(self) -> None:
        with self.assertRaises(ValueError):
            EEMD().get_imfs_and_residue()

    def test_get_imfs_and_trend(self) -> None:
        time = np.linspace(0.0, 1.0, 256, endpoint=False)
        signal = np.sin(2.0 * np.pi * 5.0 * time) + 2.0 * time
        eemd = EEMD(trials=12, max_imfs=3, random_seed=3)
        eemd.fit_transform(signal, time=time)
        imfs, trend = eemd.get_imfs_and_trend()
        self.assertEqual(imfs.ndim, 2)
        self.assertEqual(trend.shape, signal.shape)
        self.assertTrue(np.all(np.isfinite(trend)))

    def test_different_length_inputs(self) -> None:
        eemd = EEMD(trials=4, random_seed=0)
        with self.assertRaises(ValueError):
            eemd.fit_transform(signal=np.random.randn(125), time=np.arange(100))

    def test_generate_noise_kinds(self) -> None:
        eemd_n = EEMD(noise_kind="normal", random_seed=0)
        eemd_u = EEMD(noise_kind="uniform", random_seed=0)
        n = eemd_n.generate_noise(scale=0.2, size=500)
        u = eemd_u.generate_noise(scale=0.2, size=500)
        self.assertEqual(n.shape, (500,))
        self.assertEqual(u.shape, (500,))
        self.assertLess(np.abs(n.mean()), 0.08)
        self.assertTrue(np.all(u >= -0.1) and np.all(u <= 0.1))

        eemd_bad = EEMD(noise_kind="laplace")
        with self.assertRaises(ValueError):
            eemd_bad.generate_noise(0.1, 10)

    def test_ensemble_statistics(self) -> None:
        """After fitting, ensemble mean / std / count must be consistent."""
        signal = _intermittent_signal(n=300)
        eemd = EEMD(trials=15, noise_width=0.1, max_imfs=3, random_seed=4)
        imfs = eemd.fit_transform(signal)
        mean = eemd.ensemble_mean()
        std = eemd.ensemble_std()
        counts = eemd.ensemble_count()
        self.assertTrue(np.allclose(mean, imfs))
        self.assertEqual(std.shape, imfs.shape)
        self.assertEqual(len(counts), imfs.shape[0])
        self.assertTrue(all(c >= 1 for c in counts))
        self.assertEqual(len(eemd.all_imfs), imfs.shape[0])

    def test_reproducible_with_seed(self) -> None:
        signal = _intermittent_signal(n=250)
        a = EEMD(trials=10, max_imfs=3, random_seed=7).fit_transform(signal)
        b = EEMD(trials=10, max_imfs=3, random_seed=7).fit_transform(signal)
        self.assertTrue(np.allclose(a, b))

    def test_more_trials_reduce_ensemble_std(self) -> None:
        """
        Paper intuition: as the ensemble grows, trial-to-trial noise cancels.

        We check that the average ensemble std of IMF-0 decreases (or does not
        increase substantially) when trials increase.
        """
        signal = _intermittent_signal(n=280, seed=1)
        eemd_small = EEMD(trials=8, noise_width=0.15, max_imfs=2, random_seed=11)
        eemd_small.fit_transform(signal)
        eemd_large = EEMD(trials=40, noise_width=0.15, max_imfs=2, random_seed=11)
        eemd_large.fit_transform(signal)
        self.assertLessEqual(
            float(np.mean(eemd_large.ensemble_std()[0])),
            float(np.mean(eemd_small.ensemble_std()[0])) * 1.15,
        )

    def test_ext_emd_injection(self) -> None:
        """Custom EMD backends must be accepted."""
        emd = EMD(max_imfs=2, max_iteration=200)
        eemd = EEMD(ext_EMD=emd, trials=6, max_imfs=2, random_seed=0)
        _, signal = test_emd()
        imfs = eemd.fit_transform(signal[:300])
        self.assertEqual(imfs.ndim, 2)
        self.assertLessEqual(imfs.shape[0], 3)

    def test_separate_trends_flag(self) -> None:
        time = np.linspace(0.0, 1.0, 200, endpoint=False)
        signal = np.sin(2.0 * np.pi * 4.0 * time) + 1.5 * time
        eemd = EEMD(
            trials=8,
            max_imfs=2,
            separate_trends=True,
            random_seed=5,
        )
        imfs = eemd.fit_transform(signal, time=time)
        self.assertGreaterEqual(imfs.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(imfs)))

    def test_mode_mixing_signal_produces_multiple_imfs(self) -> None:
        """Intermittent HF bursts should not collapse into a single IMF."""
        signal = _intermittent_signal()
        imfs = EEMD(trials=20, noise_width=0.2, max_imfs=4, random_seed=9).fit_transform(
            signal
        )
        self.assertGreaterEqual(imfs.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
