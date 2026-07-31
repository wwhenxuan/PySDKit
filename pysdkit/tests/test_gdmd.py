# -*- coding: utf-8 -*-
"""
Created on 2025/07/31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest

import numpy as np

from pysdkit import GDMD, gdmd
from pysdkit._gdmd import (
    curve_smooth,
    differ,
    gdmd_core,
    make_dispersive_signal,
    spectrum_to_time,
    unilateral_spectrum,
)


class GDMDTest(unittest.TestCase):
    """Unit tests for Generalized Dispersion Mode Decomposition."""

    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.t,
            cls.signal,
            cls.f,
            cls.spectrum,
            cls.true_gds,
            cls.true_modes,
        ) = make_dispersive_signal(samp_freq=100.0, duration=15.0)
        cls.fs = 100.0
        cls.duration = 15.0
        cls.init_gd = curve_smooth(cls.true_gds, 1e-7)

    def test_fit_transform_shape(self) -> None:
        decomp = GDMD(alpha=1e-3, beta=1e-7, tol=1e-6, max_iter=80)
        modes = decomp.fit_transform(
            self.signal, fs=self.fs, init_gd=self.init_gd, smooth_init_beta=None
        )
        self.assertEqual(modes.ndim, 2)
        self.assertEqual(modes.shape, (3, self.signal.size))

    def test_default_call(self) -> None:
        decomp = GDMD(alpha=1e-3, beta=1e-7, tol=1e-6, max_iter=60)
        a = decomp(self.signal, fs=self.fs, init_gd=self.init_gd)
        b = decomp.fit_transform(
            self.signal, fs=self.fs, init_gd=self.init_gd, smooth_init_beta=None
        )
        # second call re-runs; shapes must match
        self.assertEqual(a.shape, b.shape)

    def test_return_all(self) -> None:
        decomp = GDMD(alpha=1e-3, beta=1e-7, tol=1e-6, max_iter=80)
        modes, gds, modes_f = decomp.fit_transform(
            self.signal,
            fs=self.fs,
            init_gd=self.init_gd,
            return_all=True,
            smooth_init_beta=None,
        )
        self.assertEqual(modes.shape[0], gds.shape[0])
        self.assertEqual(modes_f.shape, gds.shape)
        recon = modes.sum(axis=0)
        rel = np.linalg.norm(recon - self.signal) / np.linalg.norm(self.signal)
        self.assertLess(rel, 0.05)

    def test_functional_interface(self) -> None:
        modes = gdmd(
            self.signal,
            fs=self.fs,
            init_gd=self.init_gd,
            alpha=1e-3,
            beta=1e-7,
            tol=1e-6,
            max_iter=60,
        )
        self.assertEqual(modes.shape[1], self.signal.size)

    def test_recovers_group_delays(self) -> None:
        """Paper Example-1 style: recovered GDs should be close to ground truth."""
        decomp = GDMD(alpha=1e-3, beta=1e-7, tol=1e-8, max_iter=200)
        gd, modes_f = decomp.decompose_spectrum(
            self.spectrum, self.duration, self.init_gd
        )
        for k in range(3):
            re_db = 20.0 * np.log10(
                np.linalg.norm(gd[k] - self.true_gds[k])
                / (np.linalg.norm(self.true_gds[k]) + 1e-30)
            )
            self.assertLess(re_db, -20.0, msg=f"mode {k} GD RE={re_db:.1f} dB")

        modes_t = np.vstack([spectrum_to_time(m, self.signal.size) for m in modes_f])
        for k in range(3):
            corr = np.corrcoef(modes_t[k], self.true_modes[k])[0, 1]
            self.assertGreater(corr, 0.95, msg=f"mode {k} corr={corr}")

    def test_decompose_spectrum_history(self) -> None:
        gd, modes, gd_hist, s_hist = gdmd_core(
            self.spectrum,
            self.duration,
            self.init_gd,
            alpha=1e-3,
            beta=1e-7,
            tol=1e-6,
            max_iter=40,
        )
        self.assertEqual(gd.shape, self.true_gds.shape)
        self.assertEqual(modes.shape, self.true_gds.shape)
        self.assertEqual(gd_hist.shape[:2], self.true_gds.shape)
        self.assertGreaterEqual(gd_hist.shape[-1], 1)
        self.assertEqual(s_hist.shape, gd_hist.shape)

    def test_successive_extraction(self) -> None:
        """Envelope-peak successive mode extraction (Example-2 style API)."""
        fs = 500.0
        t = np.arange(0.0, 0.4, 1.0 / fs)
        s = np.exp(-120.0 * (t - 0.10) ** 2) * np.sin(2 * np.pi * 60.0 * t)
        s += 0.8 * np.exp(-150.0 * (t - 0.25) ** 2) * np.sin(2 * np.pi * 90.0 * t)
        decomp = GDMD(alpha=1e-4, beta=1e-7, tol=1e-5, max_iter=50, K=2)
        modes = decomp.fit_transform(s, fs=fs)
        self.assertEqual(modes.shape, (2, s.size))
        self.assertTrue(np.all(np.isfinite(modes)))

    def test_helpers(self) -> None:
        # linspace(0, 1, 21) with delta=0.05 → unit slope
        y = np.linspace(0.0, 1.0, 21)
        dy = differ(y, 0.05)
        self.assertEqual(dy.shape, y.shape)
        self.assertTrue(np.allclose(dy[2:-2], 1.0, atol=0.05))

        smooth = curve_smooth(self.true_gds, 1e-4)
        self.assertEqual(smooth.shape, self.true_gds.shape)

        uni = unilateral_spectrum(self.signal)
        self.assertEqual(uni.size, self.signal.size // 2 + 1)
        recon = spectrum_to_time(uni, self.signal.size)
        self.assertTrue(np.allclose(recon, self.signal, atol=1e-10))

    def test_str(self) -> None:
        self.assertIn("GDMD", str(GDMD()))

    def test_short_signal(self) -> None:
        with self.assertRaises(ValueError):
            GDMD(K=1).fit_transform(np.ones(4), fs=10.0)

    def test_missing_init_and_k(self) -> None:
        with self.assertRaises(ValueError):
            GDMD().fit_transform(self.signal, fs=self.fs)

    def test_init_gd_shape(self) -> None:
        with self.assertRaises(ValueError):
            GDMD().fit_transform(self.signal, fs=self.fs, init_gd=np.ones((3, 10)))


if __name__ == "__main__":
    unittest.main()
