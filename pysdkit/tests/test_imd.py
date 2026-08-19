# -*- coding: utf-8 -*-
"""
Unit tests for Impulsive Mode Decomposition (IMD).

References
----------
[1] B. Hou et al., "Impulsive mode decomposition," Mech. Syst. Signal
    Process., 211:111227, 2024.
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy.signal import hilbert

from pysdkit import IMD, imd
from pysdkit._imd.imd import (
    fft_bandpass,
    fre_am,
    segment_sparsity,
    cesm_pq_mean,
    band_split,
)
from pysdkit.data import load_imd_gearbox_snippet, load_imd_input_sig


def _make_impulsive_mixture(
    fs: float = 2000.0,
    duration: float = 0.4,
    seed: int = 0,
):
    """Two band-limited impulse trains plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, 1.0 / fs)
    n = t.size

    def impulse_train(period: float, carrier: float, decay: float) -> np.ndarray:
        y = np.zeros(n)
        idx = np.arange(0, n, int(round(period * fs)))
        for k in idx:
            tau = t[k:] - t[k]
            y[k:] += np.exp(-decay * tau) * np.sin(2.0 * np.pi * carrier * tau)
        return y

    m1 = impulse_train(period=0.05, carrier=180.0, decay=80.0)
    m2 = 0.8 * impulse_train(period=0.08, carrier=420.0, decay=100.0)
    signal = m1 + m2 + 0.05 * rng.standard_normal(n)
    return t, signal, m1, m2


class IMDHelpersTest(unittest.TestCase):
    def test_fft_bandpass_passband_energy(self):
        fs = 1000.0
        t = np.arange(0.0, 1.0, 1.0 / fs)
        x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)
        y = fft_bandpass(x, fs, 40.0, 60.0)
        f, a = fre_am(y, fs)
        self.assertGreater(
            a[np.argmin(np.abs(f - 50.0))], a[np.argmin(np.abs(f - 200.0))]
        )

    def test_segment_sparsity_impulsive_vs_noise(self):
        rng = np.random.default_rng(1)
        n = 4000
        noise = rng.standard_normal(n)
        impulses = np.zeros(n)
        impulses[::200] = 5.0
        impulses = np.convolve(impulses, np.exp(-np.linspace(0, 5, 50)), mode="same")
        s_noise = segment_sparsity(noise, n_seg=8)
        s_imp = segment_sparsity(impulses + 0.1 * noise, n_seg=8)
        self.assertGreater(s_imp, s_noise)

    def test_cesm_and_band_split(self):
        x = np.abs(np.random.default_rng(0).standard_normal(1000)) + 0.1
        val = cesm_pq_mean(x, n_seg=5, a=-10.0, p=2.0, q=1.0)
        self.assertTrue(np.isfinite(val))
        self.assertGreater(val, 0.0)
        parts = band_split(np.array([0.0, 100.0]), np.array([20.0, 40.0]), 1.0)
        self.assertEqual(parts.shape[0], 2)


class IMDAlgorithmTest(unittest.TestCase):
    def setUp(self):
        self.t, self.signal, self.m1, self.m2 = _make_impulsive_mixture()
        self.fs = 2000.0

    def test_call_and_fit_transform(self):
        decomp = IMD(
            n_particles=12,
            max_iter=5,
            threshold=1.05,
            max_modes=2,
            seg_num=5,
            seed=0,
        )
        a = decomp.fit_transform(self.signal, fs=self.fs)
        b = decomp(self.signal, fs=self.fs)
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.shape[1], self.signal.size)
        self.assertGreaterEqual(a.shape[0], 1)

    def test_functional_api(self):
        modes = imd(
            self.signal,
            fs=self.fs,
            n_particles=10,
            max_iter=4,
            threshold=1.05,
            max_modes=2,
            seg_num=5,
            seed=1,
        )
        self.assertEqual(modes.ndim, 2)
        self.assertEqual(modes.shape[1], self.signal.size)

    def test_return_all_and_bands(self):
        decomp = IMD(
            n_particles=12,
            max_iter=5,
            threshold=1.05,
            max_modes=3,
            seg_num=5,
            seed=2,
        )
        out = decomp.fit_transform(self.signal, fs=self.fs, return_all=True)
        self.assertIsInstance(out, dict)
        self.assertIn("modes", out)
        self.assertIn("selected_bands", out)
        self.assertIn("residual", out)
        bands = out["selected_bands"]
        self.assertEqual(bands.shape[1], 3)
        self.assertTrue(np.all(bands[:, 0] <= bands[:, 1]))
        # fitness sorted descending
        self.assertTrue(np.all(np.diff(bands[:, 2]) <= 1e-12))

    def test_modes_are_bandlimited(self):
        decomp = IMD(
            n_particles=15,
            max_iter=6,
            threshold=1.05,
            max_modes=2,
            seg_num=5,
            seed=3,
        )
        modes = decomp.fit_transform(self.signal, fs=self.fs)
        self.assertIsNotNone(decomp.selected_bands)
        assert decomp.selected_bands is not None
        for i, mode in enumerate(modes):
            f_lo, f_hi = decomp.selected_bands[i, :2]
            # re-filter should nearly reproduce the mode
            again = fft_bandpass(self.signal, self.fs, f_lo, f_hi)
            corr = abs(float(np.corrcoef(mode, again)[0, 1]))
            self.assertGreater(corr, 0.99)

    def test_invalid_inputs(self):
        decomp = IMD(n_particles=5, max_iter=2, seed=0)
        with self.assertRaises(ValueError):
            decomp.fit_transform(np.arange(8.0), fs=1000.0)
        with self.assertRaises(ValueError):
            decomp.fit_transform(np.arange(64.0))
        with self.assertRaises(ValueError):
            decomp.fit_transform(np.arange(64.0), fs=0.0)

    def test_str(self):
        self.assertIn("IMD", str(IMD()))


class IMDDemoDataTest(unittest.TestCase):
    def test_load_input_sig(self):
        demo = load_imd_input_sig()
        self.assertEqual(demo["signal"].shape, (25600,))
        self.assertAlmostEqual(float(demo["fs"]), 12800.0)
        self.assertEqual(demo["t"].size, demo["signal"].size)

    def test_load_gearbox_snippet(self):
        demo = load_imd_gearbox_snippet()
        self.assertEqual(demo["signal"].shape, (4096,))
        self.assertAlmostEqual(float(demo["fs"]), 12800.0)
        self.assertTrue(np.isfinite(demo["signal"]).all())

    def test_input_sig_decomposition_smoke(self):
        demo = load_imd_input_sig()
        # Use a shorter window for a fast unit test.
        x = demo["signal"][:4096]
        fs = float(demo["fs"])
        decomp = IMD(
            n_particles=10,
            max_iter=4,
            threshold=1.2,
            max_modes=2,
            seg_num=5,
            seed=0,
        )
        modes = decomp.fit_transform(x, fs=fs)
        self.assertGreaterEqual(modes.shape[0], 1)
        self.assertEqual(modes.shape[1], x.size)
        self.assertTrue(np.isfinite(modes).all())
        # Squared-envelope spectrum of first mode should be non-trivial.
        se = np.abs(hilbert(modes[0])) ** 2
        f, a = fre_am(se, fs)
        self.assertGreater(float(np.max(a[1:])), 0.0)


if __name__ == "__main__":
    unittest.main()
