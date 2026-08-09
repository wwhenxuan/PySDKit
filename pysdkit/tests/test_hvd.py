# -*- coding: utf-8 -*-
"""
Unit tests for Hilbert Vibration Decomposition (HVD).

Feldman, M. (2006). Time-varying vibration decomposition and analysis based
on the Hilbert transform. Journal of Sound and Vibration, 295:518–530.
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit import HVD
from pysdkit._tid.hvd import filter_by_window, make_window_real_valued
from pysdkit.data import test_emd


def _two_tone(
    n: int = 2000,
    fs: float = 1000.0,
    f1: float = 40.0,
    f2: float = 12.0,
    a1: float = 1.2,
    a2: float = 0.5,
) -> tuple:
    t = np.arange(n, dtype=float) / fs
    s1 = a1 * np.cos(2.0 * np.pi * f1 * t)
    s2 = a2 * np.cos(2.0 * np.pi * f2 * t)
    return t, s1 + s2, s1, s2, fs


def _paper_square_wave(n: int = 2048) -> np.ndarray:
    """Non-stationary square wave from Feldman (2006), Sec. 5.1."""
    k = np.arange(n, dtype=float)
    return (1.0 + 0.003 * k) * np.sign(np.sin((0.02 + 3.0e-5 * k) * k))


class HVDHelperTest(unittest.TestCase):
    def test_make_window_real_valued_symmetry(self) -> None:
        n = 64
        h = np.zeros(n)
        h[:8] = 1.0
        hr = make_window_real_valued(h, n)
        self.assertTrue(np.allclose(hr[n // 2 + 1 :], hr[1 : n // 2][::-1]))

    def test_square_window_shape(self) -> None:
        hvd = HVD(fpar=10)
        w = hvd.square_window(128)
        self.assertEqual(w.shape, (128,))
        self.assertTrue(np.all(w[:10] == 1.0))

    def test_filter_by_window_dc(self) -> None:
        n = 128
        x = np.ones(n)
        h = np.ones(n)
        y = filter_by_window(x, h)
        self.assertTrue(np.allclose(np.real(y), x, atol=1e-10))


class HVDTest(unittest.TestCase):
    def test_str(self) -> None:
        self.assertIn("HVD", str(HVD()))

    def test_invalid_params(self) -> None:
        with self.assertRaises(ValueError):
            HVD(K=0)
        with self.assertRaises(ValueError):
            HVD(fpar=0)

    def test_fit_transform_shape(self) -> None:
        _, y, _, _, _ = _two_tone()
        imfs = HVD(K=2, fpar=40).fit_transform(y)
        self.assertEqual(imfs.shape, (2, y.size))
        self.assertTrue(np.isrealobj(imfs))

    def test_default_call(self) -> None:
        _, y, _, _, _ = _two_tone(n=1024)
        hvd = HVD(K=2, fpar=30)
        a = hvd(y)
        b = hvd.fit_transform(y)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_return_all_frequencies(self) -> None:
        _, y, _, _, fs = _two_tone()
        imfs, freqs = HVD(K=2, fpar=40).fit_transform(y, return_all=True)
        self.assertEqual(imfs.shape[0], freqs.size)
        # Carriers in cycles/sample → Hz
        f_hz = np.sort(freqs * fs)[::-1]
        self.assertAlmostEqual(f_hz[0], 40.0, delta=0.5)
        self.assertAlmostEqual(f_hz[1], 12.0, delta=0.5)

    def test_two_tone_separation(self) -> None:
        """Largest-amplitude tone is extracted first (paper §4.1)."""
        _, y, s1, s2, _ = _two_tone()
        imfs = HVD(K=2, fpar=40).fit_transform(y)
        c0 = np.corrcoef(imfs[0], s1)[0, 1]
        c1 = np.corrcoef(imfs[1], s2)[0, 1]
        self.assertGreater(c0, 0.98)
        self.assertGreater(c1, 0.95)
        self.assertGreater(np.linalg.norm(imfs[0]), np.linalg.norm(imfs[1]))

    def test_reconstruction_two_tone(self) -> None:
        _, y, _, _, _ = _two_tone()
        imfs = HVD(K=2, fpar=40).fit_transform(y)
        err = np.linalg.norm(imfs.sum(0) - y) / np.linalg.norm(y)
        self.assertLess(err, 0.05)

    def test_amplitude_ordering(self) -> None:
        """Swap amplitudes: the stronger tone must still come first."""
        t = np.arange(2000) / 1000.0
        strong = 1.5 * np.cos(2 * np.pi * 15 * t)
        weak = 0.4 * np.cos(2 * np.pi * 45 * t)
        y = strong + weak
        imfs, freqs = HVD(K=2, fpar=40).fit_transform(y, return_all=True)
        self.assertAlmostEqual(freqs[0] * 1000.0, 15.0, delta=1.0)
        self.assertGreater(np.corrcoef(imfs[0], strong)[0, 1], 0.95)

    def test_mirror_flag(self) -> None:
        _, y, _, _, _ = _two_tone(n=1500)
        a = HVD(K=2, fpar=40, mirror=True).fit_transform(y)
        b = HVD(K=2, fpar=40, mirror=False).fit_transform(y)
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.shape[1], y.size)

    def test_paper_square_wave_components(self) -> None:
        """Sec. 5.1: non-stationary square wave yields ordered modes."""
        x = _paper_square_wave(2048)
        # Paper chooses LPF cut-off ≈ lowest frequency 0.02 rad/sample.
        # In FFT-bin units: fpar ≈ 0.02/(2π) * N_work.
        imfs = HVD(K=5, fpar=50, mirror=True).fit_transform(x)
        self.assertEqual(imfs.shape, (5, x.size))
        energies = np.sum(imfs**2, axis=1)
        # First few modes should carry most of the energy
        self.assertGreater(energies[0], energies[-1])
        recon = imfs.sum(0)
        err = np.linalg.norm(recon - x) / np.linalg.norm(x)
        self.assertLess(err, 0.40)

    def test_stores_attributes(self) -> None:
        _, y, _, _, _ = _two_tone(n=800)
        hvd = HVD(K=2, fpar=30)
        imfs = hvd.fit_transform(y)
        self.assertTrue(np.allclose(hvd.imfs, imfs))
        self.assertEqual(hvd.frequencies.shape, (2,))

    def test_short_signal_raises(self) -> None:
        with self.assertRaises(ValueError):
            HVD().fit_transform(np.ones(5))

    def test_builtin_signal_runs(self) -> None:
        _, signal = test_emd()
        imfs = HVD(K=3, fpar=25).fit_transform(signal)
        self.assertEqual(imfs.shape[1], signal.size)
        self.assertEqual(imfs.shape[0], 3)


if __name__ == "__main__":
    unittest.main()
