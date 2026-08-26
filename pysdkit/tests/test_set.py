# -*- coding: utf-8 -*-
"""
Unit tests for the Synchroextracting Transform (SET).
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit import SET, set
from pysdkit._tfa.set import SET as ModuleSET
from pysdkit._tfa.set import (
    brevridge,
    brevridge_mult,
    frequency_axis_set,
    gaussian_window,
    odd_window_length,
    reconstruct_from_ridges,
    set_transform,
    stft_gaussian_pair,
)
from pysdkit.data import load_set_batdata2, load_set_vibdata


def _fm_tone(n_samples: int = 256, freq: float = 0.12) -> np.ndarray:
    time = np.arange(n_samples, dtype=float)
    return np.cos(2.0 * np.pi * freq * time)


class SETHelperTest(unittest.TestCase):
    """Window, STFT and ridge helpers."""

    def test_odd_window_length(self) -> None:
        self.assertEqual(odd_window_length(40), 41)
        self.assertEqual(odd_window_length(41), 41)

    def test_odd_window_rejects_tiny(self) -> None:
        with self.assertRaises(ValueError):
            odd_window_length(2)

    def test_gaussian_window_peak_and_derivative(self) -> None:
        window, deriv = gaussian_window(41)
        self.assertEqual(window.size, 41)
        self.assertEqual(int(np.argmax(window)), 20)
        self.assertAlmostEqual(
            float(np.max(np.abs(deriv + deriv[::-1]))), 0.0, places=12
        )
        self.assertGreater(float(window[20]), float(window[0]))

    def test_frequency_axis_set(self) -> None:
        freq = frequency_axis_set(400, fs=100.0)
        self.assertEqual(freq.size, 200)
        self.assertAlmostEqual(freq[0], 0.0)
        self.assertAlmostEqual(freq[1], 100.0 / 400.0)
        self.assertEqual(frequency_axis_set(401, fs=1.0).size, 201)

    def test_stft_shape(self) -> None:
        samples = _fm_tone(128)
        tfr, tfr_d, window, va = stft_gaussian_pair(samples, 17)
        self.assertEqual(tfr.shape, (64, 128))
        self.assertEqual(tfr_d.shape, (64, 128))
        self.assertEqual(window.size, 17)
        self.assertAlmostEqual(va, 128 / 17.0)

    def test_set_transform_shape_and_extraction(self) -> None:
        samples = _fm_tone(128)
        result = set_transform(samples, hlength=17, fs=1.0)
        self.assertEqual(result["te"].shape, (64, 128))
        self.assertEqual(result["tfr"].shape, (64, 128))
        self.assertTrue(np.all(np.isfinite(result["te"])))
        n_stft = int(np.count_nonzero(np.abs(result["tfr"]) > 1e-12))
        n_set = int(np.count_nonzero(np.abs(result["te"]) > 1e-12))
        self.assertLess(n_set, n_stft)
        self.assertTrue(np.all((result["seo"] == 0.0) | (result["seo"] == 1.0)))

    def test_rejects_2d_and_short(self) -> None:
        with self.assertRaises(ValueError):
            set_transform(np.ones((4, 4)))
        with self.assertRaises(ValueError):
            set_transform(np.ones(5))


class SETPublicTest(unittest.TestCase):
    """Public ``SET`` TFR, IF peak, and modal reconstruction."""

    def test_public_alias(self) -> None:
        self.assertIs(SET, ModuleSET)

    def test_transform_tone_peaks_near_carrier(self) -> None:
        freq_c = 0.12
        samples = _fm_tone(256, freq=freq_c)
        te, freq = SET(hlength=33, fs=1.0).transform(samples)
        peak_bin = int(np.argmax(np.mean(np.abs(te), axis=1)))
        self.assertAlmostEqual(float(freq[peak_bin]), freq_c, delta=0.03)

    def test_example1_if_track(self) -> None:
        fs = 100.0
        time = np.arange(0.0, 4.0, 1.0 / fs)
        phase = 25.0 * time + 10.0 * np.sin(1.5 * time)
        samples = np.exp(-0.5 * time) * np.sin(2.0 * np.pi * phase)
        inst_freq = 25.0 + 15.0 * np.cos(1.5 * time)
        engine = SET(hlength=40, fs=fs)
        te, freq = engine.transform(samples)
        peak_bin = np.argmax(np.abs(te), axis=0)
        est_hz = freq[peak_bin]
        mid = slice(40, -40)
        self.assertLess(float(np.median(np.abs(est_hz[mid] - inst_freq[mid]))), 3.0)

    def test_fit_transform_shape_and_residue(self) -> None:
        samples = _fm_tone(192, freq=0.1)
        imfs = SET(hlength=25, n_imfs=1, fs=1.0).fit_transform(samples)
        self.assertEqual(imfs.shape, (2, samples.size))
        recon = np.sum(imfs, axis=0)
        np.testing.assert_allclose(recon, samples, atol=1e-10)
        self.assertTrue(np.all(np.isfinite(imfs)))

    def test_call_matches_fit_transform(self) -> None:
        samples = _fm_tone(96, freq=0.08)
        engine = SET(hlength=17, n_imfs=1)
        np.testing.assert_allclose(engine(samples), engine.fit_transform(samples))

    def test_functional_set(self) -> None:
        samples = _fm_tone(96, freq=0.09)
        imfs = set(samples, hlength=17, n_imfs=1)
        self.assertEqual(imfs.shape[0], 2)
        self.assertEqual(imfs.shape[1], samples.size)

    def test_two_tones_modes_recover_sources(self) -> None:
        n_samples = 256
        time = np.arange(n_samples, dtype=float)
        s1 = np.sin(2.0 * np.pi * 0.06 * time)
        s2 = np.sin(2.0 * np.pi * 0.18 * time)
        mix = s1 + s2
        imfs = SET(hlength=41, n_imfs=2, clear_win=8).fit_transform(mix)
        modes = imfs[:2]
        corr = np.array(
            [
                [np.abs(np.corrcoef(modes[i], src)[0, 1]) for src in (s1, s2)]
                for i in range(2)
            ]
        )
        self.assertGreater(float(np.max(corr[:, 0])), 0.75)
        self.assertGreater(float(np.max(corr[:, 1])), 0.75)

    def test_multicomponent_example_energy(self) -> None:
        fs = 120.0
        time = np.arange(0.0, 4.0, 1.0 / fs)
        s1 = np.sin(
            2.0 * np.pi * (10.0 * time + 2.0 * np.arctan((2.0 * time - 2.0) ** 2))
        )
        s2 = np.sin(2.0 * np.pi * (32.0 * time + 10.0 * np.sin(time)))
        s3 = np.sin(2.0 * np.pi * (44.0 * time + 10.0 * np.sin(time)))
        mix = s1 + s2 + s3
        imfs = SET(hlength=55, fs=fs, n_imfs=3, clear_win=5).fit_transform(mix)
        self.assertEqual(imfs.shape, (4, mix.size))
        recon = np.sum(imfs[:3], axis=0)
        err = float(np.linalg.norm(mix - recon) / np.linalg.norm(mix))
        self.assertLess(err, 0.55)

    def test_batdata_smoke(self) -> None:
        record = load_set_batdata2()
        samples = np.asarray(record["signal"])
        engine = SET(hlength=45, fs=float(record["fs"]), n_imfs=4)
        te, freq = engine.transform(samples)
        self.assertEqual(te.shape[1], samples.size)
        self.assertEqual(freq.size, te.shape[0])
        imfs = engine.fit_transform(samples, n_imfs=4)
        self.assertEqual(imfs.shape, (5, samples.size))
        np.testing.assert_allclose(np.sum(imfs, axis=0), samples, atol=1e-10)

    def test_vibdata_tfr_finite(self) -> None:
        record = load_set_vibdata()
        te, freq = SET(hlength=50, fs=float(record["fs"])).transform(record["signal"])
        self.assertTrue(np.all(np.isfinite(te)))
        self.assertEqual(te.shape[1], np.asarray(record["signal"]).size)
        self.assertGreater(float(np.max(np.abs(te))), 0.0)

    def test_ridge_helpers(self) -> None:
        n_freq, n_time = 40, 60
        tx = np.zeros((n_freq, n_time))
        ridge = np.clip(
            np.round(10 + 4 * np.sin(np.linspace(0, 2 * np.pi, n_time))).astype(int),
            0,
            n_freq - 1,
        )
        tx[ridge, np.arange(n_time)] = 3.0
        tx += 0.05 * np.random.default_rng(0).random((n_freq, n_time))
        freq = np.arange(n_freq, dtype=float)
        curve, energy = brevridge(tx, freq, ridge_lambda=1.0)
        self.assertEqual(curve.size, n_time)
        self.assertGreater(energy, -np.inf)
        self.assertLess(float(np.mean(np.abs(curve - ridge))), 3.0)
        ridges, energies = brevridge_mult(tx, freq, n_ridges=1, clear_win=3)
        self.assertEqual(ridges.shape, (1, n_time))
        self.assertEqual(energies.size, 1)
        modes = reconstruct_from_ridges(tx.astype(np.complex128), ridges)
        self.assertEqual(modes.shape, (1, n_time))

    def test_invalid_init(self) -> None:
        with self.assertRaises(ValueError):
            SET(hlength=1)
        with self.assertRaises(ValueError):
            SET(fs=0.0)
        with self.assertRaises(ValueError):
            SET(n_imfs=0)


if __name__ == "__main__":
    unittest.main()
