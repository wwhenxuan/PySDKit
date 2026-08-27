# -*- coding: utf-8 -*-
"""
Unit tests for underdetermined BSS.
"""

from __future__ import annotations

import unittest

import numpy as np

from pysdkit.data import load_bss_beam, load_bss_yk9
from pysdkit.utils import BSS as UtilsBSS
from pysdkit.utils.bss import (
    BSS,
    bss,
    cosine_distance,
    cosine_masks,
    default_window_length,
    frequency_axis_stft,
    frequency_energy,
    hamming_window,
    matlab_round,
    modal_assurance_criterion,
    mrsp2mpfd,
    odd_window_length,
    padding_line,
    peakdet,
    sdof_local,
    sign_from_correlation,
    tfristft,
    tfrstft,
    tfrstft_uniform,
)
from pysdkit.utils.bss._bss import BSS as ModuleBSS


def _tone(n_samples: int = 256, freq: float = 0.12) -> np.ndarray:
    time = np.arange(n_samples, dtype=float)
    return np.cos(2.0 * np.pi * freq * time)


def _example1_mixtures() -> np.ndarray:
    """MATLAB ``Example_1.m`` delay mixture (noise-free)."""
    time = np.arange(0.0, 10.0, 0.01)
    s1 = np.sin(2.0 * np.pi * 3.0 * time)
    s2 = np.sin(2.0 * np.pi * 6.0 * time)
    s3 = np.sin(2.0 * np.pi * 10.0 * time)
    s4 = np.sin(2.0 * np.pi * 15.0 * time)
    s5 = np.sin(2.0 * np.pi * 20.0 * time)

    def _delay(samples: np.ndarray, shift: int = 4) -> np.ndarray:
        return np.concatenate([samples[shift:], samples[:shift]])

    first = s1 + 0.8 * s2 + 0.5 * s3 + 0.3 * s4 + 0.7 * s5
    second = (
        0.5 * _delay(s1)
        + 0.8 * _delay(s2)
        + 0.95 * _delay(s3)
        + 1.1 * _delay(s4)
        + 0.15 * _delay(s5)
    )
    return np.vstack([first, second])


class WindowHelperTest(unittest.TestCase):
    """Odd Hamming, rounding, default length, frequency axis."""

    def test_odd_window_length(self) -> None:
        self.assertEqual(odd_window_length(40), 41)
        self.assertEqual(odd_window_length(41), 41)

    def test_odd_window_rejects_tiny(self) -> None:
        with self.assertRaises(ValueError):
            odd_window_length(2)

    def test_default_window_length(self) -> None:
        self.assertEqual(default_window_length(1000), 251)
        self.assertEqual(default_window_length(1280), 321)

    def test_matlab_round_ties_away(self) -> None:
        self.assertEqual(matlab_round(2.5), 3)
        self.assertEqual(matlab_round(-2.5), -3)
        self.assertEqual(matlab_round(2.4), 2)

    def test_hamming_odd_and_symmetric(self) -> None:
        window = hamming_window(41)
        self.assertEqual(window.size, 41)
        self.assertEqual(int(np.argmax(window)), 20)
        np.testing.assert_allclose(window, window[::-1], atol=1e-15)

    def test_hamming_rejects_even(self) -> None:
        with self.assertRaises(ValueError):
            hamming_window(40)

    def test_frequency_axis_stft(self) -> None:
        freq = frequency_axis_stft(400, fs=100.0)
        self.assertEqual(freq.size, 400)
        self.assertAlmostEqual(freq[0], 0.0)
        self.assertAlmostEqual(freq[1], 100.0 / 400.0)
        self.assertAlmostEqual(freq[200], -50.0)


class PeakdetTest(unittest.TestCase):
    """Billauer ``peakdet``."""

    def test_known_sequence(self) -> None:
        values = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
        maxtab, mintab = peakdet(values, 0.5)
        np.testing.assert_array_equal(maxtab[:, 0], [1.0, 3.0, 5.0])
        np.testing.assert_array_equal(maxtab[:, 1], [1.0, 2.0, 3.0])
        self.assertEqual(mintab.shape[1], 2)
        self.assertGreaterEqual(mintab.shape[0], 2)

    def test_optional_axis(self) -> None:
        values = np.array([0.0, 1.0, 0.0])
        axis = np.array([10.0, 20.0, 30.0])
        maxtab, _mintab = peakdet(values, 0.5, x=axis)
        self.assertAlmostEqual(float(maxtab[0, 0]), 20.0)

    def test_delta_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            peakdet(np.ones(8), 0.0)
        with self.assertRaises(ValueError):
            peakdet(np.ones(8), -1.0)

    def test_delta_must_be_scalar(self) -> None:
        with self.assertRaises(ValueError):
            peakdet(np.ones(8), [0.1, 0.2])

    def test_axis_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            peakdet(np.ones(8), 0.1, x=np.arange(3))


class StftTest(unittest.TestCase):
    """TFTB STFT / ISTFT and padding line."""

    def test_tfrstft_shape(self) -> None:
        samples = _tone(128)
        window = hamming_window(33)
        tfr = tfrstft(samples, window=window)
        self.assertEqual(tfr.shape, (128, 128))
        self.assertTrue(np.iscomplexobj(tfr))

    def test_tfrstft_uniform_matches_tfrstft(self) -> None:
        samples = _tone(64)
        window = hamming_window(17)
        left = tfrstft(samples, window=window)
        right = tfrstft_uniform(samples, window)
        np.testing.assert_allclose(left, right)

    def test_roundtrip_tone(self) -> None:
        samples = _tone(256)
        window = hamming_window(65)
        recovered = tfristft(tfrstft(samples, window=window), window=window)
        np.testing.assert_allclose(np.real(recovered), samples, atol=1e-12)

    def test_tfristft_requires_unit_grid(self) -> None:
        tfr = tfrstft(_tone(32), window=hamming_window(9))
        with self.assertRaises(ValueError):
            tfristft(tfr, times=np.arange(0.0, 16.0, 0.5), window=hamming_window(9))

    def test_tfristft_even_window_raises(self) -> None:
        tfr = np.ones((16, 16), dtype=complex)
        with self.assertRaises(ValueError):
            tfristft(tfr, window=np.ones(8))

    def test_tfristft_requires_window(self) -> None:
        tfr = np.ones((16, 16), dtype=complex)
        with self.assertRaises(ValueError):
            tfristft(tfr)

    def test_padding_line_interior_and_ends(self) -> None:
        window = hamming_window(65)
        pad = padding_line(256, window)
        self.assertEqual(pad.size, 256)
        np.testing.assert_allclose(pad[80:176], 1.0, atol=1e-12)
        self.assertGreater(float(pad[0]), 1.0)
        self.assertGreater(float(pad[-1]), 1.0)

    def test_frequency_energy_shapes(self) -> None:
        tfr = tfrstft(_tone(64), window=hamming_window(17))
        energy = frequency_energy(tfr)
        self.assertEqual(energy.shape, (64,))
        stack = np.stack([tfr, tfr], axis=0)
        self.assertEqual(frequency_energy(stack).shape, (2, 64))

    def test_frequency_energy_rejects_1d(self) -> None:
        with self.assertRaises(ValueError):
            frequency_energy(np.ones(8))


class CosineMaskTest(unittest.TestCase):
    """Cosine distance and energy masks."""

    def test_identical_vectors(self) -> None:
        vector = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(cosine_distance(vector, vector), 0.0, places=12)

    def test_orthogonal_vectors(self) -> None:
        self.assertAlmostEqual(
            cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            1.0,
            places=12,
        )

    def test_zero_vector_is_nan(self) -> None:
        self.assertTrue(np.isnan(cosine_distance(np.zeros(3), np.ones(3))))

    def test_cosine_masks_keeps_peak_bin(self) -> None:
        energy = np.array(
            [[1.0, 0.0, 2.0, 0.0], [2.0, 0.0, 4.0, 0.0]],
            dtype=float,
        )
        masks = cosine_masks(energy, np.array([0, 2]), e2=0.01, e3=0.1)
        self.assertEqual(masks.shape, (2, 4))
        self.assertEqual(float(masks[0, 0]), 1.0)
        self.assertEqual(float(masks[1, 2]), 1.0)

    def test_cosine_masks_empty_peaks(self) -> None:
        energy = np.ones((2, 8))
        masks = cosine_masks(energy, np.array([], dtype=int))
        self.assertEqual(masks.shape, (0, 8))


class ModalHelperTest(unittest.TestCase):
    """MAC, correlation signs, SDOF / mrsp2mpfd."""

    def test_mac_identical_columns(self) -> None:
        shapes = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        mac = modal_assurance_criterion(shapes, shapes)
        np.testing.assert_allclose(np.diag(mac), np.ones(2), atol=1e-12)
        self.assertLess(float(mac[0, 1]), 0.6)

    def test_mac_orthogonal(self) -> None:
        left = np.array([[1.0], [0.0]])
        right = np.array([[0.0], [1.0]])
        mac = modal_assurance_criterion(left, right)
        self.assertAlmostEqual(float(mac[0, 0]), 0.0, places=12)

    def test_mac_rejects_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            modal_assurance_criterion(np.ones((3, 2)), np.ones((2, 2)))

    def test_sign_from_correlation_flips(self) -> None:
        observations = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, -1.0, 1.0, -1.0]])
        sources = np.vstack([-observations[0], observations[1]])
        signs = sign_from_correlation(sources, observations)
        self.assertEqual(signs.shape, (2, 2))
        self.assertEqual(int(np.sign(signs[0, 0])), -1)

    def test_sdof_local_damped_cosine(self) -> None:
        fs = 1000.0
        fd = 40.0
        zeta = 0.02
        time = np.arange(2000, dtype=float) / fs
        wn = 2.0 * np.pi * fd / np.sqrt(1.0 - zeta**2)
        samples = np.exp(-zeta * wn * time) * np.cos(2.0 * np.pi * fd * time)
        spectrum = np.fft.fft(samples) / samples.size
        freq = np.arange(samples.size, dtype=float) * fs / samples.size
        _lam, _res, fd_hat, damping, n_used = sdof_local(spectrum, freq, n_points=7)
        self.assertEqual(n_used, 7)
        self.assertAlmostEqual(fd_hat, fd, delta=0.5)
        self.assertAlmostEqual(damping, 100.0 * zeta, delta=1.5)

    def test_sdof_local_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            sdof_local(np.ones(8), np.arange(4), 3)

    def test_mrsp2mpfd_sorts_by_frequency(self) -> None:
        fs = 500.0
        time = np.arange(1024, dtype=float) / fs
        low = np.exp(-0.5 * time) * np.cos(2.0 * np.pi * 12.0 * time)
        high = np.exp(-0.5 * time) * np.cos(2.0 * np.pi * 40.0 * time)
        result = mrsp2mpfd(np.vstack([high, low]), fs)
        self.assertLess(float(result["fd"][0]), float(result["fd"][1]))
        self.assertAlmostEqual(float(result["fd"][0]), 12.0, delta=1.0)
        self.assertAlmostEqual(float(result["fd"][1]), 40.0, delta=1.0)


class BSSPublicTest(unittest.TestCase):
    """Public ``BSS`` / ``bss`` API and Example 1."""

    def test_public_alias(self) -> None:
        self.assertIs(BSS, ModuleBSS)
        self.assertIs(UtilsBSS, BSS)

    def test_str(self) -> None:
        self.assertIn("BSS", str(BSS()))

    def test_rejects_one_channel(self) -> None:
        with self.assertRaises(ValueError):
            BSS().fit_transform(np.ones((1, 64)))

    def test_rejects_column_vectors(self) -> None:
        with self.assertRaises(ValueError):
            BSS().fit_transform(np.ones((64, 2)))

    def test_rejects_channel_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            BSS(channel=2).fit_transform(np.ones((2, 64)))

    def test_rejects_even_window(self) -> None:
        with self.assertRaises(ValueError):
            BSS(window_length=64)

    def test_rejects_tiny_window(self) -> None:
        with self.assertRaises(ValueError):
            BSS(window_length=1)

    def test_negative_channel(self) -> None:
        with self.assertRaises(ValueError):
            BSS(channel=-1)

    def test_example1_five_tones(self) -> None:
        mixtures = _example1_mixtures()
        engine = BSS()
        sources = engine(mixtures)
        self.assertEqual(sources.shape, (5, 1000))
        self.assertEqual(engine.mixing_.shape, (2, 5))
        np.testing.assert_allclose(
            np.linalg.norm(engine.mixing_, axis=0), np.ones(5), atol=1e-12
        )
        freq = np.fft.rfftfreq(sources.shape[1], d=0.01)
        found = [
            float(freq[int(np.argmax(np.abs(np.fft.rfft(row))))]) for row in sources
        ]
        self.assertEqual(sorted(found), [3.0, 6.0, 10.0, 15.0, 20.0])
        self.assertEqual(engine.energy_.shape[0], 2)
        self.assertEqual(engine.masks_.shape, (5, 1000))
        self.assertEqual(engine.pad_.size, 1000)
        self.assertEqual(engine.peaks_.size, 5)

    def test_bss_matches_class(self) -> None:
        mixtures = _example1_mixtures()
        engine = BSS()
        sources_cls = engine.fit_transform(mixtures)
        sources_fn, mixing = bss(mixtures)
        np.testing.assert_allclose(sources_fn, sources_cls)
        np.testing.assert_allclose(mixing, engine.mixing_)


class BSSPackagedDataTest(unittest.TestCase):
    """Paper experiments 1 and 2 (packaged ``.npy``)."""

    def test_beam_modes_and_mac(self) -> None:
        record = load_bss_beam()
        mixtures = record["signal"]
        self.assertEqual(mixtures.shape, (3, 1280))
        engine = BSS(window_length=321, e1=0.01)
        sources = engine.fit_transform(mixtures)
        self.assertGreaterEqual(sources.shape[0], 4)
        self.assertEqual(sources.shape[1], 1280)
        signed = engine.mixing_ * sign_from_correlation(sources, mixtures)
        n_modes = min(signed.shape[1], record["mode_shape"].shape[1])
        mac = modal_assurance_criterion(
            signed[:, :n_modes], record["mode_shape"][:, :n_modes]
        )
        self.assertTrue(np.all(np.diag(mac) > 0.9))
        params = mrsp2mpfd(sources[:n_modes], record["fs"])
        expected = np.array([30.78, 215.6, 585.9, 1115.3])
        np.testing.assert_allclose(params["fd"][:4], expected, rtol=0.02, atol=2.0)

    def test_yk9_separates_five_sources(self) -> None:
        record = load_bss_yk9()
        mixtures = record["signal"]
        self.assertEqual(mixtures.shape[0], 3)
        engine = BSS(window_length=351, e1=1.0e-4)
        sources = engine.fit_transform(mixtures)
        self.assertGreaterEqual(sources.shape[0], 4)
        signed = engine.mixing_ * sign_from_correlation(sources, mixtures)
        n_modes = min(4, signed.shape[1], record["mode_shape"].shape[1])
        mac = modal_assurance_criterion(
            signed[:, :n_modes], record["mode_shape"][:, :n_modes]
        )
        self.assertTrue(np.all(np.diag(mac) > 0.9))


if __name__ == "__main__":
    unittest.main()
