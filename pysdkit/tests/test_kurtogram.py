# -*- coding: utf-8 -*-
"""
Unit tests for the Fast Kurtogram (Antoni, MSSP 2007).
"""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from pysdkit.data import load_fast_kurtogram_x
from pysdkit.utils import (
    analytic_filters,
    dbfb,
    fast_kurtogram,
    find_wav_kurt,
    kurt,
    max_ij,
    plot_kurtogram,
    prewhiten_ar,
    tbfb,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


class KurtosisHelperTest(unittest.TestCase):
    """MATLAB ``kurt`` / ``max_IJ`` helpers."""

    def test_kurt2_gaussian_near_zero(self) -> None:
        samples = _rng(1).normal(size=40_000)
        self.assertAlmostEqual(kurt(samples, "kurt2"), 0.0, delta=0.08)

    def test_kurt2_complex_circular_gaussian(self) -> None:
        rng = _rng(2)
        samples = rng.normal(size=30_000) + 1j * rng.normal(size=30_000)
        self.assertAlmostEqual(kurt(samples, "kurt2"), 0.0, delta=0.08)

    def test_kurt2_zeros(self) -> None:
        self.assertEqual(kurt(np.zeros(32), "kurt2"), 0.0)

    def test_kurt1_zeros(self) -> None:
        self.assertEqual(kurt(np.zeros(8), "kurt1"), 0.0)

    def test_kurt_invalid_opt(self) -> None:
        with self.assertRaises(ValueError):
            kurt(np.ones(8), opt="skew")

    def test_max_ij_column_then_row(self) -> None:
        matrix = np.array([[1.0, 3.0], [4.0, 2.0]])
        row, col, value = max_ij(matrix)
        self.assertEqual((row, col), (1, 0))
        self.assertEqual(value, 4.0)

    def test_max_ij_empty(self) -> None:
        with self.assertRaises(ValueError):
            max_ij(np.array([]))


class FilterBankTest(unittest.TestCase):
    """Decimation lengths of the binary / ternary banks."""

    def setUp(self) -> None:
        self.h, self.g, self.h1, self.h2, self.h3 = analytic_filters()

    def test_analytic_filter_lengths(self) -> None:
        self.assertEqual(self.h.size, 17)
        self.assertEqual(self.g.size, 16)
        self.assertEqual(self.h1.size, 25)
        self.assertEqual(self.h2.size, 25)
        self.assertEqual(self.h3.size, 25)

    def test_dbfb_even_length(self) -> None:
        approx, detail = dbfb(np.arange(20.0), self.h, self.g)
        self.assertEqual(approx.size, 10)
        self.assertEqual(detail.size, 10)

    def test_dbfb_odd_length(self) -> None:
        approx, detail = dbfb(np.arange(21.0), self.h, self.g)
        self.assertEqual(approx.size, 10)
        self.assertEqual(detail.size, 10)

    def test_tbfb_stride_three(self) -> None:
        a1, a2, a3 = tbfb(np.arange(30.0), self.h1, self.h2, self.h3)
        self.assertEqual(a1.size, 10)
        self.assertEqual(a2.size, 10)
        self.assertEqual(a3.size, 10)


class FastKurtogramTest(unittest.TestCase):
    """Public ``fast_kurtogram`` map, peak, and filtering."""

    def test_shape_and_nonnegative(self) -> None:
        nlevel = 3
        samples = _rng(3).normal(size=2048)
        k_wav, info = fast_kurtogram(samples, nlevel, fs=1.0)
        self.assertEqual(k_wav.shape, (2 * nlevel, 3 * 2**nlevel))
        self.assertTrue(np.all(k_wav >= 0.0))
        self.assertEqual(info["nlevel"], nlevel)
        self.assertEqual(np.asarray(info["freq_w"]).size, k_wav.shape[1])
        self.assertEqual(np.asarray(info["level_w"]).size, k_wav.shape[0])

    def test_nlevel_too_large_raises(self) -> None:
        samples = _rng(4).normal(size=2048)
        with self.assertRaises(ValueError):
            fast_kurtogram(samples, nlevel=5, fs=1.0)

    def test_nlevel_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            fast_kurtogram(np.ones(2048), nlevel=0)

    def test_demo_peak_in_theoretical_band(self) -> None:
        record = load_fast_kurtogram_x()
        k_wav, info = fast_kurtogram(record["signal"], nlevel=7, fs=1.0)
        self.assertEqual(k_wav.shape, (14, 384))
        self.assertGreater(float(info["Kmax"]), 0.0)
        self.assertGreaterEqual(float(info["fc"]), 0.15)
        self.assertLessEqual(float(info["fc"]), 0.19)

    def test_find_wav_kurt_envelope(self) -> None:
        samples = _rng(5).normal(size=2048)
        _, info = fast_kurtogram(samples, nlevel=3, fs=1.0)
        fr = float(info["fc"]) / float(info["fs"])
        out = find_wav_kurt(
            np.asarray(info["x"]),
            float(info["level"]),
            fr,
            fs=1.0,
            filters=info["filters"],
        )
        envelope = np.asarray(out["c"])
        self.assertTrue(np.iscomplexobj(envelope) or np.isrealobj(envelope))
        self.assertTrue(np.all(np.isfinite(envelope)))
        self.assertLess(envelope.size, samples.size)
        self.assertGreater(envelope.size, 0)

    def test_prewhiten_shortens_by_order(self) -> None:
        samples = _rng(6).normal(size=500)
        order = 20
        whitened = prewhiten_ar(samples, order=order)
        self.assertEqual(whitened.size, samples.size - order)
        self.assertTrue(np.all(np.isfinite(whitened)))

    def test_prewhiten_rejects_short_record(self) -> None:
        with self.assertRaises(ValueError):
            prewhiten_ar(np.ones(8), order=10)

    def test_plot_kurtogram_on_axes(self) -> None:
        samples = _rng(7).normal(size=2048)
        k_wav, info = fast_kurtogram(samples, nlevel=3, fs=1.0)
        fig, axes = plt.subplots()
        returned = plot_kurtogram(k_wav, info, ax=axes)
        self.assertIs(returned, axes)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
