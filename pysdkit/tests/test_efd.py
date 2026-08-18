# -*- coding: utf-8 -*-
"""
Automated tests for Empirical Fourier Decomposition (EFD).

Covers every public function and method in ``pysdkit._ewt.efd``, including
helpers ported from MATLAB ``EFD.m``, ``Segm_tec.m``, ``plotbounds.m`` and
``IFIA.m``.
"""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysdkit import EFD, efd
from pysdkit._ewt.efd import (
    apply_ideal_bandpass,
    copy_matlab_range,
    ifia,
    matlab_half_length,
    mirror_extend,
    plot_bounds,
    segm_tec,
)
from pysdkit.utils import fft


def _paper_example1(fs: float = 1000.0) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Zhou et al. (2022) Eq. (22): linear trend + 4 Hz + 20 Hz harmonics."""
    t = np.arange(0.0, 1.0 + 1.0 / fs, 1.0 / fs)
    f11 = 6.0 * t
    f12 = 2.0 * np.cos(8.0 * np.pi * t)
    f13 = np.cos(40.0 * np.pi * t)
    return t, f11 + f12 + f13, (f11, f12, f13)


class MatlabHalfLengthTest(unittest.TestCase):
    def test_even_and_odd(self) -> None:
        self.assertEqual(matlab_half_length(10), 5)
        self.assertEqual(matlab_half_length(11), 6)
        self.assertEqual(matlab_half_length(9), 5)
        self.assertEqual(matlab_half_length(1), 1)

    def test_matches_matlab_round_half_away(self) -> None:
        for n in range(1, 40):
            matlab = int(np.fix(n / 2.0 + 0.5))
            self.assertEqual(matlab_half_length(n), matlab)

    def test_invalid(self) -> None:
        with self.assertRaises(ValueError):
            matlab_half_length(0)


class CopyMatlabRangeTest(unittest.TestCase):
    def test_inclusive_1based_slice(self) -> None:
        src = np.arange(10, dtype=float)
        dst = np.zeros(10)
        copy_matlab_range(dst, src, 3, 6)
        expected = np.zeros(10)
        expected[2:6] = src[2:6]
        np.testing.assert_array_equal(dst, expected)

    def test_empty_when_a_gt_b(self) -> None:
        src = np.arange(8, dtype=float)
        dst = np.ones(8)
        copy_matlab_range(dst, src, 5, 2)
        np.testing.assert_array_equal(dst, np.ones(8))

    def test_full_vector(self) -> None:
        src = np.arange(7, dtype=float)
        dst = np.zeros(7)
        copy_matlab_range(dst, src, 1, 7)
        np.testing.assert_array_equal(dst, src)


class MirrorExtendTest(unittest.TestCase):
    def test_even_length(self) -> None:
        x = np.arange(10, dtype=float)
        ext, l = mirror_extend(x)
        self.assertEqual(l, 5)
        np.testing.assert_array_equal(ext[:4], x[:4][::-1])
        np.testing.assert_array_equal(ext[4:14], x)
        np.testing.assert_array_equal(ext[14:], x[-5:][::-1])
        cropped = ext[l - 1 : ext.size - l]
        np.testing.assert_array_equal(cropped, x)

    def test_odd_length_keeps_all_samples(self) -> None:
        x = np.arange(11, dtype=float)
        ext, l = mirror_extend(x)
        self.assertEqual(l, 6)
        self.assertEqual(ext.size, 11 + 5 + 6)
        cropped = ext[l - 1 : ext.size - l]
        np.testing.assert_array_equal(cropped, x)


class ApplyIdealBandpassTest(unittest.TestCase):
    def test_dc_band_copies_hermitian_tail(self) -> None:
        rng = np.random.default_rng(0)
        ff = rng.normal(size=20) + 1j * rng.normal(size=20)
        bound_right = 4
        band = apply_ideal_bandpass(ff, 0, bound_right)
        expected = np.zeros_like(ff)
        expected[0:bound_right] = ff[0:bound_right]
        expected[20 + 1 - bound_right : 20] = ff[20 + 1 - bound_right : 20]
        np.testing.assert_array_equal(band, expected)

    def test_interior_band_uses_1based_inclusive_ends(self) -> None:
        ff = np.arange(16, dtype=float) + 1j * np.arange(16, dtype=float)
        left, right = 3, 6
        band = apply_ideal_bandpass(ff, left, right)
        self.assertTrue(np.all(band[left - 1 : right] == ff[left - 1 : right]))
        self.assertTrue(np.all(band[: left - 1] == 0))
        self.assertTrue(np.all(band[right:8] == 0))


class SegmTecTest(unittest.TestCase):
    def _two_tone_spectrum(self, n: int = 1000, fs: float = 1000.0) -> np.ndarray:
        t = np.arange(n) / fs
        x = np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(2 * np.pi * 50 * t)
        spec = np.abs(fft(x))
        return spec[: matlab_half_length(spec.size)]

    def test_returns_n_plus_one_bounds_and_n_cerf(self) -> None:
        f = self._two_tone_spectrum()
        bounds, cerf = segm_tec(f, n_segments=3)
        self.assertEqual(bounds.size, 4)
        self.assertEqual(cerf.size, 3)
        self.assertTrue(np.all(cerf >= 0.0))
        self.assertTrue(np.all(cerf <= np.pi + 1e-12))

    def test_bounds_are_nondecreasing(self) -> None:
        f = self._two_tone_spectrum()
        bounds, _ = segm_tec(f, n_segments=3)
        self.assertTrue(np.all(np.diff(bounds) >= -1e-12))

    def test_keeps_n_highest_maxima_on_synthetic_peaks(self) -> None:
        f = np.zeros(21)
        f[2] = 5.0
        f[8] = 9.0
        f[14] = 3.0
        f[0] = 0.1
        f[-1] = 0.2
        bounds, cerf = segm_tec(f, n_segments=3)
        self.assertEqual(bounds.size, 4)
        self.assertEqual(cerf.size, 3)
        expected_cerf = np.array([3, 9, 15], dtype=float) * np.pi / 21.0
        np.testing.assert_allclose(np.sort(cerf), np.sort(expected_cerf))

    def test_interior_min_is_inclusive(self) -> None:
        f = np.array([1.0, 3.0, 0.2, 4.0, 1.0], dtype=float)
        bounds, _ = segm_tec(f, n_segments=2)
        self.assertGreaterEqual(bounds.size, 2)

    def test_invalid_n_and_short_spectrum(self) -> None:
        with self.assertRaises(ValueError):
            segm_tec(np.ones(10), n_segments=0)
        with self.assertRaises(ValueError):
            segm_tec(np.ones(2), n_segments=3)


class IfiaTest(unittest.TestCase):
    def test_pure_tone_frequency(self) -> None:
        fs = 200.0
        t = np.arange(400) / fs
        x = np.cos(2.0 * np.pi * 10.0 * t)
        freq, amp = ifia(x, fs)
        self.assertEqual(freq.shape, x.shape)
        self.assertEqual(amp.shape, x.shape)
        mid = slice(50, -50)
        np.testing.assert_allclose(np.mean(freq[mid]), 10.0, atol=0.3)
        np.testing.assert_allclose(np.mean(amp[mid]), 1.0, atol=0.1)

    def test_2d_modes(self) -> None:
        fs = 100.0
        t = np.arange(200) / fs
        modes = np.vstack([np.cos(2 * np.pi * 5 * t), np.sin(2 * np.pi * 12 * t)])
        freq, amp = ifia(modes, fs)
        self.assertEqual(freq.shape, modes.shape)
        self.assertEqual(amp.shape, modes.shape)

    def test_short_signal(self) -> None:
        freq, amp = ifia(np.array([1.0, -1.0]), fs=10.0)
        self.assertEqual(freq.size, 2)
        self.assertEqual(amp.size, 2)

    def test_rejects_3d(self) -> None:
        with self.assertRaises(ValueError):
            ifia(np.ones((2, 2, 2)), fs=1.0)


class PlotBoundsTest(unittest.TestCase):
    def test_returns_axes(self) -> None:
        t = np.linspace(0, 1, 256, endpoint=False)
        x = np.cos(2 * np.pi * 8 * t)
        ax = plot_bounds(x, np.array([0.5, 1.5]))
        self.assertIsNotNone(ax)
        plt.close("all")


class EFDClassTest(unittest.TestCase):
    def test_str_init_and_call(self) -> None:
        decomp = EFD(max_imfs=3)
        self.assertEqual(str(decomp), "Empirical Fourier Decomposition (EFD)")
        self.assertEqual(decomp.max_imfs, 3)
        _, x, _ = _paper_example1()
        imfs = decomp(x)
        self.assertEqual(imfs.ndim, 2)
        self.assertEqual(imfs.shape[1], x.size)
        self.assertEqual(imfs.shape[0], 3)

    def test_call_equals_fit_transform(self) -> None:
        _, x, _ = _paper_example1()
        decomp = EFD(max_imfs=3)
        np.testing.assert_allclose(decomp(x), decomp.fit_transform(x))

    def test_invalid_max_imfs(self) -> None:
        with self.assertRaises(ValueError):
            EFD(max_imfs=0)

    def test_short_signal(self) -> None:
        with self.assertRaises(ValueError):
            EFD(max_imfs=2).fit_transform(np.array([1.0, 2.0]))

    def test_odd_length_is_preserved(self) -> None:
        t = np.linspace(0, 1, 501)
        x = np.cos(2 * np.pi * 7 * t) + 0.4 * np.cos(2 * np.pi * 30 * t)
        imfs = EFD(max_imfs=3).fit_transform(x)
        self.assertEqual(imfs.shape[1], x.size)

    def test_return_all_tuple(self) -> None:
        _, x, _ = _paper_example1()
        imfs, cerf, bounds = EFD(max_imfs=3).fit_transform(x, return_all=True)
        self.assertEqual(imfs.shape[1], x.size)
        self.assertEqual(cerf.size, imfs.shape[0])
        self.assertEqual(bounds.size, imfs.shape[0] + 1)
        self.assertTrue(np.all(bounds >= -1e-12))
        self.assertTrue(np.all(bounds <= np.pi + 1e-6))
        self.assertTrue(np.all(cerf >= 0.0))
        self.assertTrue(np.all(cerf <= np.pi + 1e-12))

    def test_paper_example1_separates_harmonics(self) -> None:
        t, x, (f11, f12, f13) = _paper_example1()
        imfs = EFD(max_imfs=3).fit_transform(x)
        corrs = [float(np.corrcoef(imfs[k], f13)[0, 1]) for k in range(imfs.shape[0])]
        self.assertGreater(max(np.abs(corrs)), 0.85)
        trend_corrs = [
            float(np.corrcoef(imfs[k], f11)[0, 1]) for k in range(imfs.shape[0])
        ]
        self.assertGreater(max(np.abs(trend_corrs)), 0.85)

    def test_functional_interface(self) -> None:
        _, x, _ = _paper_example1()
        a = efd(x, max_imfs=3)
        b = EFD(max_imfs=3).fit_transform(x)
        np.testing.assert_allclose(a, b)
        imfs, cerf, bounds = efd(x, max_imfs=3, return_all=True)
        self.assertEqual(imfs.shape, a.shape)
        self.assertEqual(cerf.size, 3)
        self.assertEqual(bounds.size, 4)

    def test_package_export(self) -> None:
        _, x, _ = _paper_example1()
        imfs = EFD(max_imfs=3)(x)
        self.assertEqual(imfs.shape[1], x.size)

    def test_row_vector_input(self) -> None:
        _, x, _ = _paper_example1()
        imfs = EFD(max_imfs=3).fit_transform(x.reshape(1, -1))
        self.assertEqual(imfs.shape[1], x.size)


if __name__ == "__main__":
    unittest.main()
