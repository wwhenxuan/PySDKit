# -*- coding: utf-8 -*-
"""
Unit tests for Adaptive Local Iterative Filtering (ALIF).
"""
import unittest
import numpy as np

from pysdkit import ALIF
from pysdkit._alif import IterativeFiltering
from pysdkit._alif._helpers import get_mask_v1, load_prefixed_filter, maxmins


def _make_chirp_mixture(n: int = 512):
    """Two chirps + constant, similar to ALIF Example.m (shortened)."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    d = 8.0
    t_period = 2.0 * np.pi
    x = np.cos(-0.5 * d / t_period * t**2 - 4.0 * t)
    y = np.cos(-0.5 * d / t_period * t**2 - 20.0 * t)
    signal = x + y + 1.0
    return t, signal, x, y


class HelpersTest(unittest.TestCase):
    """Tests for ALIF helper routines."""

    def test_load_filter(self) -> None:
        mm = load_prefixed_filter()
        self.assertEqual(mm.ndim, 1)
        self.assertGreater(mm.size, 1000)
        self.assertAlmostEqual(float(mm.sum()), 1.0, places=6)

    def test_get_mask_integer(self) -> None:
        mm = load_prefixed_filter()
        a = get_mask_v1(mm, 10)
        self.assertEqual(a.size, 21)
        self.assertTrue(np.all(np.isfinite(a)))

    def test_maxmins_sine(self) -> None:
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        y = np.sin(4 * t)
        extrema = maxmins(y, extension_type="p")
        self.assertGreaterEqual(extrema.size, 6)


class IterativeFilteringTest(unittest.TestCase):
    """Tests for the IF subroutine."""

    def test_fit_transform_shapes(self) -> None:
        _, signal, _, _ = _make_chirp_mixture(256)
        iff = IterativeFiltering(max_imfs=2, delta=1e-2, max_inner=50, verbose=0)
        imfs = iff.fit_transform(signal)
        self.assertEqual(imfs.ndim, 2)
        self.assertEqual(imfs.shape[1], signal.size)
        self.assertGreaterEqual(imfs.shape[0], 1)

    def test_reconstruction(self) -> None:
        _, signal, _, _ = _make_chirp_mixture(256)
        iff = IterativeFiltering(max_imfs=3, delta=1e-2, max_inner=40, verbose=0)
        imfs = iff.fit_transform(signal)
        recon = imfs.sum(axis=0)
        self.assertTrue(np.allclose(signal, recon, atol=1e-8))


class ALIFTest(unittest.TestCase):
    """Automated tests for ALIF."""

    def test_fit_transform_shapes(self) -> None:
        _, signal, _, _ = _make_chirp_mixture(256)
        alif = ALIF(max_imfs=2, delta=1e-3, xi=1.6, max_inner=40, verbose=0)
        imfs = alif.fit_transform(signal)
        self.assertEqual(imfs.ndim, 2)
        self.assertEqual(imfs.shape[1], signal.size)
        self.assertGreaterEqual(imfs.shape[0], 1)

    def test_default_call(self) -> None:
        _, signal, _, _ = _make_chirp_mixture(256)
        alif = ALIF(max_imfs=2, delta=1e-3, max_inner=30, verbose=0)
        out = alif(signal)
        self.assertEqual(out.shape, alif.fit_transform(signal).shape)

    def test_reconstruction(self) -> None:
        _, signal, _, _ = _make_chirp_mixture(256)
        alif = ALIF(max_imfs=2, delta=1e-3, max_inner=40, verbose=0)
        imfs = alif.fit_transform(signal)
        recon = imfs.sum(axis=0)
        self.assertTrue(np.allclose(signal, recon, atol=1e-8))

    def test_return_masks(self) -> None:
        _, signal, _, _ = _make_chirp_mixture(256)
        alif = ALIF(max_imfs=1, delta=1e-3, max_inner=30, verbose=0)
        imfs, masks = alif.fit_transform(signal, return_masks=True)
        self.assertEqual(imfs.shape[1], signal.size)
        if masks.size:
            self.assertEqual(masks.shape[1], signal.size)

    def test_str(self) -> None:
        self.assertIn("ALIF", str(ALIF()))


if __name__ == "__main__":
    unittest.main()
