# -*- coding: utf-8 -*-
"""
Automated tests for FAEMD2D.
"""

import unittest

import numpy as np

from pysdkit import FAEMD2D


def _synthetic_image(n: int = 64, seed: int = 0) -> np.ndarray:
    """Smooth trend + two oscillatory spatial modes."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    xx = xx / n
    yy = yy / n
    trend = 0.3 * xx + 0.2 * yy
    mode1 = 0.8 * np.sin(2 * np.pi * 3 * xx) * np.cos(2 * np.pi * 2 * yy)
    mode2 = 0.4 * np.sin(2 * np.pi * 8 * xx + 2 * np.pi * 6 * yy)
    noise = 0.02 * rng.standard_normal((n, n))
    return trend + mode1 + mode2 + noise


class FAEMD2DTest(unittest.TestCase):
    """Automated tests for FAEMD2D."""

    def test_fit_transform_shape(self) -> None:
        img = _synthetic_image(48)
        imfs = FAEMD2D(max_imfs=3, tol=0.05).fit_transform(img)
        self.assertEqual(imfs.shape, (3, 48, 48))
        self.assertTrue(np.all(np.isfinite(imfs)))

    def test_reconstruction(self) -> None:
        img = _synthetic_image(48)
        imfs = FAEMD2D(max_imfs=3, tol=0.05).fit_transform(img)
        self.assertTrue(np.allclose(imfs.sum(axis=0), img, atol=1e-6))

    def test_default_call(self) -> None:
        img = _synthetic_image(40)
        decomp = FAEMD2D(max_imfs=2, tol=0.08)
        self.assertTrue(np.allclose(decomp(img), decomp.fit_transform(img)))

    def test_multichannel(self) -> None:
        a = _synthetic_image(40, seed=0)
        b = _synthetic_image(40, seed=1)
        data = np.stack([a, b], axis=0)
        imfs = FAEMD2D(max_imfs=3, tol=0.08).fit_transform(data)
        self.assertEqual(imfs.shape, (3, 2, 40, 40))
        self.assertTrue(np.allclose(imfs.sum(0), data, atol=1e-6))

    def test_str(self) -> None:
        self.assertIn("FAEMD2D", str(FAEMD2D()))

    def test_invalid_window(self) -> None:
        with self.assertRaises(ValueError):
            FAEMD2D(window_type=7)

    def test_residue_last(self) -> None:
        """Last mode should be smoother than the first IMF."""
        img = _synthetic_image(48)
        imfs = FAEMD2D(max_imfs=3, tol=0.05).fit_transform(img)
        high = np.std(np.diff(imfs[0], axis=0))
        low = np.std(np.diff(imfs[-1], axis=0))
        self.assertLess(low, high)


if __name__ == "__main__":
    unittest.main()
