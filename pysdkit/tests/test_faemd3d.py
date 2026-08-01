# -*- coding: utf-8 -*-
"""Unit tests for Tridimensional FAEMD (FAEMD3D)."""

import unittest

import numpy as np

from pysdkit import FAEMD3D


def _synthetic_volume(n: int = 24, seed: int = 0) -> np.ndarray:
    """Compact 3-D field with a trend and one oscillatory mode."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    xx = xx / n
    yy = yy / n
    zz = zz / n
    trend = 0.2 * xx + 0.1 * yy + 0.15 * zz
    osc = 0.6 * np.sin(2 * np.pi * 2 * xx) * np.cos(2 * np.pi * 2 * yy)
    noise = 0.02 * rng.standard_normal((n, n, n))
    return trend + osc + noise


class FAEMD3DTest(unittest.TestCase):
    """Automated tests for FAEMD3D."""

    def test_fit_transform_shape(self) -> None:
        vol = _synthetic_volume(20)
        imfs = FAEMD3D(max_imfs=2, tol=0.1).fit_transform(vol)
        self.assertEqual(imfs.shape, (2, 20, 20, 20))
        self.assertTrue(np.all(np.isfinite(imfs)))

    def test_reconstruction(self) -> None:
        vol = _synthetic_volume(20)
        imfs = FAEMD3D(max_imfs=2, tol=0.1).fit_transform(vol)
        self.assertTrue(np.allclose(imfs.sum(axis=0), vol, atol=1e-6))

    def test_default_call(self) -> None:
        vol = _synthetic_volume(16)
        decomp = FAEMD3D(max_imfs=2, tol=0.15)
        self.assertTrue(np.allclose(decomp(vol), decomp.fit_transform(vol)))

    def test_multichannel(self) -> None:
        a = _synthetic_volume(16, seed=0)
        b = _synthetic_volume(16, seed=1)
        data = np.stack([a, b], axis=0)
        imfs = FAEMD3D(max_imfs=2, tol=0.15).fit_transform(data)
        self.assertEqual(imfs.shape, (2, 2, 16, 16, 16))
        self.assertTrue(np.allclose(imfs.sum(0), data, atol=1e-6))

    def test_str(self) -> None:
        self.assertIn("FAEMD3D", str(FAEMD3D()))

    def test_too_small(self) -> None:
        with self.assertRaises(ValueError):
            FAEMD3D().fit_transform(np.ones((2, 2, 2)))


if __name__ == "__main__":
    unittest.main()
