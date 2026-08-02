# -*- coding: utf-8 -*-
"""
Unit tests for Bidimensional Multivariate Empirical Mode Decomposition (BMEMD).
"""

import unittest

import numpy as np

from pysdkit import BMEMD
from pysdkit._emd2d.bmemd import local_var_img, fuse_images


def _texture_pair(n: int = 32, seed: int = 0) -> np.ndarray:
    """Two-channel synthetic texture stack shaped ``(2, n, n)``."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    xx = xx / float(n)
    yy = yy / float(n)
    ch0 = 0.6 * np.sin(2 * np.pi * 4 * xx) + 0.3 * np.sin(2 * np.pi * 2 * yy)
    ch1 = 0.5 * np.cos(2 * np.pi * 3 * xx + 0.4) + 0.35 * np.sin(
        2 * np.pi * 5 * yy
    )
    ch0 = ch0 + 0.05 * rng.standard_normal((n, n))
    ch1 = ch1 + 0.05 * rng.standard_normal((n, n))
    return np.stack([ch0, ch1], axis=0)


class BMEMDTest(unittest.TestCase):
    """Automated tests for BMEMD."""

    def test_fit_transform_shape(self) -> None:
        images = _texture_pair(28)
        imfs = BMEMD(n_dir=6, max_imfs=2, max_sift=8).fit_transform(images)
        self.assertEqual(imfs.ndim, 4)
        self.assertEqual(imfs.shape[1:], images.shape)
        self.assertGreaterEqual(imfs.shape[0], 2)  # ≥1 BIMF + residue
        self.assertTrue(np.all(np.isfinite(imfs)))

    def test_reconstruction(self) -> None:
        images = _texture_pair(28)
        imfs = BMEMD(n_dir=6, max_imfs=2, max_sift=8).fit_transform(images)
        recon = imfs.sum(axis=0)
        self.assertTrue(np.allclose(recon, images, atol=1e-5))

    def test_default_call(self) -> None:
        images = _texture_pair(24)
        decomp = BMEMD(n_dir=6, max_imfs=2, max_sift=6)
        a = decomp(images)
        b = decomp.fit_transform(images)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.allclose(a, b))

    def test_str(self) -> None:
        self.assertIn("BMEMD", str(BMEMD()))

    def test_direction_bivariate(self) -> None:
        dirs = BMEMD(n_dir=8)._direction_vectors(2)
        self.assertEqual(dirs.shape, (8, 2))
        norms = np.linalg.norm(dirs, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-10))

    def test_direction_trivariate(self) -> None:
        dirs = BMEMD(n_dir=8)._direction_vectors(3)
        self.assertEqual(dirs.shape, (8, 3))
        norms = np.linalg.norm(dirs, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-8))

    def test_invalid_channels(self) -> None:
        with self.assertRaises(ValueError):
            BMEMD().fit_transform(np.ones((1, 16, 16)))
        with self.assertRaises(ValueError):
            BMEMD(n_dir=4)

    def test_local_var_img(self) -> None:
        img = np.arange(25, dtype=float).reshape(5, 5)
        var = local_var_img(img, window=3)
        self.assertEqual(var.shape, img.shape)
        self.assertTrue(np.all(var >= 0))

    def test_fuse_shape(self) -> None:
        images = _texture_pair(24)
        bmemd = BMEMD(n_dir=6, max_imfs=2, max_sift=6)
        imfs = bmemd.fit_transform(images)
        fused = bmemd.fuse(images, imfs=imfs, var_window=5)
        self.assertEqual(fused.shape, images.shape[1:])
        self.assertTrue(np.all(np.isfinite(fused)))

    def test_fuse_images_helper(self) -> None:
        images = _texture_pair(20)
        fused, imfs = fuse_images(images, n_dir=6, max_imfs=2, max_sift=5)
        self.assertEqual(fused.shape, (20, 20))
        self.assertEqual(imfs.shape[1:], images.shape)

    def test_fix_h_runs(self) -> None:
        images = _texture_pair(20)
        imfs = BMEMD(
            n_dir=6, max_imfs=2, stop_crit="fix_h", stop_cnt=2, max_sift=10
        ).fit_transform(images)
        self.assertTrue(np.allclose(imfs.sum(0), images, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
