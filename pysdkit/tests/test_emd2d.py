# -*- coding: utf-8 -*-
"""
Automated tests for Empirical Mode Decomposition 2D (EMD2D).
"""

import unittest

import numpy as np

from pysdkit import EMD2D
from pysdkit._emd2d.emd2d import EMD2D as ModuleEMD2D


def _synthetic_image(n: int = 32) -> np.ndarray:
    """Smooth trend plus two spatially oscillatory modes."""
    yy, xx = np.mgrid[0:n, 0:n]
    xx = xx / float(n)
    yy = yy / float(n)
    trend = 0.3 * xx + 0.2 * yy
    mode1 = 0.8 * np.sin(2.0 * np.pi * 3.0 * xx) * np.cos(2.0 * np.pi * 2.0 * yy)
    mode2 = 0.4 * np.sin(2.0 * np.pi * 8.0 * xx + 2.0 * np.pi * 6.0 * yy)
    return trend + mode1 + mode2


class EMD2DTest(unittest.TestCase):
    """Tests for :class:`pysdkit.EMD2D`."""

    def setUp(self) -> None:
        self.image = _synthetic_image(32)
        self.emd = EMD2D(max_imfs=3, max_iter=30)

    def test_str(self) -> None:
        """``str(EMD2D)`` reports the algorithm name."""
        self.assertEqual(str(self.emd), "Empirical Mode Decomposition 2D (EMD2D)")

    def test_init_stores_parameters(self) -> None:
        """Constructor stores sifting limits and proto-IMF thresholds."""
        self.assertEqual(self.emd.max_imfs, 3)
        self.assertEqual(self.emd.mse_thr, 0.01)
        self.assertEqual(self.emd.mean_thr, 0.01)
        self.assertEqual(self.emd.max_iter, 30)
        self.assertEqual(self.emd.fix_epochs, 0)
        self.assertEqual(self.emd.fix_epochs_h, 0)

    def test_import_from_package_root(self) -> None:
        """EMD2D is exported from the package root and the ``_emd2d`` module."""
        self.assertIs(EMD2D, ModuleEMD2D)

    def test_fit_transform_shape(self) -> None:
        """``fit_transform`` returns ``(K, H, W)`` modes matching the image."""
        imfs = self.emd.fit_transform(self.image)
        self.assertEqual(imfs.ndim, 3)
        self.assertEqual(imfs.shape[1:], self.image.shape)
        self.assertGreaterEqual(imfs.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(imfs)))

    def test_reconstruction(self) -> None:
        """Summing the 2-D IMFs (including residue) recovers the input image."""
        imfs = self.emd.fit_transform(self.image)
        np.testing.assert_allclose(imfs.sum(axis=0), self.image, atol=1e-10)

    def test_call_matches_fit_transform(self) -> None:
        """Calling the instance is equivalent to ``fit_transform``."""
        a = self.emd(self.image)
        b = self.emd.fit_transform(self.image)
        np.testing.assert_allclose(a, b)

    def test_max_imfs_caps_oscillatory_modes(self) -> None:
        """A positive ``max_imfs`` stops extracting further oscillatory IMFs."""
        imfs = EMD2D(max_imfs=1, max_iter=20).fit_transform(self.image)
        self.assertEqual(imfs.ndim, 3)
        self.assertEqual(imfs.shape[1:], self.image.shape)
        self.assertGreaterEqual(imfs.shape[0], 1)
        np.testing.assert_allclose(imfs.sum(axis=0), self.image, atol=1e-10)

    def test_find_extrema_excludes_border(self) -> None:
        """``find_extrema`` returns interior maxima / minima of a 2-D cosine."""
        yy, xx = np.mgrid[0:21, 0:21]
        image = np.cos(2.0 * np.pi * xx / 10.0) * np.cos(2.0 * np.pi * yy / 10.0)
        min_peaks, max_peaks = EMD2D.find_extrema(image)
        self.assertEqual(len(min_peaks), 2)
        self.assertEqual(len(max_peaks), 2)
        self.assertGreater(max_peaks[0].size, 0)
        self.assertGreater(min_peaks[0].size, 0)
        self.assertFalse(np.any(max_peaks[0] == 0))
        self.assertFalse(np.any(max_peaks[0] == image.shape[0] - 1))
        self.assertFalse(np.any(max_peaks[1] == 0))
        self.assertFalse(np.any(max_peaks[1] == image.shape[1] - 1))

    def test_find_extrema_isolated_peak(self) -> None:
        """A single interior peak is reported as a maximum."""
        image = np.zeros((7, 7))
        image[3, 3] = 1.0
        _, max_peaks = EMD2D.find_extrema(image)
        self.assertIn(3, max_peaks[0])
        self.assertIn(3, max_peaks[1])

    def test_prepare_image_mirrors_3x3(self) -> None:
        """``prepare_image`` tiles a 3×3 mirrored copy around the original."""
        image = np.arange(12, dtype=float).reshape(3, 4)
        big = EMD2D.prepare_image(image)
        self.assertEqual(big.shape, (9, 12))
        np.testing.assert_array_equal(big[3:6, 4:8], image)
        np.testing.assert_array_equal(big[3:6, :4], np.fliplr(image))
        np.testing.assert_array_equal(big[3:6, 8:], np.fliplr(image))
        np.testing.assert_array_equal(big[:3, 4:8], np.flipud(image))
        np.testing.assert_array_equal(big[6:, 4:8], np.flipud(image))

    def test_spline_points_interpolates_plane(self) -> None:
        """``spline_points`` recovers a planar surface from a regular grid."""
        xx, yy = np.mgrid[0:5, 0:5]
        values = (xx + 2.0 * yy).astype(float)
        surface = EMD2D.spline_points(
            xx.ravel(), yy.ravel(), values.ravel(), np.arange(5), np.arange(5)
        )
        self.assertEqual(surface.shape, (5, 5))
        np.testing.assert_allclose(surface, values, atol=1e-6)

    def test_stop_condition(self) -> None:
        """Decomposition stops when the stacked IMFs already reconstruct the image."""
        image = np.arange(9, dtype=float).reshape(3, 3)
        exact = np.stack([image, np.zeros_like(image)])
        self.assertTrue(EMD2D.stop_condition(image, exact))
        self.assertFalse(EMD2D.stop_condition(image, exact * 2.0))

    def test_check_proto_imf_flat_mean_env(self) -> None:
        """A spatially constant envelope mean is accepted as an IMF."""
        emd = EMD2D(mean_thr=0.01, mse_thr=0.01)
        proto = np.ones((5, 5))
        mean_env = np.full((5, 5), 3.0)
        self.assertTrue(emd.check_proto_imf(proto, proto + 1.0, mean_env))

    def test_check_proto_imf_unchanged_or_small(self) -> None:
        """Unchanged iterates or near-zero proto-IMFs pass the IMF check."""
        emd = EMD2D(mean_thr=0.05, mse_thr=0.05)
        proto = np.full((4, 4), 0.01)
        varying = np.arange(16, dtype=float).reshape(4, 4)
        self.assertTrue(emd.check_proto_imf(proto, proto, varying))
        self.assertTrue(emd.check_proto_imf(proto, varying, varying))
        large = np.arange(16, dtype=float).reshape(4, 4) + 1.0
        self.assertFalse(emd.check_proto_imf(large, large + 5.0, large))

    def test_extract_max_min_spline_shape(self) -> None:
        """Envelope interpolation returns two images of the original size."""
        min_env, max_env = self.emd.extract_max_min_spline(self.image)
        self.assertEqual(min_env.shape, self.image.shape)
        self.assertEqual(max_env.shape, self.image.shape)
        self.assertTrue(np.all(np.isfinite(min_env)))
        self.assertTrue(np.all(np.isfinite(max_env)))
        self.assertGreater(np.mean(max_env), np.mean(min_env))

    def test_fix_epochs_sifting(self) -> None:
        """``fix_epochs`` still yields a finite, reconstructible decomposition."""
        emd = EMD2D(max_imfs=2, max_iter=20)
        emd.fix_epochs = 2
        imfs = emd.fit_transform(self.image)
        self.assertTrue(np.all(np.isfinite(imfs)))
        np.testing.assert_allclose(imfs.sum(axis=0), self.image, atol=1e-10)

    def test_fix_epochs_h_sifting(self) -> None:
        """``fix_epochs_h`` still yields a finite, reconstructible decomposition."""
        emd = EMD2D(max_imfs=2, max_iter=20)
        emd.fix_epochs_h = 2
        imfs = emd.fit_transform(self.image)
        self.assertTrue(np.all(np.isfinite(imfs)))
        np.testing.assert_allclose(imfs.sum(axis=0), self.image, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
