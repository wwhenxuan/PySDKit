# -*- coding: utf-8 -*-
"""
Created on 2025/02/21 23:40:33
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import ewt2d, EWT2D
from pysdkit.data import test_univariate_image, get_meshgrid_2D


def _synthetic_texture(size: int = 64) -> np.ndarray:
    """Build a simple multi-frequency synthetic image for EWT2D tests."""
    x, y = get_meshgrid_2D(low=0.0, high=1.0, sampling_rate=size)
    image = (
        np.sin(2.0 * np.pi * 3.0 * x)
        + 0.6 * np.sin(2.0 * np.pi * 12.0 * y)
        + 0.4 * np.sin(2.0 * np.pi * 8.0 * (x + y))
    )
    return image


class EWT2DTest(unittest.TestCase):
    """Test whether 2D Empirical Wavelet Transform (EWT2D) runs normally."""

    image = _synthetic_texture(64)

    def test_fit_transform_tensor(self) -> None:
        """Verify Tensor EWT2D decomposition shape and mode count."""
        for K in range(2, 5):
            ewt2d_c = EWT2D(K=K, method="tensor")
            modes = ewt2d_c.fit_transform(self.image)

            self.assertEqual(modes.ndim, 3, msg="Modes must be (n_modes, H, W)")
            n_modes, H, W = modes.shape
            self.assertEqual(H, self.image.shape[0])
            self.assertEqual(W, self.image.shape[1])
            # Tensor yields (≈K) x (≈K) rectangular supports
            self.assertGreaterEqual(n_modes, K)
            self.assertLessEqual(n_modes, K * K)

    def test_default_call(self) -> None:
        """Verify that the call method mirrors fit_transform."""
        ewt2d_c = EWT2D(K=3, method="tensor")
        modes_call = ewt2d_c(self.image)
        modes_fit = ewt2d_c.fit_transform(self.image)
        self.assertEqual(modes_call.shape, modes_fit.shape)
        self.assertTrue(np.allclose(modes_call, modes_fit))

    def test_ewt2d_function(self) -> None:
        """Test the functional ewt2d interface."""
        modes = ewt2d(self.image, K=3, method="tensor")
        self.assertEqual(modes.ndim, 3)
        self.assertEqual(modes.shape[1:], self.image.shape)

    def test_return_all_tensor(self) -> None:
        """Verify return_all extras for the Tensor method."""
        ewt2d_c = EWT2D(K=3, method="tensor")
        modes, extras = ewt2d_c.fit_transform(self.image, return_all=True)

        self.assertEqual(extras["method"], "tensor")
        self.assertIn("mfb_row", extras)
        self.assertIn("mfb_col", extras)
        self.assertIn("bounds_row", extras)
        self.assertIn("bounds_col", extras)
        self.assertEqual(modes.shape[0], extras["n_row"] * extras["n_col"])

    def test_inverse_tensor(self) -> None:
        """Verify near-perfect reconstruction for Tensor EWT2D."""
        ewt2d_c = EWT2D(K=3, method="tensor", detect="locmax", reg="average")
        modes, extras = ewt2d_c.fit_transform(self.image, return_all=True)
        recon = ewt2d_c.inverse_transform(modes, extras=extras)

        self.assertEqual(recon.shape, self.image.shape)
        rel_err = np.linalg.norm(recon - self.image) / (
            np.linalg.norm(self.image) + 1e-12
        )
        self.assertLess(
            rel_err,
            1e-6,
            msg=f"Tensor reconstruction error too large: {rel_err}",
        )

    def test_inverse_from_cache(self) -> None:
        """inverse_transform should work from filters cached on the instance."""
        ewt2d_c = EWT2D(K=3, method="tensor")
        modes = ewt2d_c.fit_transform(self.image)
        recon = ewt2d_c.inverse_transform(modes)
        rel_err = np.linalg.norm(recon - self.image) / (
            np.linalg.norm(self.image) + 1e-12
        )
        self.assertLess(rel_err, 1e-6)

    def test_fit_transform_lp(self) -> None:
        """Verify Littlewood-Paley EWT2D decomposition shape."""
        ewt2d_c = EWT2D(K=3, method="lp", detect="locmax", reg="average")
        modes = ewt2d_c.fit_transform(self.image)

        self.assertEqual(modes.ndim, 3)
        n_modes, H, W = modes.shape
        self.assertEqual((H, W), self.image.shape)
        self.assertGreaterEqual(n_modes, 2)

    def test_inverse_lp(self) -> None:
        """Verify near-perfect reconstruction for LP EWT2D."""
        ewt2d_c = EWT2D(K=3, method="lp", detect="locmax", reg="average")
        modes, extras = ewt2d_c.fit_transform(self.image, return_all=True)
        recon = ewt2d_c.inverse_transform(modes, extras=extras)

        rel_err = np.linalg.norm(recon - self.image) / (
            np.linalg.norm(self.image) + 1e-12
        )
        self.assertLess(
            rel_err,
            1e-5,
            msg=f"LP reconstruction error too large: {rel_err}",
        )

    def test_detect_methods(self) -> None:
        """Test the three shared boundary-detection methods on Tensor EWT2D."""
        for detect in ["locmax", "locmaxmin", "locmaxminf"]:
            ewt2d_c = EWT2D(K=3, method="tensor", detect=detect)
            modes = ewt2d_c.fit_transform(self.image)
            self.assertEqual(modes.shape[1:], self.image.shape)

    def test_wrong_detect(self) -> None:
        """Invalid detect method should raise ValueError."""
        ewt2d_c = EWT2D(K=3, method="tensor", detect="wrong")
        with self.assertRaises(ValueError):
            ewt2d_c.fit_transform(self.image)

    def test_wrong_method(self) -> None:
        """Invalid construction method should raise ValueError at init."""
        with self.assertRaises(ValueError):
            EWT2D(K=3, method="curvelet")

    def test_wrong_ndim(self) -> None:
        """Non-2D input should raise ValueError."""
        ewt2d_c = EWT2D(K=3)
        with self.assertRaises(ValueError):
            ewt2d_c.fit_transform(np.ones(32))

    def test_str(self) -> None:
        """Verify the printable algorithm name."""
        name = str(EWT2D())
        self.assertIn("EWT2D", name)

    def test_on_library_image(self) -> None:
        """Run Tensor EWT2D on a library synthetic image (smaller size)."""
        image = test_univariate_image(case=1, sampling_rate=64)
        ewt2d_c = EWT2D(K=3, method="tensor")
        modes = ewt2d_c.fit_transform(image)
        self.assertEqual(modes.shape[1:], image.shape)
        recon = ewt2d_c.inverse_transform(modes)
        rel_err = np.linalg.norm(recon - image) / (np.linalg.norm(image) + 1e-12)
        self.assertLess(rel_err, 1e-6)


if __name__ == "__main__":
    unittest.main()
