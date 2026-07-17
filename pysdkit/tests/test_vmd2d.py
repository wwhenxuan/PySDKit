# -*- coding: utf-8 -*-
"""
Created on 2025/02/20 22:44:31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import VMD2D
from pysdkit.data import test_univariate_image, test_grayscale, get_meshgrid_2D


class VMD2DTest(unittest.TestCase):
    """Verify two-dimensional Variational Mode Decomposition (VMD2D)"""

    image = test_univariate_image(case=1)
    grayscale = test_grayscale()

    def test_fit_transform(self) -> None:
        """Verify that signal decomposition runs normally"""
        # Number of decomposition modes
        K = 5

        # Create a 2D signal decomposition test instance
        vmd2d = VMD2D(
            K=K, alpha=5000, tau=0.25, DC=True, init="random", tol=1e-6, max_iter=64
        )

        # Run signal decomposition
        IMFs = vmd2d.fit_transform(self.grayscale)

        # Verify that output image height and width match the input image
        H, W, C = IMFs.shape
        self.assertEqual(
            first=H,
            second=self.grayscale.shape[0],
            msg="Wrong height of the decomposed image",
        )
        self.assertEqual(
            first=W,
            second=self.grayscale.shape[1],
            msg="Wrong width of the decomposed image",
        )
        self.assertEqual(
            first=C, second=K, msg="Wrong number of modes produced by the algorithm"
        )

    def test_default_call(self) -> None:
        """Verify that the call method can run normally"""
        # Number of decomposition modes
        K = 5

        # Create a 2D signal decomposition test instance
        vmd2d = VMD2D(
            K=K, alpha=5000, tau=0.25, DC=True, init="random", tol=1e-6, max_iter=64
        )

        # Run signal decomposition
        IMFs = vmd2d(self.grayscale)

        # Verify that output image height and width match the input image
        H, W, C = IMFs.shape
        self.assertEqual(
            first=H,
            second=self.grayscale.shape[0],
            msg="Wrong height of the decomposed image",
        )
        self.assertEqual(
            first=W,
            second=self.grayscale.shape[1],
            msg="Wrong width of the decomposed image",
        )
        self.assertEqual(
            first=C, second=K, msg="Wrong number of modes produced by the algorithm"
        )

    def test_VMD2D_DC(self) -> None:
        """Verify that VMD2D can separate the DC signal component"""

        # Create a signal decomposition algorithm instance
        vmd2d = VMD2D(
            K=2, alpha=5000, tau=0.0, DC=True, init="random", tol=1e-6, max_iter=5000
        )

        for size in [16, 32, 64]:
            # Generate a simple DC component
            DC = np.ones(shape=(size, size)) * 100

            # Generate mesh grids
            x, y = get_meshgrid_2D(low=0, high=2 * np.pi, sampling_rate=size)

            # Generate a single-mode function
            mode = np.sin(2 * np.pi * x) + np.cos(2 * np.pi * y)

            # Synthesize a 2D signal
            image = DC + mode

            # Run signal decomposition
            IMFs = vmd2d.fit_transform(image)

            # Verify that the DC component can be extracted
            diff_DC = np.allclose(IMFs[:, :, 0], DC, atol=1)
            self.assertTrue(
                expr=diff_DC, msg="VMD2D failed to extract the DC component"
            )

    def test_init_omega(self) -> None:
        """Verify multiple initialization methods for omega_k in VMD2D"""
        # Iterate over different initialization methods
        for init in ["random", "zero"]:
            # Create an algorithm instance from the input parameters
            vmd2d = VMD2D(K=5, alpha=5000, tau=0.25, init=init, tol=1e-6, max_iter=64)
            # Run signal decomposition
            IMFs = vmd2d.fit_transform(self.grayscale)
            self.assertEqual(
                first=IMFs.shape[-1], second=5, msg="VMD2D algorithm output is wrong"
            )

    def test_wrong_init_omega(self) -> None:
        """Verify invalid omega_k initialization methods"""
        init = "uniform"
        # Create an algorithm instance with invalid parameters
        with self.assertRaises(ValueError):
            vmd2d = VMD2D(K=5, alpha=5000, tau=0.25, init=init, tol=1e-6, max_iter=64)
            vmd2d.fit_transform(self.grayscale)

    def test_return_all(self) -> None:
        """Verify that VMD2D can return all information via parameters"""

        # Create a VMD2D instance and obtain all outputs
        vmd = VMD2D(
            K=5, alpha=5000, tau=0.25, DC=False, init="random", tol=1e-6, max_iter=64
        )
        outputs = vmd.fit_transform(self.grayscale, return_all=True)

        # Verify the number of returned variables
        self.assertEqual(first=len(outputs), second=3, msg="Expecting three outputs")

        # Unpack all returned variables
        u, u_hat, omega = outputs

        # Verify frequency component information
        self.assertEqual(
            first=len(u_hat.shape), second=3, msg="Expecting three dimensions of u_hat"
        )
        self.assertEqual(
            first=len(omega.shape), second=3, msg="Expecting three dimensions of omega"
        )


if __name__ == "__main__":
    unittest.main()
