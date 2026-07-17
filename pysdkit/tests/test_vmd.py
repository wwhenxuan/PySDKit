# -*- coding: utf-8 -*-
"""
Created on 2025/02/15 18:26:17
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import VMD
from pysdkit import vmd as vmd_f
from pysdkit.data import test_emd


class VMDTest(unittest.TestCase):
    """Automated tests for Variational Mode Decomposition (VMD)"""

    def test_fit_transform(self) -> None:
        """Verify that signal decomposition can be performed normally"""
        # Create an algorithm instance object
        vmd = VMD(K=3, alpha=1000, tau=0.0)

        time, signal = test_emd()
        IMFs = vmd.fit_transform(signal)
        # Check the output dimensionality
        dim = len(IMFs.shape)
        self.assertEqual(
            first=dim,
            second=2,
            msg="The output shape of the decomposed signal is wrong",
        )
        # Check the length of the output signal
        _, length = IMFs.shape
        self.assertEqual(
            first=len(signal),
            second=length,
            msg="Wrong length of the decomposed signal",
        )

    def test_default_call(self) -> None:
        """Verify that the call method can run normally"""
        time, signal = test_emd()
        # Create an algorithm instance object
        vmd = VMD(K=3, alpha=1000, tau=0.0)
        IMFs = vmd(signal)

        # Check the output dimensionality
        dim = len(IMFs.shape)
        self.assertEqual(
            first=dim,
            second=2,
            msg="The output shape of the decomposed signal is wrong",
        )
        # Check the length of the output signal
        _, length = IMFs.shape
        self.assertEqual(
            first=len(signal),
            second=length,
            msg="Wrong length of the decomposed signal",
        )

    def test_vmd_function(self) -> None:
        """Verify that the VMD function interface can run normally"""
        time, signal = test_emd()
        # Create an algorithm instance object
        IMFs, _, _ = vmd_f(signal, alpha=1000, K=3, tau=0.0)

        # Check the output dimensionality
        dim = len(IMFs.shape)
        self.assertEqual(
            first=dim,
            second=2,
            msg="The output shape of the decomposed signal is wrong",
        )
        # Check the length of the output signal
        _, length = IMFs.shape
        self.assertEqual(
            first=len(signal),
            second=length,
            msg="Wrong length of the decomposed signal",
        )

    def test_trend(self) -> None:
        """Test decomposition of a pure trend signal"""
        vmd = VMD(K=1, alpha=1000, tau=0.0)

        # Create a timestamp array and a pure trend signal
        time = np.arange(0, 1, 0.01)
        signal = 2 * time

        # Run the decomposition and obtain IMFs
        IMFs = vmd.fit_transform(signal=signal)
        self.assertEqual(first=IMFs.shape[0], second=1, msg="Expecting single IMF")
        self.assertTrue(np.allclose(signal, IMFs[0], atol=0.1))

    def test_single_imf(self) -> None:
        """Test decomposition of a single intrinsic mode"""
        vmd = VMD(K=1, alpha=1000, tau=0.0)

        # Create a timestamp array
        time = np.arange(0, 1, 0.001)

        # Create a cosine signal
        cosine = np.cos(2 * np.pi * 4 * time)

        # Decompose a pure cosine input
        IMFs = vmd.fit_transform(signal=cosine.copy())
        # Check numerical difference between input and output
        diff = np.allclose(IMFs[0], cosine, atol=0.1)
        self.assertTrue(diff, "Expecting 1st IMF to be cos(8 * pi * t)")

        # Create a trend component
        trend = 3 * (time - 0.5)
        vmd = VMD(K=2, alpha=1000, tau=0.0)

        # Decompose a mixture of cosine and trend
        IMFs = vmd.fit_transform(signal=trend.copy() + cosine.copy())
        self.assertEqual(
            first=IMFs.shape[0], second=2, msg="Expecting two IMF of cosine and trend!"
        )

        # Further check numerical differences of the two modes
        diff_cosine = np.allclose(IMFs[0], trend, atol=0.2)
        self.assertTrue(diff_cosine, "Expecting 1st IMF to be trend")

        diff_trend = np.allclose(IMFs[1], cosine, atol=0.2)
        self.assertTrue(diff_trend, "Expecting 2nd IMF to be cosine")

    def test_vmd_DC(self) -> None:
        """Verify that VMD can separate the DC component"""
        # Create a VMD instance
        vmd = VMD(K=2, alpha=1000, tau=0.0, DC=True)

        # Create a signal with a DC component
        time = np.arange(0, 1, 0.001)
        DC = 2
        cosine = np.cos(2 * np.pi * 4 * time)
        signal = DC + cosine.copy()

        # Run the signal decomposition algorithm
        IMFs = vmd.fit_transform(signal=signal)

        # First check the number of modes
        self.assertEqual(
            first=IMFs.shape[0], second=2, msg="Expecting the number of IMFs is Two"
        )

        # Further check numerical differences of the two modes
        diff_DC = np.allclose(IMFs[0], np.ones_like(time) * DC, atol=0.1)
        self.assertTrue(diff_DC, "Expecting 1st IMF to be DC")

        diff_cosine = np.allclose(IMFs[1], cosine, atol=0.2)
        self.assertTrue(diff_cosine, "Expecting 2nd IMF to be cosine")

    def test_init_omega(self) -> None:
        """Verify the three initialization methods for omega_k"""
        time, signal = test_emd()

        # Iterate over the three initialization methods
        for init in ["uniform", "random", "zero"]:
            # Create an algorithm instance from the input parameters
            vmd = VMD(K=2, alpha=1000, tau=0.0, init=init)
            # Run the signal decomposition algorithm
            IMFs = vmd.fit_transform(signal=signal)
            self.assertEqual(
                first=IMFs.shape[0], second=2, msg="Expecting IMF number is Two"
            )

    def test_wrong_init_omega(self) -> None:
        """Verify that an invalid omega_k initialization raises"""
        time, signal = test_emd()
        init = "wrong"
        # Create an algorithm instance with invalid parameters
        with self.assertRaises(ValueError):
            vmd = VMD(K=2, alpha=1000, tau=0.0, init=init)
            vmd.fit_transform(signal=signal)

    def test_return_all(self) -> None:
        """Verify all information returned by VMD"""
        time, signal = test_emd()

        # Create VMD and request all outputs
        vmd = VMD(K=2, alpha=1000, tau=0.0)
        outputs = vmd.fit_transform(signal=signal, return_all=True)

        # Check the number of returned variables
        self.assertEqual(first=len(outputs), second=3, msg="Expecting three outputs")

        # Unpack all returned variables
        u, u_hat, omega = outputs

        # Check frequency-component information
        self.assertEqual(
            first=len(u_hat.shape), second=2, msg="Expecting two dimensions of u_hat"
        )
        self.assertEqual(
            first=len(omega.shape), second=2, msg="Expecting two dimensions of omega"
        )

    def test_fmirror(self) -> None:
        """Verify the mirror-extension helper `fmirror`"""
        # Create an input signal
        array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        vmd = VMD(K=2, alpha=1000, tau=0.0)

        # Iterate over partial lengths to verify the output
        for i in range(1, len(array)):
            fMirr = vmd.fmirror(ts=array, sym=i)
            self.assertEqual(
                len(fMirr),
                len(array) + i * 2,
                msg=f"Something went wrong on the fMirr with length {len(fMirr)} and {len(array)}",
            )


if __name__ == "__main__":
    unittest.main()
