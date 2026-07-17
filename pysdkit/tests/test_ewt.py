# -*- coding: utf-8 -*-
"""
Created on 2025/02/21 23:40:33
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import ewt, EWT
from pysdkit.data import test_emd


class EWTTest(unittest.TestCase):
    """Test whether Empirical Wavelet Transform (EWT) runs normally"""

    def test_fit_transform(self) -> None:
        """Verify that signal decomposition can be performed normally"""
        # Generate a test signal
        time, signal = test_emd()
        # Iterate over different numbers of modes
        for K in range(2, 6):
            # Create a signal-decomposition instance
            ewt_c = EWT(K=K)
            # Run the signal decomposition algorithm
            IMFs = ewt_c.fit_transform(signal)
            # Check the output dimensionality
            dim = len(IMFs.shape)
            self.assertEqual(
                first=dim,
                second=2,
                msg="The output shape of the decomposed signal is wrong",
            )
            # Check the length of the output signal
            number, length = IMFs.shape
            self.assertEqual(
                first=len(signal),
                second=length,
                msg="Wrong length of the decomposed signal",
            )
            # Check the number of decomposed modes
            self.assertEqual(first=number, second=K)

    def test_default_call(self) -> None:
        """Verify that the call method can run normally"""
        # Generate a test signal
        time, signal = test_emd()
        # Iterate over different numbers of modes
        for K in range(2, 6):
            # Create a signal-decomposition instance
            ewt_c = EWT(K=K)
            # Run the signal decomposition algorithm
            IMFs = ewt_c(signal)
            # Check the output dimensionality
            dim = len(IMFs.shape)
            self.assertEqual(
                first=dim,
                second=2,
                msg="The output shape of the decomposed signal is wrong",
            )
            # Check the length of the output signal
            number, length = IMFs.shape
            self.assertEqual(
                first=len(signal),
                second=length,
                msg="Wrong length of the decomposed signal",
            )
            # Check the number of decomposed modes
            self.assertEqual(first=number, second=K)

    def test_ewt_function(self) -> None:
        """Test the functional EWT interface"""
        # Generate a test signal
        time, signal = test_emd()
        # Iterate over different numbers of modes
        for K in range(2, 6):
            # Run the signal decomposition algorithm
            IMFs = ewt(signal, K=K)
            # Check the output dimensionality
            dim = len(IMFs.shape)
            self.assertEqual(
                first=dim,
                second=2,
                msg="The output shape of the decomposed signal is wrong",
            )
            # Check the length of the output signal
            number, length = IMFs.shape
            self.assertEqual(
                first=len(signal),
                second=length,
                msg="Wrong length of the decomposed signal",
            )
            # Check the number of decomposed modes
            self.assertEqual(first=number, second=K)

    def test_fmirror(self) -> None:
        """Verify the mirror-extension helper used by EWT"""
        inputs = np.array([1, 2, 3, 4, 5])
        # Create an instance under test
        ewt_c = EWT()
        for sym in range(1, len(inputs)):
            outputs = ewt_c.fmirror(inputs, sym=sym, end=1)
            # Check that the mirrored length matches the expectation
            self.assertEqual(
                first=len(outputs),
                second=len(inputs) + 2 * sym - 1,
                msg="Mirror-extension result does not match expectation",
            )

    def test_detect(self) -> None:
        """Test various boundary-detection methods"""
        # Generate a test signal
        time, signal = test_emd()

        # Iterate over boundary-detection methods
        for detect in ["locmax", "locmaxmin", "locmaxminf"]:
            # Create a signal-decomposition instance
            ewt_c = EWT(K=5, detect=detect)
            # Run the signal decomposition algorithm
            IMFs = ewt_c(signal)
            # Check that decomposition succeeds
            num_imfs, length = IMFs.shape
            self.assertEqual(
                first=length,
                second=len(signal),
                msg="Decomposed signal length does not match the original",
            )

    def test_wrong_detect(self) -> None:
        """Test an invalid boundary-detection method"""
        # Generate a test signal
        time, signal = test_emd()

        # Invalid method name
        detect = "wrong"

        # Create an algorithm instance with invalid parameters
        ewt_c = EWT(K=5, detect=detect)
        with self.assertRaises(ValueError):
            ewt_c(signal)

    def test_reg(self) -> None:
        """Test the regularization method applied to the filter bank"""
        # Generate a test signal
        time, signal = test_emd()

        # Iterate over regularization methods
        for reg in ["average", "gaussian", "wrong"]:
            # Create a signal-decomposition instance
            ewt_c = EWT(K=3, reg=reg)
            # Run the signal decomposition algorithm
            IMFs = ewt_c(signal)
            # Check that decomposition succeeds
            num_imfs, length = IMFs.shape
            self.assertEqual(
                first=num_imfs, second=3, msg="Wrong number of intrinsic mode functions"
            )
            self.assertEqual(
                first=length,
                second=len(signal),
                msg="Decomposed signal length does not match the original",
            )


if __name__ == "__main__":
    unittest.main()
