import unittest

import matplotlib.pyplot as plt
import numpy as np

from pysdkit.data import test_emd, test_multivariate_signal

from pysdkit import FAEMD


class FAEMDTest(unittest.TestCase):
    """Test whether Fast Adaptive EMD (FAEMD) runs normally"""

    def test_fit_transform(self) -> None:
        """Verify that signal decomposition can be performed normally"""
        # Create an algorithm validation instance
        faemd = FAEMD(max_imfs=3)

        # Get a univariate test signal
        _, signal = test_emd()

        # Test univariate signals
        for num_imfs in range(2, 5):
            # Run the signal decomposition algorithm
            IMFs = faemd.fit_transform(signal, max_imfs=num_imfs)

            # Get the decomposed IMFs
            num_vars, seq_len = IMFs.shape

            # Check the number of decomposition modes
            self.assertEqual(
                first=num_vars,
                second=num_imfs,
                msg="Wrong number of decomposition modes",
            )

            # Check the signal length
            self.assertEqual(
                first=seq_len,
                second=len(signal),
                msg="Wrong length of decomposed signal",
            )

            # Verify reconstruction quality
            diff = np.allclose(np.sum(IMFs, axis=0), signal, atol=1e-6)
            self.assertTrue(
                expr=diff,
                msg="Decomposed sub-modes cannot reconstruct the original signal",
            )

        # Get a bivariate test signal
        _, signal = test_multivariate_signal()
        # Get signal shape information
        num_vars, seq_len = signal.shape

        # Run multivariate signal decomposition
        IMFs = faemd.fit_transform(signal, max_imfs=num_vars)

        # Verify the number of decomposed modes
        self.assertEqual(
            first=IMFs.shape[0],
            second=num_vars,
            msg="Wrong number of modes in the decomposed signal",
        )

        # Verify the length of the decomposed signal
        self.assertEqual(
            first=IMFs.shape[1], second=seq_len, msg="Wrong length of decomposed signal"
        )

        # Verify reconstruction quality
        for num in range(num_vars):
            diff = np.allclose(np.sum(IMFs[:, :, num], axis=0), signal[num], atol=1e-6)
            self.assertTrue(
                expr=diff,
                msg="Decomposed sub-modes cannot reconstruct the original signal",
            )

    def test_default_call(self) -> None:
        """Verify that signal decomposition can be performed normally"""
        # Create an algorithm validation instance
        faemd = FAEMD(max_imfs=3)

        # Get a univariate test signal
        _, signal = test_emd()

        # Test univariate signals
        for num_imfs in range(2, 5):
            # Run the signal decomposition algorithm
            IMFs = faemd(signal, max_imfs=num_imfs)

            # Get the decomposed IMFs
            num_vars, seq_len = IMFs.shape

            # Check the number of decomposition modes
            self.assertEqual(
                first=num_vars,
                second=num_imfs,
                msg="Wrong number of decomposition modes",
            )

            # Check the signal length
            self.assertEqual(
                first=seq_len,
                second=len(signal),
                msg="Wrong length of decomposed signal",
            )

            # Verify reconstruction quality
            diff = np.allclose(np.sum(IMFs, axis=0), signal, atol=1e-6)
            self.assertTrue(
                expr=diff,
                msg="Decomposed sub-modes cannot reconstruct the original signal",
            )

        # Get a bivariate test signal
        _, signal = test_multivariate_signal()
        # Get signal shape information
        num_vars, seq_len = signal.shape

        # Run multivariate signal decomposition
        IMFs = faemd(signal, max_imfs=num_vars)

        # Verify the number of decomposed modes
        self.assertEqual(
            first=IMFs.shape[0],
            second=num_vars,
            msg="Wrong number of modes in the decomposed signal",
        )

        # Verify the length of the decomposed signal
        self.assertEqual(
            first=IMFs.shape[1], second=seq_len, msg="Wrong length of decomposed signal"
        )

        # Verify reconstruction quality
        for num in range(num_vars):
            diff = np.allclose(np.sum(IMFs[:, :, num], axis=0), signal[num], atol=1e-6)
            self.assertTrue(
                expr=diff,
                msg="Decomposed sub-modes cannot reconstruct the original signal",
            )

    def test_trend(self) -> None:
        """Verify input consisting of a single trend signal"""
        faemd = FAEMD(max_imfs=3)

        # Create timestamps and a signal with only a trend component
        time = np.arange(0, 1, 0.01)
        signal = 2 * time

        # Run signal decomposition to obtain IMFs
        IMFs = faemd.fit_transform(signal=signal)

        # Verify the last IMF captures the trend
        diff = np.allclose(IMFs[-1], signal, atol=1e-6)
        self.assertTrue(expr=diff, msg="Failed to extract trend information correctly")

    def test_signal_imf(self) -> None:
        """Verify extraction of a single-mode signal"""
        # Create a signal decomposition instance
        faemd = FAEMD(max_imfs=2, tol=1e-10)

        # Create a timestamp array
        time = np.arange(0, 1, 0.001)

        # Create a cosine signal
        cosine = np.cos(2 * np.pi * 4 * time)

        # Test input consisting of a single cosine
        IMFs = faemd.fit_transform(signal=cosine.copy())

        # Verify the first IMF captures the cosine component
        diff = np.allclose(IMFs[0], cosine, atol=1)
        self.assertTrue(
            expr=diff, msg="Failed to extract single-mode information correctly"
        )

    def test_window_type(self) -> None:
        """Verify supported smoothing window types"""
        # Create a test signal
        time, signal = test_emd()

        # Iterate over supported window type indices
        for index in range(7):
            # Create a test instance
            faemd = FAEMD(max_imfs=2, window_type=index)

            # Run the signal decomposition algorithm
            IMFs = faemd.fit_transform(signal=signal)

            # Verify reconstruction quality
            diff = np.allclose(np.sum(IMFs, axis=0), signal, atol=1e-6)
            self.assertTrue(
                expr=diff, msg="Decomposed IMFs cannot reconstruct the original signal"
            )

    def test_wrong_window_type(self) -> None:
        """Verify invalid `window_type` parameters"""
        with self.assertRaises(ValueError):
            # Create a signal decomposition instance with invalid parameters
            FAEMD(max_imfs=2, window_type=-1)


if __name__ == "__main__":
    unittest.main()
