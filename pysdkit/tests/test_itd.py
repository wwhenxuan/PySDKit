# -*- coding: utf-8 -*-
"""
Tests for the Intrinsic Time-Scale Decomposition (ITD) algorithm.
"""
import unittest
import numpy as np

from pysdkit import ITD
from pysdkit.data import test_emd, test_univariate_signal


class ITDTest(unittest.TestCase):
    """Test that ITD runs correctly on various signal types."""

    def test_fit_transform(self) -> None:
        """Verify signal decomposition on built-in test signals."""
        itd = ITD(N_max=5)
        for index in range(1, 4):
            time, signal = test_univariate_signal(case=index)
            IMFs = itd.fit_transform(signal)
            dim = len(IMFs.shape)
            self.assertEqual(
                first=dim,
                second=2,
                msg="The output shape of the decomposed signal is wrong",
            )
            _, length = IMFs.shape
            self.assertEqual(
                first=len(signal),
                second=length,
                msg="Wrong length of decomposed signal",
            )

    def test_default_call(self) -> None:
        """Verify the __call__ interface works."""
        time, signal = test_emd()
        itd = ITD(N_max=5)
        IMFs = itd(signal)

        dim = len(IMFs.shape)
        self.assertEqual(
            first=dim,
            second=2,
            msg="The output shape of the decomposed signal is wrong",
        )
        _, length = IMFs.shape
        self.assertEqual(
            first=len(signal), second=length, msg="Wrong length of decomposed signal"
        )

    def test_reconstruction(self) -> None:
        """Verify that IMFs sum back to the original signal."""
        time, signal = test_univariate_signal(case=1)
        itd = ITD(N_max=5)
        IMFs = itd.fit_transform(signal)

        reconstructed = np.sum(IMFs, axis=0)
        self.assertTrue(
            np.allclose(signal, reconstructed, atol=1e-10),
            "IMFs should reconstruct the original signal",
        )

    def test_near_monotonic_signal(self) -> None:
        """Verify ITD handles near-monotonic signals without crashing (issue #29)."""
        np.random.seed(42)
        N = 200
        t = np.linspace(0, 1, N)
        signal = -1100 * t + 0.01 * np.sin(50 * t) + 0.005 * np.random.randn(N)

        itd = ITD(N_max=3)
        IMFs = itd.fit_transform(signal)

        self.assertTrue(IMFs.shape[0] >= 1, "Should produce at least one component")
        self.assertEqual(IMFs.shape[1], N, "Output length should match input")
        reconstructed = np.sum(IMFs, axis=0)
        self.assertTrue(
            np.allclose(signal, reconstructed, atol=1e-10),
            "IMFs should reconstruct the original signal",
        )

    def test_pure_trend(self) -> None:
        """Verify ITD handles a pure trend (no oscillation)."""
        signal = np.linspace(0, 10, 100)
        itd = ITD(N_max=3)
        IMFs = itd.fit_transform(signal)

        self.assertTrue(IMFs.shape[0] >= 1, "Should produce at least one component")
        reconstructed = np.sum(IMFs, axis=0)
        self.assertTrue(
            np.allclose(signal, reconstructed, atol=1e-10),
            "IMFs should reconstruct the original signal",
        )

    def test_asymmetric_extrema(self) -> None:
        """Verify ITD works when maxima and minima counts differ."""
        t = np.linspace(0, 4 * np.pi, 500)
        signal = np.sin(t) + 0.5 * np.sin(3 * t) + 0.3 * np.cos(0.5 * t)

        itd = ITD(N_max=5)
        IMFs = itd.fit_transform(signal)

        self.assertTrue(IMFs.shape[0] >= 2, "Should produce multiple components")
        reconstructed = np.sum(IMFs, axis=0)
        self.assertTrue(
            np.allclose(signal, reconstructed, atol=1e-10),
            "IMFs should reconstruct the original signal",
        )

    def test_str(self) -> None:
        """Verify string representation."""
        itd = ITD()
        self.assertIn("ITD", str(itd))


if __name__ == "__main__":
    unittest.main()
