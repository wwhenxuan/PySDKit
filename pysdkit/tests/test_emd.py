# -*- coding: utf-8 -*-
"""
Created on 2025/02/15 16:18:33
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import EMD
from pysdkit.data import test_emd, test_univariate_signal


class EMDTest(unittest.TestCase):
    """Test whether Empirical Mode Decomposition (EMD) runs normally"""

    def test_fit_transform(self) -> None:
        """Verify that signal decomposition can be performed normally"""
        # Create an algorithm instance object
        emd = EMD()
        for index in range(1, 4):
            # Get the test signal
            time, signal = test_univariate_signal(case=index)
            IMFs = emd.fit_transform(signal)
            # Determine the output dimension
            dim = len(IMFs.shape)
            self.assertEqual(
                first=dim,
                second=2,
                msg="The output shape of the decomposed signal is wrong",
            )
            # Determine the length of the output signal
            _, length = IMFs.shape
            self.assertEqual(
                first=len(signal),
                second=length,
                msg="Wrong length of decomposed signal",
            )

    def test_default_call(self) -> None:
        """Verify that the call method can run normally"""
        time, signal = test_emd()
        # Create an algorithm instance object
        emd = EMD()
        IMFs = emd(signal)

        # Determine the output dimension
        dim = len(IMFs.shape)
        self.assertEqual(
            first=dim,
            second=2,
            msg="The output shape of the decomposed signal is wrong",
        )
        # Determine the length of the output signal
        _, length = IMFs.shape
        self.assertEqual(
            first=len(signal), second=length, msg="Wrong length of decomposed signal"
        )

    def test_different_length_inputs(self) -> None:
        """Verify the exception when the timestamp array and input signal length differ"""
        time = np.arange(100)
        signal = np.random.randn(125)

        emd = EMD()
        with self.assertRaises(ValueError):
            emd.fit_transform(signal=signal, time=time)

    def test_trend(self) -> None:
        """Verify input consisting of a single trend signal"""
        emd = EMD()

        # Create timestamps and a signal with only a trend component
        time = np.arange(0, 1, 0.01)
        signal = 2 * time

        # Run signal decomposition to obtain IMFs
        IMFs = emd.fit_transform(signal=signal, time=time)
        self.assertEqual(first=IMFs.shape[0], second=1, msg="Expecting single IMF")
        self.assertTrue(np.allclose(signal, IMFs[0]))

    def test_single_imf(self) -> None:
        """Verify input consisting of a single IMF"""
        emd = EMD()

        # Create a timestamp array
        time = np.arange(0, 1, 0.001)

        # Create a cosine signal
        cosine = np.cos(2 * np.pi * 4 * time)

        # Test input consisting of a single cosine
        IMFs = emd.fit_transform(signal=cosine.copy(), time=time)
        self.assertEqual(first=IMFs.shape[0], second=1, msg="Expecting single IMF!")

        # Verify numerical difference between input and output
        diff = np.allclose(IMFs[0], cosine)
        self.assertTrue(diff, "Expecting 1st IMF to be cos(8 * pi * t)")

        # Create an input trend component
        trend = 3 * (time - 0.5)

        # Test input consisting of cosine plus trend
        IMFs = emd.fit_transform(signal=trend.copy() + cosine.copy(), time=time)
        self.assertEqual(
            first=IMFs.shape[0], second=2, msg="Expecting two IMF of cosine and trend!"
        )

        # Further verify numerical differences between the two modal outputs
        diff_cosine = np.allclose(IMFs[0], cosine, atol=0.2)
        self.assertTrue(diff_cosine, "Expecting 1st IMF to be cosine")
        diff_trend = np.allclose(IMFs[1], trend, atol=0.2)
        self.assertTrue(diff_trend, "Expecting 2nd IMF to be trend")

    def test_spline_kind(self) -> None:
        """Verify that all interpolation algorithms in EMD run normally"""
        # Create a test signal with two different components
        time = np.arange(0, 1, 0.01)
        cosine = np.cos(2 * np.pi * 4 * time)
        trend = 3 * (time - 0.1)
        signal = cosine.copy() + trend.copy()

        for spline_kind in [
            "akima",
            "cubic",
            "pchip",
            "cubic_hermite",
            "slinear",
            "quadratic",
            "linear",
        ]:
            # Iterate over all interpolation algorithms and create instances
            emd = EMD(spline_kind=spline_kind)

            # Run signal decomposition and verify the output
            IMFs = emd.fit_transform(signal=signal, time=time)

            # Verify that the number of IMFs meets the requirement
            self.assertEqual(
                first=IMFs.shape[0],
                second=2,
                msg=f"Expecting two IMF of cosine and trend when the `spline_kind` is {spline_kind}",
            )

            # Further verify numerical differences between the two modal outputs
            diff_cosine = np.allclose(IMFs[0], cosine, atol=0.3)
            self.assertTrue(
                diff_cosine,
                msg=f"Expecting 1st IMF to be cosine when the `spline_kind` is {spline_kind}",
            )
            diff_trend = np.allclose(IMFs[1], trend, atol=0.3)
            self.assertTrue(
                diff_trend,
                msg=f"Expecting 2nd IMF to be trend when the `spline_kind` is {spline_kind}",
            )

    def test_wrong_spline_kind(self) -> None:
        """Verify that invalid interpolation type input raises an exception"""
        spline_kind = "wrong"

        # Create random input
        time = np.arange(10)
        signal = np.random.randn(10)

        # Validate the invalid interpolation type
        with self.assertRaises(ValueError):
            emd = EMD(spline_kind=spline_kind)
            emd.fit_transform(signal=signal, time=time)

    def test_extrema_detection(self) -> None:
        """Verify that all extrema detection algorithms in EMD run normally"""
        # Create a test signal with two different components
        time = np.arange(0, 1, 0.01)
        cosine = np.cos(2 * np.pi * 4 * time)
        trend = 3 * (time - 0.1)
        signal = cosine.copy() + trend.copy()

        for extrema_detection in ["parabol", "simple"]:
            # Iterate over extrema detection methods and create instances
            emd = EMD(extrema_detection=extrema_detection)

            # Run signal decomposition and verify the output
            IMFs = emd.fit_transform(signal=signal, time=time)

            # Verify that the number of IMFs meets the requirement
            self.assertEqual(
                first=IMFs.shape[0],
                second=2,
                msg=f"Expecting two IMF of cosine and trend when the `spline_kind` is {extrema_detection}",
            )

            # Further verify numerical differences between the two modal outputs
            diff_cosine = np.allclose(IMFs[0], cosine, atol=0.3)
            self.assertTrue(
                diff_cosine,
                msg=f"Expecting 1st IMF to be cosine when the `extrema_detection` is {extrema_detection}",
            )
            diff_trend = np.allclose(IMFs[1], trend, atol=0.3)
            self.assertTrue(
                diff_trend,
                msg=f"Expecting 2nd IMF to be trend when the `extrema_detection` is {extrema_detection}",
            )

    def test_wrong_extrema_detection(self) -> None:
        """Verify that invalid extrema detection type input raises an exception"""
        extrema_detection = "wrong"

        # Create random input
        time = np.arange(10)
        signal = np.random.randn(10)

        # Validate the invalid extrema detection type
        with self.assertRaises(ValueError):
            emd = EMD(extrema_detection=extrema_detection)
            emd.fit_transform(signal=signal, time=time)

    def test_max_iteration_flag(self) -> None:
        """Verify the model maximum iteration count"""
        # Create a random signal for validation
        signal = np.random.random(200)
        emd = EMD()
        emd.MAX_ITERATION = 10
        emd.FIXE = 20
        IMFs = emd.fit_transform(signal)

        # There's not much to test, except that it doesn't fail.
        # With low MAX_ITERATION value for random signal it's
        # guaranteed to have at least 2 IMFs.
        self.assertTrue(IMFs.shape[0] > 1)

    def test_get_imfs_and_residue(self) -> None:
        """Verify that IMFs and residue can be obtained normally after decomposition"""
        signal = np.random.random(200)
        emd = EMD(**{"MAX_ITERATION": 10, "FIXE": 20})
        all_imfs = emd(signal, max_imfs=3)

        imfs, residue = emd.get_imfs_and_residue()
        self.assertEqual(
            all_imfs.shape[0], imfs.shape[0] + 1, msg="Compare number of components"
        )
        self.assertTrue(
            np.array_equal(all_imfs[:-1], imfs),
            msg="Shouldn't matter where imfs are from",
        )
        self.assertTrue(
            np.array_equal(all_imfs[-1], residue),
            msg="Residue, if any, is the last row",
        )

    def test_get_imfs_and_residue_without_running(self) -> None:
        """Verify that output cannot be obtained when the algorithm has not been run"""
        emd = EMD()
        with self.assertRaises(ValueError):
            # Since decomposition was not performed, IMFs and residue should be unavailable
            _, _ = emd.get_imfs_and_residue()

    def test_get_imfs_and_trend(self) -> None:
        """Verify that IMFs and trend can be obtained normally after decomposition"""
        # Create an algorithm instance and test signal
        emd = EMD()
        time = np.linspace(0, 2 * np.pi, 100)
        expected_trend = 5 * time
        signal = (
            2 * np.sin(4.1 * 6.28 * time)
            + 1.2 * np.cos(7.4 * 6.28 * time)
            + expected_trend
        )

        # Run signal decomposition
        IMFs = emd(signal)
        # Obtain the trend component
        imfs, trend = emd.get_imfs_and_trend()

        # Further numerical validation of the trend component
        onset_trend = trend - trend.mean()
        onset_expected_trend = expected_trend - expected_trend.mean()
        self.assertEqual(
            IMFs.shape[0], imfs.shape[0] + 1, "Compare number of components"
        )
        self.assertTrue(
            np.array_equal(IMFs[:-1], imfs), "Shouldn't matter where imfs are from"
        )
        self.assertTrue(
            np.allclose(onset_trend, onset_expected_trend, rtol=0.1, atol=0.5),
            "Extracted trend should be close to the actual trend",
        )


if __name__ == "__main__":
    unittest.main()
