# -*- coding: utf-8 -*-
"""
Created on 2025/04/16 23:54:01
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest
import numpy as np

from pysdkit import CEEMDAN
from pysdkit.data import test_emd, test_univariate_signal


class CEEMDANTest(unittest.TestCase):
    """
    Test whether CEEMDAN can run normally
    This algorithm is relatively slow
    """

    def test_fit_transform(self) -> None:
        """Verify that signal decomposition can be performed normally"""
        # Create an algorithm instance object
        ceemdan = CEEMDAN()
        for index in range(1, 4):
            # Get the test signal
            time, signal = test_univariate_signal(case=index)
            IMFs = ceemdan.fit_transform(signal)
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
        ceemdan = CEEMDAN()
        IMFs = ceemdan(signal)

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
        """Verify the exception when the timestamp array and the input signal length are inconsistent"""
        time = np.arange(100)
        signal = np.random.randn(125)

        ceemdan = CEEMDAN()
        with self.assertRaises(ValueError):
            ceemdan.fit_transform(signal=signal, time=time)

    def test_trend(self) -> None:
        """Determine the input of a single trend signal"""
        ceemdan = CEEMDAN()

        # Creating timestamps and signals with only a trend component
        time = np.arange(0, 1, 0.01)
        signal = 2 * time

        # Execute the signal decomposition algorithm to obtain the intrinsic mode functions
        IMFs = ceemdan.fit_transform(signal=signal, time=time)
        self.assertEqual(first=IMFs.shape[0], second=1, msg="Expecting single IMF")
        self.assertTrue(np.allclose(signal, IMFs[0]))

    def test_single_imf(self) -> None:
        """Determine the input of a single eigenmode function"""
        ceemdan = CEEMDAN()

        # Create an array of timestamps
        time = np.arange(0, 1, 0.001)

        # Creating a cosine signal
        cosine = np.cos(2 * np.pi * 4 * time)

        # Determine the input of a single cosine function
        IMFs = ceemdan.fit_transform(signal=cosine.copy(), time=time)
        self.assertEqual(first=IMFs.shape[0], second=1, msg="Expecting single IMF!")

        # Determine the numerical difference between input and output
        diff = np.allclose(IMFs[0], cosine)
        self.assertTrue(diff, "Expecting 1st IMF to be cos(8 * pi * t)")

        # Create a trend component of the input
        trend = 3 * (time - 0.5)

        # Determine cosine and trend component input
        IMFs = ceemdan.fit_transform(signal=trend.copy() + cosine.copy(), time=time)
        self.assertEqual(
            first=IMFs.shape[0], second=2, msg="Expecting two IMF of cosine and trend!"
        )

        # Further determine the numerical difference between the two modal outputs
        diff_cosine = np.allclose(IMFs[0], cosine, atol=0.2)
        self.assertTrue(diff_cosine, "Expecting 1st IMF to be cosine")
        diff_trend = np.allclose(IMFs[1], trend, atol=0.2)
        self.assertTrue(diff_trend, "Expecting 2nd IMF to be trend")

    # def test_spline_kind(self) -> None:
    #     """Verify that all interpolation algorithms in the EMD algorithm can run normally"""
    #
    #     # Create two test signals with different components
    #     time = np.arange(0, 1, 0.01)
    #     cosine = np.cos(2 * np.pi * 4 * time)
    #     trend = 3 * (time - 0.1)
    #     signal = cosine.copy() + trend.copy()
    #
    #     for spline_kind in [
    #         "akima",
    #         "cubic",
    #         "pchip",
    #         "cubic_hermite",
    #         "slinear",
    #         "quadratic",
    #         "linear",
    #     ]:
    #         # Traverse all interpolation algorithms and create instances
    #         ceemdan = CEEMDAN(spline_kind=spline_kind)
    #
    #         # Execute the signal decomposition algorithm and determine the output results
    #         IMFs = ceemdan.fit_transform(signal=signal, time=time)
    #
    #         # Determine whether the number of IMFs meets the requirements
    #         self.assertEqual(
    #             first=IMFs.shape[0],
    #             second=2,
    #             msg=f"Expecting two IMF of cosine and trend when the `spline_kind` is {spline_kind}",
    #         )
    #
    #         # Further determine the numerical difference between the two modal outputs
    #         diff_cosine = np.allclose(IMFs[0], cosine, atol=0.3)
    #         self.assertTrue(
    #             diff_cosine,
    #             msg=f"Expecting 1st IMF to be cosine when the `spline_kind` is {spline_kind}",
    #         )
    #         diff_trend = np.allclose(IMFs[1], trend, atol=0.3)
    #         self.assertTrue(
    #             diff_trend,
    #             msg=f"Expecting 2nd IMF to be trend when the `spline_kind` is {spline_kind}",
    #         )
    #
    # def test_wrong_spline_kind(self) -> None:
    #     """Verify that incorrect interpolation type input will cause an exception"""
    #     spline_kind = "wrong"
    #
    #     # Creating random input
    #     time = np.arange(10)
    #     signal = np.random.randn(10)
    #
    #     # Start validating the wrong interpolation type
    #     with self.assertRaises(ValueError):
    #         ceemdan = CEEMDAN(spline_kind=spline_kind)
    #         ceemdan.fit_transform(signal=signal, time=time)

    # def test_extrema_detection(self) -> None:
    #     """Verify that all extreme point detection algorithms in the EMD algorithm can run normally"""
    #     # Create two test signals with different components
    #     time = np.arange(0, 1, 0.01)
    #     cosine = np.cos(2 * np.pi * 4 * time)
    #     trend = 3 * (time - 0.1)
    #     signal = cosine.copy() + trend.copy()
    #
    #     for extrema_detection in ["parabol", "simple"]:
    #         # Traverse all interpolation algorithms and create instances
    #         ceemdan = CEEMDAN(extrema_detection=extrema_detection)
    #
    #         # Execute the signal decomposition algorithm and determine the output results
    #         IMFs = ceemdan.fit_transform(signal=signal, time=time)
    #
    #         # Determine whether the number of IMFs meets the requirements
    #         self.assertEqual(
    #             first=IMFs.shape[0],
    #             second=2,
    #             msg=f"Expecting two IMF of cosine and trend when the `spline_kind` is {extrema_detection}",
    #         )
    #
    #         # Further determine the numerical difference between the two modal outputs
    #         diff_cosine = np.allclose(IMFs[0], cosine, atol=0.3)
    #         self.assertTrue(
    #             diff_cosine,
    #             msg=f"Expecting 1st IMF to be cosine when the `extrema_detection` is {extrema_detection}",
    #         )
    #         diff_trend = np.allclose(IMFs[1], trend, atol=0.3)
    #         self.assertTrue(
    #             diff_trend,
    #             msg=f"Expecting 2nd IMF to be trend when the `extrema_detection` is {extrema_detection}",
    #         )
    #
    # def test_wrong_extrema_detection(self) -> None:
    #     """Verify whether an incorrect extreme point detection type input will cause an exception"""
    #     extrema_detection = "wrong"
    #
    #     # Creating random input
    #     time = np.arange(10)
    #     signal = np.random.randn(10)
    #
    #     # Start verification of wrong extreme point detection type
    #     with self.assertRaises(ValueError):
    #         ceemdan = CEEMDAN(extrema_detection=extrema_detection)
    #         ceemdan.fit_transform(signal=signal, time=time)

    def test_max_iteration_flag(self) -> None:
        """The maximum number of iterations to validate the model"""
        # Creating random signals for verification
        signal = np.random.random(200)
        ceemdan = CEEMDAN()
        ceemdan.MAX_ITERATION = 10
        ceemdan.FIXE = 20
        IMFs = ceemdan.fit_transform(signal)

        # There's not much to test, except that it doesn't fail.
        # With low MAX_ITERATION value for random signal it's
        # guaranteed to have at least 2 IMFs.
        self.assertTrue(IMFs.shape[0] > 1)

    # def test_get_imfs_and_residue(self) -> None:
    #     """Verify whether the intrinsic mode function and trend component can be obtained normally after decomposition"""
    #     signal = np.random.random(200)
    #     ceemdan = CEEMDAN(**{"MAX_ITERATION": 10, "FIXE": 20})
    #     all_imfs = ceemdan(signal, max_imfs=3)
    #
    #     imfs, residue = ceemdan.get_imfs_and_residue()
    #     self.assertEqual(
    #         all_imfs.shape[0], imfs.shape[0] + 1, msg="Compare number of components"
    #     )
    #     self.assertTrue(
    #         np.array_equal(all_imfs[:-1], imfs),
    #         msg="Shouldn't matter where imfs are from",
    #     )
    #     self.assertTrue(
    #         np.array_equal(all_imfs[-1], residue),
    #         msg="Residue, if any, is the last row",
    #     )

    def test_get_imfs_and_residue_without_running(self) -> None:
        """Verify that the output can be obtained when the algorithm is not executed"""
        ceemdan = CEEMDAN()
        with self.assertRaises(ValueError):
            # Since the decomposition process is not performed,
            # it is reasonable to not be able to obtain IMFs and residual results.
            _, _ = ceemdan.get_imfs_and_residue()

    def test_get_imfs_and_trend(self) -> None:
        """Verify whether the intrinsic mode function and trend component can be obtained normally after decomposition"""
        # Creating Algorithm Examples and Test Signals
        ceemdan = CEEMDAN()
        time = np.linspace(0, 2 * np.pi, 100)
        expected_trend = 5 * time
        signal = (
            2 * np.sin(4.1 * 6.28 * time)
            + 1.2 * np.cos(7.4 * 6.28 * time)
            + expected_trend
        )

        # Execute the signal decomposition algorithm
        IMFs = ceemdan(signal)
        # Try to get the trend component
        imfs, trend = ceemdan.get_imfs_and_trend()

        # Further numerical verification of the trend component
        onset_trend = trend - trend.mean()
        onset_expected_trend = expected_trend - expected_trend.mean()
        self.assertEqual(IMFs.shape[0], imfs.shape[0], "Compare number of components")
        # self.assertTrue(
        #     np.array_equal(IMFs[:-1], imfs), "Shouldn't matter where imfs are from"
        # )
        # self.assertTrue(
        #     np.allclose(onset_trend, onset_expected_trend, rtol=0.1, atol=0.5),
        #     "Extracted trend should be close to the actual trend",
        # )


if __name__ == "__main__":
    unittest.main()
