# -*- coding: utf-8 -*-
"""
Created on 2025/07/18 11:17:41
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import unittest

from pysdkit.data import (
    test_emd,
    test_univariate_signal,
    test_univariate_nonlinear_chip,
    test_univariate_duffing,
    test_univariate_logistic_am,
    test_univariate_gaussamp_quadfm,
    test_univariate_cubic_quad,
)


class Univariate_Signal_Generator_Test(unittest.TestCase):
    """Verify various test modules that generate one-dimensional signals"""

    def test_emd(self) -> None:
        """Verify the data generator that generates EMD test X"""
        for duration in range(1, 6):
            # Traversal cycle time point information
            for sampling_rate in [100, 200, 500]:
                # Traverse sampling frequency information
                for noise_level in [0, 0.1, 0.5]:
                    # Traversal noise value size
                    time, signal = test_emd(
                        duration=duration,
                        sampling_rate=sampling_rate,
                        noise_level=noise_level,
                    )
                    # target length of the generated signal
                    target = duration * sampling_rate
                    # test the time array
                    self.assertEqual(first=target, second=len(signal))

    def test_univariate_signal(self) -> None:
        """Verify data generator that produces unary signals."""
        for case in [1, 2, 3]:
            # Traversing different test cases
            for duration in range(1, 6):
                # Traversal cycle time point information
                for sampling_rate in [100, 200, 500]:
                    # Traverse sampling frequency information
                    time, signal = test_univariate_signal(
                        case=case, duration=duration, sampling_rate=sampling_rate
                    )
                    # target length of the generated signal
                    target = duration * sampling_rate
                    # test the time array
                    self.assertEqual(first=target, second=len(signal))

    def test_univariate_nonlinear_chip(self) -> None:
        """Verify the data generator that generates a univariate nonlinear frequency modulation signal."""
        for case in [1, 2]:
            # Traversing different test cases
            for duration in range(1, 6):
                # Traversal cycle time point information
                for sampling_rate in [100, 200, 500]:
                    # Traverse sampling frequency information
                    for noise_level in [0, 0.1, 0.5]:
                        # Traversal noise value size
                        time, signal = test_univariate_nonlinear_chip(
                            case=case,
                            duration=duration,
                            sampling_rate=sampling_rate,
                            noise_level=noise_level,
                        )
                        # target length of the generated signal
                        target = duration * sampling_rate
                        # test the time array
                        self.assertEqual(first=target, second=len(signal))

    def test_univariate_duffing(self) -> None:
        """Test the generation of a Gaussian-modulated quadratic chirp signal"""
        for duration in range(1, 6):
            # Traversal cycle time point information
            for sampling_rate in [100, 200, 500]:
                # Traverse sampling frequency information
                for noise_level in [0, 0.1, 0.5]:
                    # Traversal noise value size
                    time, signal = test_univariate_duffing(
                        duration=duration,
                        sampling_rate=sampling_rate,
                        noise_level=noise_level,
                    )
                    # target length of the generated signal
                    target = duration * sampling_rate
                    # test the time array
                    self.assertEqual(first=target, second=len(signal))

    def test_univariate_logistic_am(self) -> None:
        """Test the generation of a softening Duffing-type ODE"""
        for duration in range(1, 6):
            # Traversal cycle time point information
            for sampling_rate in [100, 200, 500]:
                # Traverse sampling frequency information
                for noise_level in [0, 0.1, 0.5]:
                    # Traversal noise value size
                    time, signal = test_univariate_logistic_am(
                        duration=duration,
                        sampling_rate=sampling_rate,
                        noise_level=noise_level,
                    )
                    # target length of the generated signal
                    target = duration * sampling_rate
                    # test the time array
                    self.assertEqual(first=target, second=len(signal))

    def test_univariate_gaussamp_quadfm(self) -> None:
        """Test the generation of a carrier modulated by chaotic logistic map"""
        for duration in range(1, 6):
            # Traversal cycle time point information
            for sampling_rate in [100, 200, 500]:
                # Traverse sampling frequency information
                for noise_level in [0, 0.1, 0.5]:
                    # Traversal noise value size
                    time, signal = test_univariate_gaussamp_quadfm(
                        duration=duration,
                        sampling_rate=sampling_rate,
                        noise_level=noise_level,
                    )
                    # target length of the generated signal
                    target = duration * sampling_rate
                    # test the time array
                    self.assertEqual(first=target, second=len(signal))

    def test_univariate_cubic_quad(self) -> None:
        """Test the generation of a quadratic and cubic coupling"""
        for duration in range(1, 6):
            # Traversal cycle time point information
            for sampling_rate in [100, 200, 500]:
                # Traverse sampling frequency information
                for noise_level in [0, 0.1, 0.5]:
                    # Traversal noise value size
                    time, signal = test_univariate_cubic_quad(
                        duration=duration,
                        sampling_rate=sampling_rate,
                        noise_level=noise_level,
                    )
                    # target length of the generated signal
                    target = duration * sampling_rate
                    # test the time array
                    self.assertEqual(first=target, second=len(signal))


if __name__ == "__main__":
    unittest.main()
