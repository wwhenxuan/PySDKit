# -*- coding: utf-8 -*-
"""
Created on 2025/01/31 21:35:18
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from scipy.signal import argrelextrema

from typing import Optional


class LMD(object):
    """
    Local Mean Decomposition

    Jia, Linshan, et al. “The Empirical Optimal Envelope and Its Application to Local Mean Decomposition.”
    Digital Signal Processing, vol. 87, Elsevier BV, Apr. 2019, pp. 166–77, doi:10.1016/j.dsp.2019.01.024.

    Python code: https://github.com/shownlin/PyLMD

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/107829-local-mean-decomposition?s_tid=srchtitle
    """

    def __init__(
        self,
        K: int = 5,
        endpoints: bool = True,
        max_smooth_iter: int = 15,
        max_envelope_iter: int = 200,
        envelope_epsilon: float = 0.01,
        convergence_epsilon: float = 0.01,
        min_extrema: int = 5,
    ) -> None:
        """
        :param K: the maximum number of IFMs to be decomposed
        :param endpoints: whether to treat the endpoint of the signal as a pseudo-extreme point
        :param max_smooth_iter: maximum number of iterations of moving average algorithm
        :param max_envelope_iter: maximum number of iterations when separating local envelope signals
        :param envelope_epsilon: terminate processing when obtaining pure FM signal
        :param convergence_epsilon: terminate processing when modulation signal converges
        :param min_extrema: minimum number of local extrema
        """
        self.K = K
        self.endpoints = endpoints
        self.max_smooth_iter = max_smooth_iter
        self.max_envelope_iter = max_envelope_iter
        self.envelope_epsilon = envelope_epsilon
        self.convergence_epsilon = convergence_epsilon
        self.min_extrema = min_extrema

    def __call__(self, signal: np.ndarray, K: Optional[int] = None) -> np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal, K=K)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Local Mean Decomposition (LMD)"

    def is_monotonous(self, signal: np.ndarray) -> bool:
        """Determine whether the signal is a (non-strict) monotone sequence"""
        # This method is used to determine the termination condition of the iterative loop
        if len(signal) <= 0:
            # Sequence length less than 0 cannot be judged
            return True  # Debugging Errors
        else:
            # Returns a test for whether it is monotonic
            return self.is_monotonous_increase(signal) or self.is_monotonous_decrease(
                signal
            )

    @staticmethod
    def is_monotonous_increase(signal: np.ndarray) -> bool:
        """Determine whether the input signal is monotonically increasing"""
        # Take the first sample
        y0 = signal[0]

        for y1 in signal:
            # Traverse the entire signal
            if y1 < y0:
                return False
            y0 = y1
        return True

    @staticmethod
    def is_monotonous_decrease(signal: np.ndarray) -> bool:
        """Determine whether the input signal is monotonically decreasing"""
        # Take the first sample
        y0 = signal[0]

        for y1 in signal:
            # Traverse the entire signal
            if y1 > y0:
                return False
            y0 = y1
        return True

    def find_extrema(self, signal: np.ndarray) -> np.ndarray:
        """Find all local extreme points of the signal"""

        # The number of points can be used to determine whether further decomposition can be performed
        n = len(signal)

        # Determine the extreme points in the input sequence using scipy's argrelextrema method
        extrema = np.append(
            argrelextrema(signal, np.greater)[0], argrelextrema(signal, np.less)[0]
        )

        # Sort extreme points
        extrema.sort()

        if self.endpoints:
            # Whether to consider the two endpoints of the input signal as extreme points
            if extrema[0] != 0:
                extrema = np.insert(extrema, 0, 0)
            if extrema[-1] != n - 1:
                extrema = np.append(extrema, n - 1)

        return extrema

    def moving_average_smooth(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Smooth the input signal by sliding average"""

        n = len(signal)  # The length of the input signal

        # at least one nearby sample is needed for average
        if window < 3:
            window = 3

        # adjust length of sliding window to an odd number for symmetry
        if (window % 2) == 0:
            window += 1

        half = window // 2

        # Initialize the parameters of the sliding average decomposition
        weight = np.array(list(range(1, half + 2)) + list(range(half, 0, -1)))
        assert (
            len(weight) == window
        )  # Make sure the parameter length is the size of the window

        smoothed = signal

        # Start Iteration
        for _ in range(self.max_smooth_iter):
            head = list()
            tail = list()
            w_num = half
            for i in range(half):
                # Process the head and tail separately
                head.append(
                    np.array(
                        [smoothed[j] for j in range(i - (half - w_num), i + half + 1)]
                    )
                )
                tail.append(
                    np.flip(
                        [
                            smoothed[-(j + 1)]
                            for j in range(i - (half - w_num), i + half + 1)
                        ]
                    )
                )
                w_num -= 1

            # Smoothing through convolution
            smoothed = np.convolve(smoothed, weight, mode="same")
            smoothed[half:-half] = smoothed[half:-half] / sum(weight)

            w_num = half
            for i in range(half):
                smoothed[i] = sum(head[i] * weight[w_num:]) / sum(weight[w_num:])
                smoothed[-(i + 1)] = sum(tail[i] * weight[:-w_num]) / sum(
                    weight[:-w_num]
                )
                w_num -= 1
            if self.is_smooth(smoothed, n):
                # When the signal has been processed to a smooth state, the iteration stops
                break
        return smoothed  # Returns the smoothed signal

    @staticmethod
    def is_smooth(signal: np.ndarray, n: int) -> bool:
        """Determine whether a signal is smooth"""
        for x in range(1, n):
            # If there are two consecutive unequal points, it means it is not smooth
            if signal[x] == signal[x - 1]:
                return False
        return True

    def local_mean_and_envelope(self, signal: np.ndarray, extrema):
        """Calculate the local mean function and local envelope function according to the location of the extreme points"""
        n = len(signal)
        k = len(extrema)
        assert 1 < k <= n
        # construct square signal
        mean = []
        enve = []
        prev_mean = (signal[extrema[0]] + signal[extrema[1]]) / 2
        prev_enve = abs(signal[extrema[0]] - signal[extrema[1]]) / 2
        e = 1
        # Start calculating the local mean and envelope spectrum
        # This process is the core of the algorithm
        for x in range(n):
            if (x == extrema[e]) and (e + 1 < k):
                next_mean = (signal[extrema[e]] + signal[extrema[e + 1]]) / 2
                mean.append((prev_mean + next_mean) / 2)
                prev_mean = next_mean
                next_enve = abs(signal[extrema[e]] - signal[extrema[e + 1]]) / 2
                enve.append((prev_enve + next_enve) / 2)
                prev_enve = next_enve
                e += 1
            else:
                mean.append(prev_mean)
                enve.append(prev_enve)

        # smooth square signal
        window = max(np.diff(extrema)) // 3
        return (
            np.array(mean),
            self.moving_average_smooth(mean, window),
            np.array(enve),
            self.moving_average_smooth(enve, window),
        )

    def extract_product_function(self, signal: np.ndarray) -> np.ndarray:
        """Perform one time local mean decomposition algorithm"""
        s = signal
        n = len(signal)

        # Used to store the results of this envelope spectrum analysis
        envelopes = []

        def component():
            # Calculate PF，using PF_i(t) = a_i(t)* s_in()，其中a_i = a_i0 * a_i1 * ... * a_in
            c = s
            for e in envelopes:
                c = c * e
            return c

        # Start to separate the envelope signal through an iterative method
        for _ in range(self.max_envelope_iter):
            # First, we can determine whether early stopping is possible by observing the extreme points of the signal
            extrema = self.find_extrema(s)
            if len(extrema) <= 3:
                break

            # Get the local envelope function of the input signal
            _m0, m, _a0, a = self.local_mean_and_envelope(s, extrema)

            for i in range(len(a)):
                if a[i] <= 0:
                    a[i] = 1 - 1e-4

            # subtracted from the original data
            h = s - m

            # amplitude demodulated by dividing a
            t = h / a

            # Terminate processing when obtaining pure FM signal.
            err = sum(abs(1 - a)) / n
            if err <= self.envelope_epsilon:
                break
            # Terminate processing when modulation signal converges.
            err = sum(abs(s - t)) / n
            if err <= self.convergence_epsilon:
                break
            envelopes.append(a)
            s = t

        return component()

    def fit_transform(self, signal: np.ndarray, K: Optional[int] = None) -> np.ndarray:
        """
        Signal decomposition using Local Mean Decomposition (LMD) algorithm

        :param signal: the time domain signal (1D numpy array)  to be decomposed
        :param K: the maximum number of IFMs to be decomposed
        :return: IMFs
        """
        if K is not None:
            # Tuning Hyperparameters
            self.K = K
        pf = []

        # until the residual function is close to a monotone function
        residue = signal[:]

        # Decomposition by loop iteration
        while (len(pf) < self.K) and (not self.is_monotonous(residue)):
            # Ensure that the number of decompositions has the required sign and the remaining residuals are non-monotonic
            if len(self.find_extrema(residue)) < self.min_extrema:
                # The number of extreme points of the remaining signal must be greater than the minimum specified number
                break

            # Get the result of this local mean decomposition
            component = self.extract_product_function(residue)

            # Each iteration subtracts the decomposed part from the original signal
            residue = residue - component

            # Record the results of this decomposition
            pf.append(component)

        # Incorporate the residuals left over from the decomposition
        pf.append(residue)

        return np.array(pf)
