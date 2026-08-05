# -*- coding: utf-8 -*-
"""
Created on 2025/01/31 21:35:18
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import argrelextrema


class LMD(object):
    """
    Local Mean Decomposition (classical Smith LMD).

    Decomposes a signal into Product Functions (PFs), each being a slowly
    varying envelope multiplied by a pure FM carrier.

    Smith J.S. The local mean decomposition and its application to EEG
    perception data. Journal of the Royal Society Interface, 2(5):443–454, 2005.
    https://doi.org/10.1098/rsif.2005.0058

    Python reference (moving-average LMD):
    https://github.com/shownlin/PyLMD

    Note: the companion MATLAB ``eoe_lmd.m`` / Jia et al. DSP 2019 paper
    implement **EOE-LMD** (empirical optimal envelopes), which is a different
    envelope construction; this class follows classical Smith / PyLMD LMD.
    """

    def __init__(
        self,
        K: int = 5,
        endpoints: bool = True,
        max_smooth_iter: int = 12,
        max_envelope_iter: int = 200,
        envelope_epsilon: float = 0.01,
        convergence_epsilon: float = 0.01,
        min_extrema: int = 5,
    ) -> None:
        """
        :param K: maximum number of PFs (excluding residue)
        :param endpoints: treat signal endpoints as pseudo-extrema
        :param max_smooth_iter: max moving-average smoothing iterations
        :param max_envelope_iter: max inner sifting iterations per PF
        :param envelope_epsilon: stop when mean |1 - a(t)| is below this
        :param convergence_epsilon: stop when mean |s - t| is below this
        :param min_extrema: minimum extrema count to continue outer loop
        """
        self.K = int(K)
        self.endpoints = bool(endpoints)
        self.max_smooth_iter = int(max_smooth_iter)
        self.max_envelope_iter = int(max_envelope_iter)
        self.envelope_epsilon = float(envelope_epsilon)
        self.convergence_epsilon = float(convergence_epsilon)
        self.min_extrema = int(min_extrema)

    def __call__(self, signal: np.ndarray, K: Optional[int] = None) -> np.ndarray:
        """Allow instances to be called like functions."""
        return self.fit_transform(signal=signal, K=K)

    def __str__(self) -> str:
        return "Local Mean Decomposition (LMD)"

    def is_monotonous(self, signal: np.ndarray) -> bool:
        """Whether the signal is a (non-strict) monotone sequence."""
        if len(signal) <= 0:
            return True
        return self.is_monotonous_increase(signal) or self.is_monotonous_decrease(
            signal
        )

    @staticmethod
    def is_monotonous_increase(signal: np.ndarray) -> bool:
        """Whether the input signal is monotonically non-decreasing."""
        y0 = signal[0]
        for y1 in signal:
            if y1 < y0:
                return False
            y0 = y1
        return True

    @staticmethod
    def is_monotonous_decrease(signal: np.ndarray) -> bool:
        """Whether the input signal is monotonically non-increasing."""
        y0 = signal[0]
        for y1 in signal:
            if y1 > y0:
                return False
            y0 = y1
        return True

    def find_extrema(self, signal: np.ndarray) -> np.ndarray:
        """Find all local extreme points of the signal (optionally + endpoints)."""
        n = len(signal)
        extrema = np.append(
            argrelextrema(signal, np.greater)[0], argrelextrema(signal, np.less)[0]
        )
        extrema.sort()

        if self.endpoints:
            # Guard empty extrema (e.g. constant / strictly monotone segments)
            if extrema.size == 0:
                extrema = np.array([0, n - 1], dtype=int)
            else:
                if extrema[0] != 0:
                    extrema = np.insert(extrema, 0, 0)
                if extrema[-1] != n - 1:
                    extrema = np.append(extrema, n - 1)

        return extrema.astype(int)

    def moving_average_smooth(self, signal: np.ndarray, window: int) -> np.ndarray:
        """Smooth a square local-mean / envelope sequence by weighted MA."""
        n = len(signal)

        # at least one nearby sample is needed for average
        if window < 3:
            window = 3

        # odd window for symmetry
        if (window % 2) == 0:
            window += 1

        half = window // 2
        weight = np.array(list(range(1, half + 2)) + list(range(half, 0, -1)))
        assert len(weight) == window

        smoothed = np.asarray(signal, dtype=float).copy()

        for _ in range(self.max_smooth_iter):
            head = list()
            tail = list()
            w_num = half
            for i in range(half):
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

            smoothed = np.convolve(smoothed, weight, mode="same")
            smoothed[half:-half] = smoothed[half:-half] / sum(weight)

            w_num = half
            for i in range(half):
                smoothed[i] = sum(head[i] * weight[w_num:]) / sum(weight[w_num:])
                smoothed[-(i + 1)] = sum(tail[i] * weight[:-w_num]) / sum(
                    weight[:-w_num]
                )
                w_num -= 1
            # stop when no consecutive identical samples remain (PyLMD / Smith)
            if self.is_smooth(smoothed, n):
                break
        return smoothed

    @staticmethod
    def is_smooth(signal: np.ndarray, n: int) -> bool:
        """True iff no two consecutive samples are identical (smoothed enough)."""
        for x in range(1, n):
            if signal[x] == signal[x - 1]:
                return False
        return True

    def local_mean_and_envelope(
        self, signal: np.ndarray, extrema: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build local mean / envelope square signals and smooth them.

        Between successive extrema ``n_k``, ``n_{k+1}``:

        .. math::

            m_k = (x(n_k)+x(n_{k+1}))/2,\\quad
            a_k = |x(n_k)-x(n_{k+1})|/2
        """
        n = len(signal)
        k = len(extrema)
        assert 1 < k <= n

        mean = []
        enve = []
        prev_mean = (signal[extrema[0]] + signal[extrema[1]]) / 2
        prev_enve = abs(signal[extrema[0]] - signal[extrema[1]]) / 2
        e = 1
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

        window = max(np.diff(extrema)) // 3
        return (
            np.array(mean),
            self.moving_average_smooth(mean, window),
            np.array(enve),
            self.moving_average_smooth(enve, window),
        )

    def extract_product_function(self, signal: np.ndarray) -> np.ndarray:
        """Extract one Product Function (inner sifting loop)."""
        s = np.asarray(signal, dtype=float).copy()
        n = len(signal)
        envelopes = []

        def component() -> np.ndarray:
            # PF = (prod_j a_j) * s_n
            c = s
            for env in envelopes:
                c = c * env
            return c

        for _ in range(self.max_envelope_iter):
            extrema = self.find_extrema(s)
            if len(extrema) <= 3:
                break

            _m0, m, _a0, a = self.local_mean_and_envelope(s, extrema)
            # avoid non-positive envelope (division)
            a = np.asarray(a, dtype=float)
            a[a <= 0] = 1.0 - 1e-4

            h = s - m
            t = h / a

            # pure FM: envelope close to 1
            err = float(np.sum(np.abs(1.0 - a)) / n)
            if err <= self.envelope_epsilon:
                break
            # modulation convergence
            err = float(np.sum(np.abs(s - t)) / n)
            if err <= self.convergence_epsilon:
                break
            envelopes.append(a)
            s = t

        return component()

    def fit_transform(self, signal: np.ndarray, K: Optional[int] = None) -> np.ndarray:
        """
        Decompose a 1-D signal with Local Mean Decomposition.

        :param signal: time-domain signal ``(N,)``
        :param K: optional override of maximum PF count (does not mutate ``self.K``)
        :return: array ``(n_pf + 1, N)`` — PFs from high to low frequency, last row = residue
        """
        x = np.asarray(signal, dtype=float).ravel()
        if x.ndim != 1:
            raise ValueError("signal must be a 1-D array")
        if x.size < 4:
            raise ValueError("signal length must be >= 4")

        max_pf = int(self.K if K is None else K)
        pf = []
        residue = x.copy()

        while (
            (len(pf) < max_pf)
            and (not self.is_monotonous(residue))
            and (len(self.find_extrema(residue)) >= self.min_extrema)
        ):
            component = self.extract_product_function(residue)
            # avoid zero / NaN components stalling the loop
            if not np.all(np.isfinite(component)):
                break
            residue = residue - component
            pf.append(component)

        pf.append(residue)
        return np.asarray(pf, dtype=float)
