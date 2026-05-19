# -*- coding: utf-8 -*-
"""
Created on 2025/01/12 12:24:54
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from scipy.signal import find_peaks
from scipy.interpolate import interp1d


class ITD(object):
    """
    ITD: Intrinsic Time-Scale Decomposition

    H=_itd(x); returns the returns proper rotation components(PRC) and residual signal corresponding to the ITD of X
    Frei, M. G., & Osorio, I. (2007, February).
    Intrinsic time-scale decomposition: time-frequency-energy analysis and real-time filtering of non-stationary signals.
    In Proceedings of the Royal Society of London A: Mathematical, Physical and  Engineering Sciences
    (Vol. 463, No. 2078, pp. 321-342). The Royal Society.
    MATLAB Link: https://www.mathworks.com/matlabcentral/fileexchange/69380-intrinsic-time-scale-decomposition-itd
    """

    def __init__(self, N_max: int = 10):
        self.N_max = N_max

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Intrinsic Time-Scale Decomposition (ITD)"

    def stop_iter(self, x, counter, E_x):
        """Determine whether the ITD iteration should stop.

        :param x: current residual signal
        :param counter: number of iterations completed
        :param E_x: energy of the original input signal
        :return: True if decomposition should stop
        """
        if counter > self.N_max:
            return True

        Exx = np.sum(x**2)
        if Exx <= 0.01 * E_x:
            return True

        pks1, _ = find_peaks(x)
        pks2, _ = find_peaks(-x)
        pks = np.union1d(pks1, pks2)

        if len(pks) <= 7:
            return True

        return False

    @staticmethod
    def itd_baseline_extract(x):
        """
        Calculate the baseline L of the signal x using the ITD algorithm.

        The baseline is constructed by computing control points (LK) at each
        extremum as a weighted average (alpha=0.5) of the signal value and the
        interpolated opposite envelope, then piecewise-linearly interpolating
        between consecutive extrema using signal-value-based slopes.

        :param x: input 1D signal array
        :return: tuple (L, H) where L is the baseline and H = x - L is the
                 proper rotation component
        """
        length = len(x)
        t = np.arange(length)
        alpha = 0.5

        idx_max, _ = find_peaks(x)
        val_max = x[idx_max]

        idx_min, _ = find_peaks(-x)
        val_min = x[idx_min]

        if len(idx_max) == 0 or len(idx_min) == 0:
            return x.copy(), np.zeros_like(x)

        idx_cb = np.union1d(idx_max, idx_min)

        # Pad boundary so that idx_max and idx_min span the same index range.
        # When the first (or last) extremum on one side has no counterpart,
        # mirror the opposite extremum's position with the signal value there.
        if idx_max[0] < idx_min[0]:
            idx_min = np.concatenate(([idx_max[0]], idx_min))
            val_min = np.concatenate(([x[idx_max[0]]], val_min))
        elif idx_min[0] < idx_max[0]:
            idx_max = np.concatenate(([idx_min[0]], idx_max))
            val_max = np.concatenate(([x[idx_min[0]]], val_max))

        if idx_max[-1] > idx_min[-1]:
            idx_min = np.concatenate((idx_min, [idx_max[-1]]))
            val_min = np.concatenate((val_min, [x[idx_max[-1]]]))
        elif idx_min[-1] > idx_max[-1]:
            idx_max = np.concatenate((idx_max, [idx_min[-1]]))
            val_max = np.concatenate((val_max, [x[idx_min[-1]]]))

        Max_line_interp = interp1d(
            idx_max, val_max, kind="linear", fill_value="extrapolate"
        )
        Min_line_interp = interp1d(
            idx_min, val_min, kind="linear", fill_value="extrapolate"
        )

        Max_line = Max_line_interp(t)
        Min_line = Min_line_interp(t)

        # LK control points: weighted average at each extremum
        LK1_vals = alpha * Max_line[idx_min] + (1 - alpha) * val_min
        LK2_vals = alpha * Min_line[idx_max] + (1 - alpha) * val_max

        # Build (position, value) pairs and merge
        LK1_pts = np.column_stack((idx_min, LK1_vals))
        LK2_pts = np.column_stack((idx_max, LK2_vals))
        LK = np.vstack((LK1_pts, LK2_pts))

        # Sort by position and remove duplicates at the boundaries
        order = np.argsort(LK[:, 0])
        LK = LK[order]

        # Remove the padded boundary duplicates (first and last are mirrors)
        if len(LK) > 2:
            LK = LK[1:-1]

        # Clamp to signal boundaries
        LK = np.vstack((
            [0, LK[0, 1]],
            LK,
            [length - 1, LK[-1, 1]],
        ))

        # Compute baseline by piecewise-linear interpolation between extrema
        idx_Xk = np.concatenate(([0], idx_cb, [length - 1])).astype(int)

        L = np.zeros(length)
        for i in range(len(idx_Xk) - 1):
            denom = x[idx_Xk[i + 1]] - x[idx_Xk[i]]
            kij = (LK[i + 1, 1] - LK[i, 1]) / denom if denom != 0 else 0.0
            for j in range(idx_Xk[i], idx_Xk[i + 1]):
                L[j] = LK[i, 1] + kij * (x[j] - x[idx_Xk[i]])

        L[length - 1] = LK[-1, 1]

        H = x - L
        return L, H

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """Input x is a 1D numpy signal and return the decomposition results."""
        H = []
        E_x = np.sum(signal**2)
        counter = 0

        while True:
            counter = counter + 1
            L1, H1 = self.itd_baseline_extract(signal)
            H.append(H1)
            STOP = self.stop_iter(signal, counter, E_x)
            if STOP is True:
                H.append(L1)
                break
            signal = L1

        H = np.vstack(H)

        return H


if __name__ == "__main__":
    from pysdkit.data import test_univariate_signal
    from pysdkit.plot import plot_IMFs

    time, signal = test_univariate_signal()

    print(signal.shape)

    itd = ITD()
    imfs = itd.fit_transform(signal=signal)

    plot_IMFs(signal, IMFs=imfs)
