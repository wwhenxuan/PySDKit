# -*- coding: utf-8 -*-
"""
Created on 2025/01/12 12:24:54
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
not done!!!
"""
import numpy as np

from scipy.signal import find_peaks
from scipy.interpolate import interp1d


class ITD(object):
    """
    ITD: Intrinsic Time-Scale Decomposition
    H=itd(x); returns the returns proper rotation components(PRC) and residual signal corresponding to the ITD of X
    Frei, M. G., & Osorio, I. (2007, February).
    Intrinsic time-scale decomposition: time-frequency-energy analysis and real-time filtering of non-stationary signals.
    In Proceedings of the Royal Society of London A: Mathematical, Physical and  Engineering Sciences
    (Vol. 463, No. 2078, pp. 321-342). The Royal Society.
    MATLAB Link: https://www.mathworks.com/matlabcentral/fileexchange/69380-intrinsic-time-scale-decomposition-itd
    """

    def __init__(self, N_max: int = 10):
        self.N_max = N_max

    def stop_iter(self, x, counter, E_x):
        """Stop the main iteration loop"""
        if counter > self.N_max:
            return True

        Exx = np.sum(x ** 2)
        if Exx <= 0.01 * E_x:
            return True

        # 获取输入信号的峰值
        pks1, _ = find_peaks(x)  # 极大值位置
        pks2, _ = find_peaks(-x)  # 极小值位置
        pks = np.union1d(pks1, pks2)  # 所有的极值点

        if len(pks) <= 7:
            return True

        # EOF
        return False

    def itd_baseline_extract(self, x):
        """This function is to calculate the baseline L of the signal x."""
        length = len(x)
        t = np.arange(0, length)
        alpha = 0.5

        idx_max, _ = find_peaks(x)
        val_max = x[idx_max]

        idx_min, _ = find_peaks(-x)
        val_min = x[idx_min]

        idx_cb = np.union1d(idx_max, idx_min)

        # Check the boundary conditions

        # Left side
        if np.min(idx_max) < np.min(idx_min):
            idx_min = np.append(idx_max[0], idx_min)
            val_min = np.append(val_min[0], val_min)
        elif np.min(idx_max) > np.min(idx_min):
            idx_max = np.append(idx_min[0], idx_max)
            val_max = np.append(val_max[0], val_max)

        # Right side
        if np.max(idx_max) > np.max(idx_min):
            idx_min = np.append(idx_min, idx_max[-1])
            val_min = np.append(val_min, val_min[-1])
        elif np.max(idx_max) < np.max(idx_min):
            idx_max = np.append(idx_max, idx_min[-1])
            val_max = np.append(val_max, val_max[-1])

        # Compute the LK points
        # 创建插值函数对象，使用线性插值
        Max_line_interp = interp1d(idx_max, val_max, kind='linear', fill_value="extrapolate")
        Min_line_interp = interp1d(idx_min, val_min, kind='linear', fill_value="extrapolate")

        # 计算插值点的值
        Max_line = Max_line_interp(t)
        Min_line = Min_line_interp(t)

        LK1 = alpha * Max_line[idx_min] + val_min * (1 - alpha)
        LK2 = alpha * Min_line[idx_max] + val_max * (1 - alpha)

        LK1 = np.append(idx_min.reshape(-1, 1), LK1.reshape(-1, 1))
        LK2 = np.append(idx_max.reshape(-1, 1), LK2.reshape(-1, 1))

        LK = np.vstack((LK1, LK2)).T

        LK_col_2 = np.argsort(LK[:, 0])
        LK_sorted = LK[LK_col_2, :]
        LK = LK_sorted[1:-1, :]

        LK = np.vstack((np.array([0, LK[0, 1]]), LK, np.array([length - 1, LK[-1, 1]])))

        # Compute the Lt points
        idx_Xk = np.hstack((0, idx_cb, length - 1)).reshape([len(idx_cb) + 2])

        L = np.zeros(length)
        for i in range(0, len(idx_Xk) - 1):
            for j in range(idx_Xk[i], idx_Xk[i + 1]):
                kij = (LK[i + 1, 1] - LK[i, 1]) / (x[idx_Xk[i + 1]] - x[idx_Xk[i]])  # compute the slope K
                L[j] = LK[i, 1] + kij * (x[j] - x[idx_Xk[i]])

        H = x - L
        print(H)
        return L, H

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Input x is a 1D numpy signal and return the decomposition results."""
        H = []
        E_x = np.sum(x ** 2)
        counter = 0

        while True:
            counter = counter + 1
            L1, H1 = self.itd_baseline_extract(x)
            H.append(H1)
            STOP = self.stop_iter(x, counter, E_x)
            if STOP is True:
                H.append(L1)
                break
            x = L1

        H = np.vstack(H)

        return H


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    fs = 1000
    N = 3000
    t = np.arange(0, N) / fs
    x1 = (2 + np.cos(2 * np.pi * 0.5 * t)) * np.cos(2 * np.pi * 5 * t + 15 * t ** 2)
    x2 = np.cos(2 * np.pi * 2 * t)

    signal = x1 + x2

    itd = ITD(N_max=100)
    IMFs = itd.fit_transform(signal)
    row, length = IMFs.shape
    print(IMFs.shape)

    fig, ax = plt.subplots(row + 1, 1, figsize=(10, 8))
    ax[0].plot(t, signal)
    for index, i in enumerate(range(row), 1):
        ax[index].plot(t[:300], IMFs[i, :300])

    fig.show()
