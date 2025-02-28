# -*- coding: utf-8 -*-
"""
Created on 2025/02/01 22:30:40
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from numpy import floating
from scipy.stats import mode

from typing import Any, Optional, Tuple, Union

from pysdkit._faemd.extrema import extrema
from pysdkit.utils import simple_moving_average


class FAEMD(object):
    """
    Fast and Adaptive Empirical Mode Decomposition

    Thirumalaisamy, Mruthun R., and Phillip J. Ansell.
    “Fast and Adaptive Empirical Mode Decomposition for Multidimensional, Multivariate Signals.”
    IEEE Signal Processing Letters, vol. 25, no. 10, Institute of Electrical and Electronics Engineers (IEEE),
    Oct. 2018, pp. 1550–54, doi:10.1109/lsp.2018.2867335.

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd

    also see: `EMD`, `EEMD`, `REMD` and `CEEMDAN`.
    """

    def __init__(self, max_imfs: Optional[int], tol: Optional[float] = None, window_type: Optional[int] = 0) -> None:
        """
        相比于EMD算法
        :param max_imfs:The number of IMFs to be extracted
        :param tol:
        :param window_type:
        """
        self.max_imfs = max_imfs
        if max_imfs < 1:
            raise ValueError("`max_imfs` must be a positive integer")
        self.tol = tol
        #
        self.window_type = window_type
        # 这里需要对这个参数进行进一步的检验

        pass

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Fast and Adaptive Empirical Mode Decomposition (FAEMD)"

    def _get_tol(self, signal: np.ndarray) -> float:
        """Get the tolerance parameter for FAEMD"""
        # 当`tol`变量默认为None时
        if self.tol is None:
            tol = min(np.sqrt(np.mean(signal ** 2)), 1) * 0.001
            return tol
        # 返回用户指定的参数
        return self.tol

    def filter_size1D(self, imax: np.ndarray, imin: np.ndarray):
        """
        To determine the window size for order statistics filtering of a signal.
        The determination of the window size is based on the work of Bhuiyan et al
        :return:
        """

        # 通过差分计算极值点之间的举例
        edge_len_max = np.diff(np.sort(imax))
        edge_len_min = np.diff(np.sort(imin))

        # Window size calculations
        d1 = np.min([np.min(edge_len_max), np.min(edge_len_min)])
        d2 = np.max([np.min(edge_len_max), np.min(edge_len_min)])
        d3 = np.min([np.max(edge_len_max), np.max(edge_len_min)])
        d4 = np.max([np.max(edge_len_max), np.max(edge_len_min)])
        d5 = (d1 + d2 + d3 + d4) / 4
        concat = np.concatenate((edge_len_min, edge_len_max))
        d6 = np.median(concat)
        d7 = mode(concat)[0]

        windows = np.array([d1, d2, d3, d4, d5, d6, d7])

        # making sure w_size is an odd integer
        windows = 2 * np.floor(windows / 2) + 1

        # 遍历窗口数组规整其最小值
        for t in range(7):
            if windows[self.window_type] < 3:
                windows[self.window_type] = 3

        return windows


    def sift(self, H, w_sz):
        """"""

        # Envelope Generation
        Env_max, Env_min = self.OSF(H=H, w_sz=w_sz)

        # padding
        Env_med = env_smoothing(env_max=Env_max, env_min=Env_min, w_sz=w_sz)

        # Subtracting from residue
        H1 = H - Env_med

        return H1


    def OSF(self, H, w_sz):
        """用于生成信号的上下包络谱"""
        # Max envelope
        Max = self.ord_filt1(H, order="max", window_size=w_sz)
        # Min envelope
        Min = self.ord_filt1(H, order="min", window_size=w_sz)

        return Max, Min


    @staticmethod
    def ord_filt1(signal, order, window_size) -> np.ndarray:
        """1-D Rank order filter function"""
        # Pre-processing
        # Original signal size
        shape = signal.shape

        # Removing the singleton dimensions
        signal = np.squeeze(signal)

        # Length of the signal
        L = len(signal)

        # Ensure that the processed signal is always a column vector
        signal = np.reshape(signal, newshape=[L, 1])

        r = int((window_size - 1) / 2)

        # Padding boundaries
        x = np.concatenate([np.flip(signal[:r]), signal, np.flip(signal[-r:])])

        M = x.shape[0]

        y = np.zeros(x.shape)

        # Switch the order
        if order == "max":
            for m in range(r, M - r + 1):
                # Extract a window of size (2r+1) around (m)
                temp = x[(m - r): (m + r)]
                w = np.sort(temp)
                # Select the greatest element
                y[m] = w[-1]
        elif order == "min":
            for m in range(r, M - r + 1):
                # Extract a window of size (2r+1) around (m)
                temp = x[(m - r): (m + r)]
                w = np.sort(temp)
                # Select the smallest element
                y[m] = w[-1]
        else:
            raise ValueError("No such filering operation defined")

        f_signal = y[r: -r]

        # Restoring Signal size
        f_signal = np.reshape(f_signal, newshape=shape)

        return f_signal


    def fit_transform(self, signal: np.ndarray, return_all: bool = False):

        # 获取信号的变量数目和长度
        signal = signal.T
        seq_len, num_vars = signal.shape

        # Initialisations
        imfs = np.zeros(shape=(seq_len, num_vars, self.max_imfs))
        H1 = np.zeros(shape=(seq_len, num_vars))
        mse = np.zeros(num_vars)

        windows = np.zeros(shape=(7, self.max_imfs))

        sift_count = np.zeros(shape=self.max_imfs)

        imf = 0

        # 分解的剩余分量初始化
        Residue = signal.copy()

        # 获取算法停止的容忍度参数
        tol = self._get_tol(signal)

        # 开始进行信号的迭代分解
        while imf < self.max_imfs - 1:

            # Initialising intermediary IMFs
            H = Residue.copy()

            # flag to control sifting loop
            sift_stop = 0

            # Combining two signals with equal weights
            Combined = np.sum(H / np.sqrt(num_vars), axis=1)

            # Obtaining extrema of combined signal
            Maxima, MaxPos, Minima, MinPos = extrema(Combined)

            # Checking whether there are too few extrema in the IMF
            if np.count_nonzero(Maxima) < 3 or np.count_nonzero(Minima) < 3:
                # Fewer than three extrema found in extrema map. Stopping now...
                break

            # Window size determination by delaunay triangulation
            windows[:, imf] = self.filter_size1D(imax=MaxPos, imin=MinPos)

            # extracting window size chosen by input parameter
            w_sz = windows[self.window_type, imf]

            # Begin sifting iteration
            while not sift_stop:
                # Incrementing sift counter
                sift_count[imf] = sift_count[imf] + 1

                # Entering parallel sift calculations
                for i in range(num_vars):
                    H1[:, i] = self.sift(H[:, i], w_sz=w_sz)

                    # 计算均方误差
                    mse[i] = immse(a=H1[:, i], b=H[:, i])

                # Stop condition checks
                if (mse < tol).all() and sift_count[imf] != 1:
                    sift_stop = True

                H = H1

            # Storing IMFs
            imfs[:, :, imf] = H

            # Subtracting from Residual Signals
            Residue = Residue - imfs[:, :, imf]

            # Incrementing IMF counter
            imf = imf + 1

        # Checking for oversifting
        if np.any(sift_count >= 5 * np.ones(shape=self.max_imfs)):
            print("Decomposition may be oversifted. Checking if window size increases monotonically...")

            if np.any(np.diff(windows[self.window_type, :]) <= np.zeros(shape=self.max_imfs - 1)):
                print("Filter window size does not increase monotonically")

        # 记录残差分量
        imfs[:, :, -1] = Residue

        # 调整输出的通道顺序
        imfs = np.transpose(imfs, axes=[2, 0, 1])

        # Organising results
        if return_all is True:
            return imfs, Residue, windows, sift_count

        else:
            return imfs


def immse(a: np.ndarray, b: np.ndarray) -> floating[Any]:
    """
    复现Matlab中的immse函数
    该函数用于计算两个数组的均方误差
    :param a: 输入的数组a
    :param b: 输入的数组b
    :return: 计算得到的均方误差值
    """
    assert a.shape == b.shape, "the shape of `a` and `b` must be same"
    return np.mean((a - b) ** 2)


def env_smoothing(env_max: np.ndarray, env_min: np.ndarray, w_sz: Union[int, float]) -> np.ndarray:
    """
    平滑上下包络谱信号
    :param env_max: 上包络谱信号
    :param env_min: 下包络谱信号
    :param w_sz: 滑动平均的窗口大小
    :return: 平滑后且上下包络谱的均值
    """
    # Make sure the window_size is an integer
    w_sz = int(w_sz)

    # Smoothing
    env_maxs = simple_moving_average(signal=env_max, window_size=w_sz)
    env_mins = simple_moving_average(signal=env_min, window_size=w_sz)

    # Calculating mean envelope
    return (env_maxs + env_mins) / 2


if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    from pysdkit.data import test_emd

    faemd = FAEMD(max_imfs=2, window_type=0)

    time = np.linspace(0, 6 * np.pi, 100)
    u = 2.5 * np.cos(time)
    v = 2.5 * np.sin(5 * time)
    w = u + v
    w = w[np.newaxis, :]
    w = np.concatenate([w, w], axis=0)

    imfs = faemd.fit_transform(signal=w, return_all=False)

    fig, ax = plt.subplots(4, 2)
    for i in range(2):
        ax[0, i].plot(w[i, :])
        for j in range(2):
            ax[j + 1, i].plot(imfs[j, :, i])

    plt.show()

    time, signal = test_emd()
    signal = signal[np.newaxis, :]
    signal = np.concatenate([signal, signal], axis=0)
    imfs = faemd.fit_transform(signal=signal, return_all=False)

    fig, ax = plt.subplots(4, 2)
    for i in range(2):
        ax[0, i].plot(signal[i, :])
        for j in range(2):
            ax[j + 1, i].plot(imfs[j, :, i])

    plt.show()
