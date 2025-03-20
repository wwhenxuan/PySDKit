# -*- coding: utf-8 -*-
"""
Created on 2025/01/12 00:05:39
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
not done!!!
"""
import math
import numpy as np

from scipy.signal import correlate, hilbert

from typing import Tuple


def max_IJ(X: np.ndarray) -> Tuple[int, int, float]:
    """找到每列的最大值及其行索引"""
    temp = np.max(X, axis=0)
    tempI = np.argmax(X, axis=0)

    # 找到 temp 中的最大值及其列索引
    M = np.max(temp)
    J = np.argmax(temp)

    # 找到 X 中最大值的行索引
    I = tempI[J]

    return I, J, M


def Correlated_Kurtosis(x: np.ndarray, T: int, M: int = 2) -> float:
    """计算信号的自相关峭度（Correlated Kurtosis）"""
    # 确保 x 是行向量
    x = np.atleast_1d(x).flatten()

    # 获取一维信号的长度
    N = len(x)
    # 初始化时间延迟矩阵
    x_shift = np.zeros((M + 1, N))
    x_shift[0, :] = x
    # 生成时间延迟信号,每次延迟T个样本
    for m in range(1, M + 1):
        x_shift[m, T:] = x_shift[m - 1, :-T]
    # 计算自相关峭度
    ck = np.sum(np.prod(x_shift, axis=0) ** 2) / np.sum(x**2) ** (M + 1)
    return ck


def xcorr(x: np.ndarray, lags: int) -> np.ndarray:
    """Reproduce the functionality of Matlab's xcorr function"""

    # Get the length of the signal
    N = len(x)

    # Zero-pad the signal if the maximum lag is greater than or equal to the signal length
    if lags >= N:
        x_pad = np.zeros(lags + 1)
        x_pad[:N] = x
        x = x_pad

    # Compute the autocorrelation
    auto_corr = correlate(x, x, mode="full")

    # Sample the autocorrelation if the lag is less than the signal length
    if lags < N:
        auto_corr = auto_corr[N - lags - 1 : N + lags]
    auto_corr = auto_corr / np.max(auto_corr)
    return auto_corr


def TT(y: np.ndarray, fs: int):
    """
    Estimate the period in y based on auto-correlation function.
    
    :param y: the input numpy ndarray signal.
    :param fs: the sampling frequency of the input signal.
    :return: the estimated period.
    """
    length = len(y)

    # Find the maximum lag M
    M = fs

    NA = xcorr(y, lags=M)
    NA = NA[math.ceil(len(NA) / 2) - 1 :]

    # Find the first zero-crossing
    sample1 = NA[0]
    # zeroposi = 0

    for lag in range(1, len(NA)):
        sample2 = NA[lag]

        if sample1 > 0 > sample2:
            zeroposi = lag
            break
        elif sample1 == 0 or sample2 == 0:
            zeroposi = lag
            break
        else:
            sample1 = sample2
    # Cut from the first zero-crossing
    NA = NA[zeroposi:]
    # Find the max position (time lag)
    # Corresponding the max aside from the first zero-crossing
    max_position = np.argmax(NA)
    # Give the estimated period by autocorrelation
    T = zeroposi + max_position + 2
    return T


def xxc_mckd(fs, x, f_init, termIter, T, M, plotMode):
    """
    Y. Miao, M. Zhao, J. Lin, Y. Lei
    "Application of an improved maximum correlated kurtosis deconvolution method for fault diagnosis of rolling element bearings"
    Mechanical Systems and Signal Processing, 92 (2017) 173 - 195.
    """
    # Assign default values for inputs
    if termIter.size == 0:
        termIter = 30
    if plotMode.size == 0:
        plotMode = 0
    if M.size == 0:
        M = 3

    if T.size == 0:
        hil = np.abs(hilbert(x))
        xxenvelope = hil - np.mean(hil)
        T = TT(xxenvelope, fs)

    T = np.round(T)

    x = x.resahpe(-1, 1)
    L = len(f_init)
    N = len(x)

    # Calculate XmT
    XmT = np.zeros([L, N, M + 1])
    for m in range(0, M + 1):
        for l in range(0, L):
            if l == 0:
                XmT[l, (m * T) :, m] = x[: N - m * T]
            else:
                XmT[l, 1:, m] = XmT[l - 1, :-1, m]


if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    print(TT(a, fs=13))
