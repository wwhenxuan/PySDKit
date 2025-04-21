# -*- coding: utf-8 -*-
"""
Created on 2025/04/16 22:51:55
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Tuple

from pysdkit.utils import fft, ifft, fmirror


class EFD(object):
    """
    Empirical Fourier Decomposition

    The proposed EFD combines the uses of an improved Fourier spectrum segmentation technique and an ideal filter bank.
    The segmentation technique can solve the inconsistency problem by predefining the number of modes in a signal to be
    decomposed and filter functions in the ideal filter bank have no transition phases, which can solve the mode mixing problem.
    Numerical investigations are conducted to study the accuracy of the EFD. It is shown that the EFD can yield accurate
    and consistent decomposition results for signals with multiple non-stationary modes and those with closely-spaced modes,
    compared with decomposition results by the EWT, FDM, variational mode decomposition and empirical mode decomposition.

    Wei Zhou, Zhongren Feng, Y.F. Xu, Xiongjiang Wang, Hao Lv,
    Empirical Fourier decomposition: An accurate signal decomposition method for nonlinear and non-stationary time series analysis,
    Mechanical Systems and Signal Processing,
    Volume 163, 2022, 108155, ISSN 0888-3270, https://doi.org/10.1016/j.ymssp.2021.108155.
    """

    def __init__(self, max_imfs: Optional[int] = 3) -> None:
        self.max_imfs = max_imfs

    def __call__(self, *args, **kwargs):
        """allow instances to be called like functions"""
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Empirical Fourier Decomposition (EFD)"

    def fit_transform(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Signal decomposition using EFD algorithm

        :param signal: the time domain signal (1D numpy array) to be decomposed
        :return: the decomposed results of IMFs
        """

        # We compute the Fourier transform of input signal
        ff = fft(signal)

        # We extract the boundaries of Fourier segments
        bounds, cerf = segm_tec(f=ff, N=self.max_imfs)

        # We trun the boundaries to [0,pi]
        bounds = bounds * np.pi / np.round(len(ff / 2))

        # Filtering
        # We extend the signal by miroring to deal with the boundaries
        signal = fmirror(ts=signal, sym=len(signal) // 2)
        ff = fft(signal)


def segm_tec(f: np.ndarray, N: Optional[int] = 3) -> Tuple[np.ndarray, float]:
    """
    This function is used to implement the improved segmentation technique.
    function [boundaries,cerf] = Boundaries_Dect(ff,N)

    :param f: the Fourier spectrum of input signal
    :param N: maximum number of segments
    :return: -cerf: vector containing the central frequency of each band,[0,pi]
             -bound: vector containing the set of boundaries corresponding
             to the Fourier line segmentation (normalized between 0 and Pi)
    """
    bounds, cerf = None, None

    # detect the local maxima and minina
    locmax = np.zeros(f.shape)
    locmin = np.max(f) * np.ones(f.shape)

    for i in range(1, len(f) - 1):
        if f[i - 1] < f[i] and f[i] > f[i + 1]:
            locmax[i] = f[i]
        if f[i - 1] > f[i] and f[i] < f[i + 1]:
            locmin[i] = f[i]

    locmax[0] = f[0]
    locmax[-1] = f[-1]

    # keep the N-th highest maxima and their index
    if N != 0:
        lmax = np.sort(locmax)[::-1]  # 按列降序排序
        Imax = np.argsort(locmax)[::-1]  # 获取排序后的索引

        if len(lmax) > N:
            Imax = np.sort(Imax[0:N])
        else:
            Imax = np.sort(Imax)
            N = len(lmax)

        # detect the lowest minima between two consecutive maxima
        M = N + 1  # numbers of the boundaries
        print(Imax)
        omega = np.concatenate([np.array([0]), Imax, np.array([len(f)])])  # location

        print(omega)

        bounds = np.zeros(M)

        print(f.shape)

        for i in range(0, M):
            if (i == 0 or i == M - 1) and omega[i] == omega[i + 1]:
                bounds[i] = omega[i] - 1
            else:
                print(omega[i], omega[i + 1])
                lmin = np.min(f[omega[i] : omega[i + 1]])
                ind = np.argmin(f[omega[i] : omega[i + 1]])
                print("ind", ind)
                bounds[i] = omega[i] + ind - 2
            cerf = Imax * np.pi / np.round(len(f))

    return bounds, cerf


if __name__ == "__main__":
    T = 1
    fs = 1000
    t = np.arange(0, T, 1 / fs)
    f11 = 6 * t
    f12 = 2 * np.cos(8 * np.pi * t)
    f13 = np.cos(40 * np.pi * t)

    s = f11 + f12 + f13

    efd = EFD(max_imfs=3)
    efd.fit_transform(s)
