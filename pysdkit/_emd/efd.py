# -*- coding: utf-8 -*-
"""
Created on 2025/04/16 22:51:55
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Tuple, Union

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

    def __call__(
        self, signal: np.ndarray, return_all: Optional[bool] = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal, return_all=return_all)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Empirical Fourier Decomposition (EFD)"

    def fit_transform(
        self, signal: np.ndarray, return_all: Optional[bool] = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Signal decomposition using EFD algorithm

        :param signal: the time domain signal (1D numpy array) to be decomposed
        :param return_all: whether to return all results (including vector containing the central frequency) or just the IMFs
        :return: the decomposed results of IMFs
        """
        # Make sure the input signal length is an even number, otherwise the mirror expansion cannot be performed.
        if len(signal) % 2 == 1:
            signal = signal[:-1]

        # We compute the Fourier transform of input signal
        ff = fft(signal)

        # We extract the boundaries of Fourier segments
        bounds, cerf = segm_tec(
            f=np.abs(ff[0 : int(np.round(len(ff) / 2))]), N=self.max_imfs
        )

        # We trun the boundaries to [0,pi]
        bounds = bounds * np.pi / np.round(len(ff) / 2)

        # Filtering
        # We extend the signal by miroring to deal with the boundaries
        x = fmirror(ts=signal, sym=len(signal) // 2)

        # Half the record length is used to recover the original signal
        T = len(x) // 2

        # Perform fast Fourier transform on the image-extended signal
        ff = fft(x)

        # We obtain the boundaries in the extend f
        bound2 = np.ceil(bounds * np.round(len(ff) / 2) / np.pi).astype(int)

        # We get the core of filtering
        number = len(bound2) - 1

        # Record the basic results after decomposition
        efd = np.zeros(shape=(number, len(ff)))

        # Record the final results
        imfs = np.zeros(shape=(number, len(signal)))

        # The results are stored in the frequency domain
        ft = np.zeros([number, len(ff)], dtype=complex)

        # We define an ideal functions and extract components
        for k in range(0, number):
            if bound2[k] == 0:
                ft[k, 0 : bound2[k + 1]] = ff[0 : bound2[k + 1]]
                ft[k, len(ff) + 2 - bound2[k + 1] : len(ff)] = ff[
                    len(ff) + 2 - bound2[k + 1] : len(ff)
                ]
            else:
                ft[k, bound2[k] : bound2[k + 1]] = ff[bound2[k] : bound2[k + 1]]
                ft[k, len(ff) + 2 - bound2[k + 1] : len(ff) + 2 - bound2[k]] = ff[
                    len(ff) + 2 - bound2[k + 1] : len(ff) + 2 - bound2[k]
                ]

            # Recover the original signal from the frequency domain
            efd[k, :] = np.real(ifft(ft[k, :]))
            imfs[k, :] = efd[k][T // 2 : 3 * T // 2]

        if return_all is True:
            return imfs, cerf
        return imfs


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

    # Traverse the entire sequence to find local maxima and local minima
    for i in range(1, len(f) - 1):
        if f[i - 1] < f[i] and f[i] > f[i + 1]:
            # Local maximum
            locmax[i] = f[i]
        if f[i - 1] > f[i] and f[i] < f[i + 1]:
            # Local Minima
            locmin[i] = f[i]

    # Processing the final edge
    locmax[0] = f[0]
    locmax[-1] = f[-1]

    # keep the N-th highest maxima and their index
    if N != 0:
        lmax = np.sort(locmax)[::-1]  # Sort by column in descending order
        Imax = np.argsort(locmax)[::-1]  # Get the sorted index

        if len(lmax) > N:
            Imax = np.sort(Imax[0:N])
        else:
            Imax = np.sort(Imax)
            N = len(lmax)

        # detect the lowest minima between two consecutive maxima
        M = N + 1  # numbers of the boundaries
        omega = np.concatenate([np.array([0]), Imax, np.array([len(f)])])  # location
        bounds = np.zeros(M)

        for i in range(0, M):
            if (i == 0 or i == M - 1) and omega[i] == omega[i + 1]:
                bounds[i] = omega[i]
            else:
                # print(omega[i], omega[i + 1])
                lmin = np.min(f[omega[i] : omega[i + 1]])
                ind = np.argmin(f[omega[i] : omega[i + 1]])
                bounds[i] = omega[i] + ind
            cerf = Imax * np.pi / np.round(len(f))

    return bounds, cerf


if __name__ == "__main__":
    from pysdkit.plot import plot_IMFs
    from matplotlib import pyplot as plt

    T = 1
    fs = 1000
    t = np.arange(0, T + 1 / fs, 1 / fs)
    f11 = 6 * t
    f12 = 2 * np.cos(8 * np.pi * t)
    f13 = np.cos(40 * np.pi * t)

    s = f11 + f12 + f13

    print("len(s)", s.shape)

    efd = EFD(max_imfs=3)
    imfs = efd.fit_transform(s)
    plot_IMFs(s, imfs)
    plt.show()

    from pysdkit.data import test_emd

    print("\n")

    t, s = test_emd()
    print("len(s)", s.shape)

    imfs = efd.fit_transform(s)
    plot_IMFs(s, imfs)
    plt.show()
