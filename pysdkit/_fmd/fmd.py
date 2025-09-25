# -*- coding: utf-8 -*-
"""
Created on 2025/01/12 00:05:39
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import firwin, hilbert

class FMD:
    """
    Feature Mode Decomposition (FMD) class for decomposing a signal into a predefined
    number of modes using iterative filter bank and kurtosis-based optimization.

    Parameters
    ----------
    fs : float
        Sampling frequency of the input signal.
    filter_size : int, optional (default=30)
        Length of the FIR filters used in the filter bank.
    cut_num : int, optional (default=7)
        Number of equally spaced sub-bands to split the frequency range [0, fs/2].
    mode_num : int, optional (default=2)
        Number of modes to extract from the input signal.
    max_iter_num : int, optional (default=20)
        Maximum number of iterations for filter optimization.
    """

    def __init__(self, fs, filter_size=30, cut_num=7, mode_num=2, max_iter_num=20):
        self.fs = fs
        self.filter_size = filter_size
        self.cut_num = cut_num
        self.mode_num = mode_num
        self.max_iter_num = max_iter_num

    def fit_transform(self, x):
        """
        Decompose the input signal into modes using FMD.

        Parameters
        ----------
        x : array-like
            Input signal to be decomposed.

        Returns
        -------
        final_modes : ndarray, shape (mode_num, len(x))
            Extracted modes after decomposition.
        """
        x = np.asarray(x).flatten()
        freq_bound = np.arange(0, 1, 1 / self.cut_num)
        temp_filters = np.zeros((self.filter_size, self.cut_num))

        # Initialize a filter bank with Hann-windowed FIR bandpass filters
        for n, fb in enumerate(freq_bound):
            b = firwin(self.filter_size,
                       [fb + np.finfo(float).eps, fb + 1 / self.cut_num - np.finfo(float).eps],
                       window='hann', pass_zero=False)
            temp_filters[:, n] = b

        temp_sig = np.tile(x[:, None], (1, self.cut_num))
        itercount = 2

        while True:
            iternum = 2
            if itercount == 2:
                # Reduce iterations for non-target bands to speed convergence
                iternum = self.max_iter_num - (self.cut_num - self.mode_num) * iternum

            X = []
            for n in range(temp_filters.shape[1]):
                # Apply Multi-Cycle Kurtosis Deconvolution (MCKD) to refine each filter output
                y_iter, f_iter, k_iter, T_iter = self._xxc_mckd(
                    temp_sig[:, n], temp_filters[:, n], iternum, M=1
                )
                freq_resp = np.abs(fft(f_iter))[:self.filter_size // 2]
                peak_freq = (np.argmax(freq_resp)) * (self.fs / self.filter_size)
                X.append([y_iter[:, -1], f_iter[:, -1], k_iter[-1], freq_resp, peak_freq, T_iter])

            # Update signal matrix and filter bank using optimized filters
            temp_sig = np.column_stack([xi[0] for xi in X])
            temp_filters = np.column_stack([xi[1] for xi in X])

            # Compute pairwise correlation and remove the most redundant component
            corr_matrix = np.abs(np.corrcoef(temp_sig, rowvar=False))
            corr_matrix = np.triu(corr_matrix, 1)
            I, J, _ = self._max_IJ(corr_matrix)

            XI = temp_sig[:, I] - np.mean(temp_sig[:, I])
            XJ = temp_sig[:, J] - np.mean(temp_sig[:, J])

            # Compare cyclic kurtosis (CK) values to decide which component to drop
            KI = self._CK(XI, X[I][5], 1)
            KJ = self._CK(XJ, X[J][5], 1)

            output = J if KI > KJ else I
            temp_sig = np.delete(temp_sig, output, axis=1)
            temp_filters = np.delete(temp_filters, output, axis=1)

            if temp_filters.shape[1] == self.mode_num - 1:
                break

            itercount += 1

        final_modes = np.column_stack([xi[0] for xi in X])
        return final_modes.T

    def _xxc_mckd(self, x, f_init, term_iter, T=None, M=3):
        """
        Multi-Cycle Kurtosis Deconvolution (MCKD) optimization for filter refinement.

        Parameters
        ----------
        x : ndarray
            Input signal segment.
        f_init : ndarray
            Initial filter coefficients.
        term_iter : int
            Number of iterations to optimize filter.
        T : int, optional
            Cycle period (if None, estimated from signal envelope autocorrelation).
        M : int, optional (default=3)
            Order of cyclic kurtosis.

        Returns
        -------
        y_final : ndarray
            Filtered signal at each iteration.
        f_final : ndarray
            Optimized filter coefficients at each iteration.
        ck_iter : ndarray
            CK values over iterations.
        T : int
            Estimated cycle period.
        """
        if T is None:
            env = np.abs(hilbert(x)) - np.mean(np.abs(hilbert(x)))
            T = self._TT(env)

        T = int(round(T))
        x = np.asarray(x).flatten()
        L = len(f_init)
        N = len(x)

        # Build delayed signal matrix XmT for MCKD optimization
        XmT = np.zeros((L, N, M + 1))
        for m in range(M + 1):
            for l in range(L):
                if l == 0:
                    XmT[l, m * T:, m] = x[:N - m * T]
                else:
                    XmT[l, 1:, m] = XmT[l - 1, :-1, m]

        Xinv = np.linalg.inv(XmT[:, :, 0] @ XmT[:, :, 0].T)

        f = f_init.copy()
        y_final = []
        f_final = []
        ck_iter = []

        for _ in range(term_iter):
            y = XmT[:, :, 0].T @ f  # filter output

            # Construct shifted versions for cyclic statistics
            yt = np.zeros((N, M + 1))
            for m in range(M + 1):
                yt[:, m] = y if m == 0 else np.concatenate([np.zeros(T), yt[:-T, m-1]])

            # Compute alpha (partial derivatives) and beta for filter update
            alpha = np.zeros_like(yt)
            for m in range(M + 1):
                idx = [i for i in range(M + 1) if i != m]
                alpha[:, m] = (np.prod(yt[:, idx], axis=1) ** 2) * yt[:, m]

            beta = np.prod(yt, axis=1)

            Xalpha = np.zeros(L)
            for m in range(M + 1):
                Xalpha += XmT[:, :, m] @ alpha[:, m]

            # Update filter using gradient ascent on CK criterion
            f = (np.sum(y ** 2) / (2 * np.sum(beta ** 2))) * (Xinv @ Xalpha)
            f /= np.sqrt(np.sum(f ** 2))

            # Compute CK value for convergence monitoring
            ck_val = np.sum(np.prod(yt, axis=1) ** 2) / (np.sum(y ** 2) ** (M + 1))
            ck_iter.append(ck_val)

            # Update delay matrix with new cycle period estimate
            env_y = np.abs(hilbert(y)) - np.mean(np.abs(hilbert(y)))
            T = int(round(self._TT(env_y)))

            XmT = np.zeros((L, N, M + 1))
            for m in range(M + 1):
                for l in range(L):
                    if l == 0:
                        XmT[l, m * T:, m] = x[:N - m * T]
                    else:
                        XmT[l, 1:, m] = XmT[l - 1, :-1, m]

            Xinv = np.linalg.inv(XmT[:, :, 0] @ XmT[:, :, 0].T)

            y_final.append(signal.lfilter(f, [1], x))
            f_final.append(f.copy())

        y_final = np.column_stack(y_final) if len(y_final) > 1 else y_final[0][:, None]
        f_final = np.column_stack(f_final) if len(f_final) > 1 else f_final[0][:, None]

        return y_final, f_final, np.array(ck_iter), T

    def _TT(self, y):
        """Estimate signal period T using autocorrelation zero-crossing and peak."""
        NA = signal.correlate(y, y, mode='full') / np.dot(y, y)
        NA = NA[len(NA) // 2:]
        zeroposi = 1
        for lag in range(1, len(NA)):
            if (NA[lag - 1] > 0 and NA[lag] < 0) or NA[lag] == 0:
                zeroposi = lag
                break

        if zeroposi >= len(NA):  # fallback if no zero-crossing is found
            zeroposi = 1

        NA = NA[zeroposi:]
        max_position = np.argmax(NA)
        return zeroposi + max_position

    def _CK(self, x, T, M=2):
        """Compute cyclic kurtosis (CK) of the signal for a given period T and order M."""
        x = np.asarray(x).flatten()
        N = len(x)
        x_shift = np.zeros((M + 1, N))
        x_shift[0] = x
        for m in range(1, M + 1):
            x_shift[m, T:] = x_shift[m - 1, :-T]
        return np.sum(np.prod(x_shift, axis=0) ** 2) / (np.sum(x ** 2) ** (M + 1))

    def _max_IJ(self, X):
        """Find the indices (I, J) of the maximum value in the upper triangle of correlation matrix X."""
        tempI = np.argmax(X, axis=0)
        temp = np.max(X, axis=0)
        J = np.argmax(temp)
        I = tempI[J]
        return I, J, X[I, J]
