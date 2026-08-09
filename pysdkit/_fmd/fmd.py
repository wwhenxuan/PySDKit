# -*- coding: utf-8 -*-
"""
Feature Mode Decomposition (FMD).

Yonghao Miao et al., "Feature Mode Decomposition: New Decomposition Theory
for Rotating Machinery Fault Diagnosis," IEEE Trans. Ind. Electron., 2022.
DOI: 10.1109/TIE.2022.3156156

MATLAB reference:
https://www.mathworks.com/matlabcentral/fileexchange/108099-feature-mode-decomposition-fmd
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from scipy.fft import fft
from scipy.signal import correlate, firwin, hilbert, lfilter


class FMD(object):
    """
    Feature Mode Decomposition.

    Decomposes a vibration record into a fixed number of modes by
    (i) initializing a Hanning-windowed FIR filter bank over [0, fs/2],
    (ii) refining each filter with improved MCKD (correlated kurtosis),
    and (iii) iteratively dropping the more redundant of the most
    correlated mode pair until ``mode_num`` modes remain.
    """

    def __init__(
        self,
        fs: Union[int, float] = 1.0,
        mode_num: int = 2,
        filter_size: int = 30,
        cut_num: int = 7,
        max_iter_num: int = 20,
    ) -> None:
        """
        :param fs: sampling frequency of the input signal
        :param mode_num: number of modes to keep
        :param filter_size: FIR filter length ``L``
        :param cut_num: number of initial uniform sub-bands ``K``
        :param max_iter_num: max MCKD iterations on the first sweep
        """
        if not isinstance(mode_num, (int, np.integer)) or mode_num < 1:
            raise ValueError("mode_num must be a positive integer")
        if not isinstance(filter_size, (int, np.integer)) or filter_size < 3:
            raise ValueError("filter_size must be an integer >= 3")
        if not isinstance(cut_num, (int, np.integer)) or cut_num < 1:
            raise ValueError("cut_num must be a positive integer")
        if mode_num > cut_num:
            raise ValueError("mode_num cannot exceed cut_num")
        if not isinstance(max_iter_num, (int, np.integer)) or max_iter_num < 1:
            raise ValueError("max_iter_num must be a positive integer")
        if fs is None or float(fs) <= 0:
            raise ValueError("fs must be a positive number")

        self.fs = float(fs)
        self.filter_size = int(filter_size)
        self.cut_num = int(cut_num)
        self.mode_num = int(mode_num)
        self.max_iter_num = int(max_iter_num)

        self.imfs: Optional[np.ndarray] = None
        self.filters: Optional[np.ndarray] = None
        self.peak_freqs: Optional[np.ndarray] = None

    def __call__(self, x: np.ndarray, fs: Optional[Union[int, float]] = None) -> np.ndarray:
        return self.fit_transform(x, fs)

    def __str__(self) -> str:
        return "Feature Mode Decomposition (FMD)"

    def fit_transform(
        self, x: np.ndarray, fs: Optional[Union[int, float]] = None
    ) -> np.ndarray:
        """
        Decompose ``x`` into ``mode_num`` feature modes.

        :param x: 1-D real signal
        :param fs: optional sampling frequency override
        :return: modes of shape ``(mode_num, N)``
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.size < self.filter_size:
            raise ValueError("signal length must be >= filter_size")

        if fs is not None:
            if float(fs) <= 0:
                raise ValueError("fs must be a positive number")
            self.fs = float(fs)

        # Eq. (3): uniform Hanning FIR bank over the Nyquist band
        freq_bound = np.arange(0.0, 1.0, 1.0 / self.cut_num)
        temp_filters = np.zeros((self.filter_size, self.cut_num), dtype=float)
        eps = np.finfo(float).eps
        for n, fb in enumerate(freq_bound):
            lo = fb + eps
            hi = fb + 1.0 / self.cut_num - eps
            hi = min(hi, 1.0 - eps)
            if lo >= hi:
                lo = max(eps, hi - eps)
            temp_filters[:, n] = firwin(
                self.filter_size,
                [lo, hi],
                window="hann",
                pass_zero=False,
            )

        # MATLAB: temp_sig = repmat(x, [1, CutNum])
        temp_sig = np.tile(x[:, None], (1, self.cut_num))

        itercount = 2
        X = []  # last successful sweep (before the final drop)
        while True:
            # First sweep uses a larger iteration budget; later sweeps use 2
            iternum = 2
            if itercount == 2:
                iternum = self.max_iter_num - (self.cut_num - self.mode_num) * iternum
            iternum = max(int(iternum), 1)

            X = []
            for n in range(temp_filters.shape[1]):
                y_iter, f_iter, k_iter, T_iter = self._xxc_mckd(
                    temp_sig[:, n],
                    temp_filters[:, n],
                    iternum,
                    T=None,
                    M=1,
                )
                f_last = f_iter[:, -1]
                y_last = y_iter[:, -1]
                freq_resp = np.abs(fft(f_last))[: self.filter_size // 2]
                peak_freq = float(np.argmax(freq_resp) * (self.fs / self.filter_size))
                X.append(
                    [
                        y_last,
                        f_last,
                        float(k_iter[-1]),
                        freq_resp,
                        peak_freq,
                        int(T_iter),
                    ]
                )

            temp_sig = np.column_stack([xi[0] for xi in X])
            temp_filters = np.column_stack([xi[1] for xi in X])

            # Drop the weaker CK mode among the most correlated pair
            if temp_sig.shape[1] == 1:
                # Only one candidate left: drop it from the working set so the
                # ModeNum-1 stop condition can fire (Final_Mode still uses X).
                output = 0
            else:
                corr_matrix = np.abs(np.corrcoef(temp_sig, rowvar=False))
                corr_matrix = np.triu(corr_matrix, 1)
                I, J, _ = max_IJ(corr_matrix)

                XI = temp_sig[:, I] - np.mean(temp_sig[:, I])
                XJ = temp_sig[:, J] - np.mean(temp_sig[:, J])
                KI = CK(XI, X[I][5], 1)
                KJ = CK(XJ, X[J][5], 1)
                output = J if KI > KJ else I

            temp_sig = np.delete(temp_sig, output, axis=1)
            temp_filters = np.delete(temp_filters, output, axis=1)

            # MATLAB stops when remaining filters == ModeNum - 1;
            # Final_Mode is taken from the pre-deletion cell (still ModeNum).
            if temp_filters.shape[1] == self.mode_num - 1:
                break

            itercount += 1

        final_modes = np.column_stack([xi[0] for xi in X]).T
        self.imfs = final_modes
        self.filters = np.column_stack([xi[1] for xi in X]).T
        self.peak_freqs = np.asarray([xi[4] for xi in X], dtype=float)
        return final_modes

    def _xxc_mckd(
        self,
        x: np.ndarray,
        f_init: np.ndarray,
        term_iter: int,
        T: Optional[int] = None,
        M: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Improved multi-point correlated kurtosis deconvolution (IMCKD).

        Port of MATLAB ``xxc_mckd`` in FMD.m.  Filter coefficients used for
        ``y_final`` / ``f_final`` are stored *before* each coefficient update,
        matching the reference implementation.

        :param x: input segment
        :param f_init: initial FIR coefficients (length ``L``)
        :param term_iter: number of iterations
        :param T: cycle period in samples (estimated if None)
        :param M: CK shift order
        :return: ``(y_final, f_final, ck_iter, T)``
        """
        x = np.asarray(x, dtype=float).ravel()
        f_init = np.asarray(f_init, dtype=float).ravel()
        term_iter = max(int(term_iter), 1)
        M = int(M)

        if T is None:
            env = np.abs(hilbert(x))
            env = env - np.mean(env)
            T = TT(env, self.fs)

        T = max(int(round(T)), 1)
        L = f_init.size
        N = x.size

        # Guard against an unrealistically large period estimate
        max_T = max(N // (M + 1) - 1, 1)
        T = min(T, max_T)

        XmT, Xinv = _build_XmT(x, L, T, M)

        f = f_init.copy()
        y_final = np.zeros((N, term_iter), dtype=float)
        f_final = np.zeros((L, term_iter), dtype=float)
        ck_iter = np.zeros(term_iter, dtype=float)
        T_final = T

        for n in range(term_iter):
            y = XmT[:, :, 0].T @ f
            f_final[:, n] = f

            yt = np.zeros((N, M + 1), dtype=float)
            for m in range(M + 1):
                if m == 0:
                    yt[:, m] = y
                else:
                    yt[T:, m] = yt[:-T, m - 1]

            alpha = np.zeros_like(yt)
            for m in range(M + 1):
                idx = [i for i in range(M + 1) if i != m]
                alpha[:, m] = (np.prod(yt[:, idx], axis=1) ** 2) * yt[:, m]

            beta = np.prod(yt, axis=1)
            Xalpha = np.zeros(L, dtype=float)
            for m in range(M + 1):
                Xalpha += XmT[:, :, m] @ alpha[:, m]

            denom = 2.0 * np.sum(beta**2) + np.finfo(float).eps
            f = (np.sum(y**2) / denom) * (Xinv @ Xalpha)
            f = f / (np.sqrt(np.sum(f**2)) + np.finfo(float).eps)

            ck_iter[n] = np.sum(np.prod(yt, axis=1) ** 2) / (
                (np.sum(y**2) + np.finfo(float).eps) ** (M + 1)
            )

            # Re-estimate period from the current filtered output
            env_y = np.abs(hilbert(y))
            env_y = env_y - np.mean(env_y)
            T = max(int(round(TT(env_y, self.fs))), 1)
            T = min(T, max_T)
            T_final = T

            XmT, Xinv = _build_XmT(x, L, T, M)
            y_final[:, n] = lfilter(f_final[:, n], [1.0], x)

        return y_final, f_final, ck_iter, int(T_final)


def _build_XmT(
    x: np.ndarray, L: int, T: int, M: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the delayed Toeplitz tensors and ``(X0 X0^T)^{-1}`` used by MCKD."""
    N = x.size
    XmT = np.zeros((L, N, M + 1), dtype=float)
    for m in range(M + 1):
        for l in range(L):
            if l == 0:
                if m * T < N:
                    XmT[l, m * T :, m] = x[: N - m * T]
            else:
                XmT[l, 1:, m] = XmT[l - 1, :-1, m]

    G = XmT[:, :, 0] @ XmT[:, :, 0].T
    # Mild ridge for numerical stability (MATLAB uses bare inv)
    G.flat[:: L + 1] += 1e-12
    Xinv = np.linalg.inv(G)
    return XmT, Xinv


def TT(y: np.ndarray, fs: Union[int, float]) -> int:
    """
    Estimate the dominant period of ``y`` via autocorrelation.

    Matches MATLAB ``TT(y, fs)`` in FMD.m: ``xcorr`` with ``maxlag = fs``,
    cut after the first zero-crossing, then take the lag of the peak.
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 2:
        return 1

    maxlag = int(round(float(fs)))
    if maxlag < 1:
        maxlag = y.size - 1

    corr = correlate(y, y, mode="full")
    mid = corr.size // 2
    NA = corr[mid : mid + maxlag + 1]
    denom = float(np.dot(y, y)) + np.finfo(float).eps
    NA = NA / denom

    zeroposi = None
    sample1 = NA[0]
    for lag in range(1, NA.size):
        sample2 = NA[lag]
        if (sample1 > 0 and sample2 < 0) or sample1 == 0 or sample2 == 0:
            zeroposi = lag
            break
        sample1 = sample2

    if zeroposi is None:
        zeroposi = 1

    # MATLAB uses 1-based indices: T = zeroposi + max_position
    # with both factors 1-based ⇒ 0-based: zeroposi + argmax + 2
    NA_cut = NA[zeroposi:]
    max_position = int(np.argmax(NA_cut))
    return int(zeroposi + max_position + 2)


def CK(x: np.ndarray, T: int, M: int = 2) -> float:
    """Correlated kurtosis of ``x`` for period ``T`` and order ``M``."""
    x = np.asarray(x, dtype=float).ravel()
    T = max(int(T), 1)
    M = int(M)
    N = x.size
    x_shift = np.zeros((M + 1, N), dtype=float)
    x_shift[0] = x
    for m in range(1, M + 1):
        x_shift[m, T:] = x_shift[m - 1, :-T]
    return float(
        np.sum(np.prod(x_shift, axis=0) ** 2)
        / ((np.sum(x**2) + np.finfo(float).eps) ** (M + 1))
    )


def max_IJ(X: np.ndarray) -> Tuple[int, int, float]:
    """Row/column indices of the maximum entry of matrix ``X`` (MATLAB ``max_IJ``)."""
    X = np.asarray(X, dtype=float)
    temp_I = np.argmax(X, axis=0)
    temp = np.max(X, axis=0)
    J = int(np.argmax(temp))
    I = int(temp_I[J])
    return I, J, float(X[I, J])


# Backwards-compatible private aliases
_TT = TT
_CK = CK
_max_IJ = max_IJ


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from pysdkit.data import test_emd
    from pysdkit.plot import plot_IMFs

    time, sig = test_emd()
    fmd = FMD(fs=1000, mode_num=2, cut_num=5, max_iter_num=10)
    modes = fmd.fit_transform(sig)
    plot_IMFs(sig, modes)
    plt.show()
