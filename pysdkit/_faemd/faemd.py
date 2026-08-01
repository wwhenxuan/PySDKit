# -*- coding: utf-8 -*-
"""
Created on 2025/02/01 22:30:40
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from pysdkit._faemd.extrema import extrema
from pysdkit._faemd.filter import filter_size1D, immse, mean_envelope_1d


class FAEMD(object):
    """
    Fast and Adaptive Empirical Mode Decomposition (1-D / multivariate 1-D)

    Thirumalaisamy, Mruthun R., and Phillip J. Ansell.
    “Fast and Adaptive Empirical Mode Decomposition for Multidimensional,
    Multivariate Signals.” IEEE Signal Processing Letters, 25(10):1550–1554, 2018.

    MATLAB: https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd

    Replaces cubic-spline envelopes with order-statistics filters whose window
    length is adapted from extrema spacings (Bhuiyan / FA-MVEMD).
    """

    def __init__(
        self,
        max_imfs: int = 3,
        tol: Optional[float] = None,
        window_type: int = 0,
    ) -> None:
        """
        :param max_imfs: Number of modes returned (includes the final residue)
        :param tol: Sifting MSE tolerance; default ``min(RMS) * 0.001``
        :param window_type: Adaptive window selector in ``{0,...,6}``
            (MATLAB ``type`` 1..7).  Default ``0`` → smallest spacing ``d1``.
        """
        if not isinstance(max_imfs, (int, np.integer)) or max_imfs < 1:
            raise ValueError("`max_imfs` must be a positive integer")
        if window_type not in range(7):
            raise ValueError("`window_type` must be an integer in 0..6")

        self.max_imfs = int(max_imfs)
        self.tol = tol
        self.window_type = int(window_type)

        self.imfs: Optional[np.ndarray] = None
        self.residue: Optional[np.ndarray] = None

    def __call__(
        self,
        signal: np.ndarray,
        return_all: bool = False,
        max_imfs: Optional[int] = None,
    ):
        return self.fit_transform(
            signal=signal, return_all=return_all, max_imfs=max_imfs
        )

    def __str__(self) -> str:
        return "Fast and Adaptive Empirical Mode Decomposition (FAEMD)"

    def _get_tol(self, signal: np.ndarray) -> float:
        if self.tol is not None:
            return float(self.tol)
        # MATLAB EMD1DNV: min(rms(u)) * 0.001
        rms = np.sqrt(np.mean(np.asarray(signal, dtype=float) ** 2, axis=0))
        return float(np.min(rms) * 0.001)

    def sift(self, h: np.ndarray, w_sz: float) -> np.ndarray:
        """One FA-EMD sifting step on a single channel."""
        mean_env = mean_envelope_1d(h, int(w_sz))
        return h - mean_env

    def fit_transform(
        self,
        signal: np.ndarray,
        return_all: bool = False,
        max_imfs: Optional[int] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Decompose a uni-/multivariate 1-D signal.

        :param signal: ``(seq_len,)`` or ``(n_channels, seq_len)``
        :param return_all: Also return residue, window table and sift counts
        :param max_imfs: Override for the number of returned modes
        :return: IMFs of shape ``(K, seq_len)`` or ``(K, seq_len, n_channels)``
            (the last mode is the residue)
        """
        data, inputs_shape = check_inputs(signal)
        seq_len, num_vars = data.shape
        max_imfs = self.max_imfs if max_imfs is None else int(max_imfs)
        if max_imfs < 1:
            raise ValueError("max_imfs must be a positive integer")

        imfs = np.zeros((seq_len, num_vars, max_imfs), dtype=float)
        h1 = np.zeros((seq_len, num_vars), dtype=float)
        mse = np.zeros(num_vars, dtype=float)
        windows = np.zeros((7, max_imfs), dtype=float)
        sift_count = np.zeros(max_imfs, dtype=int)

        residue = data.copy()
        tol = self._get_tol(data)
        imf_idx = 0

        # Extract up to max_imfs-1 oscillatory modes; last slot stores residue
        while imf_idx < max_imfs - 1:
            h = residue.copy()
            combined = np.sum(h / np.sqrt(num_vars), axis=1)

            ext = extrema(combined)
            if ext[0] is None:
                break
            maxima, max_pos, minima, min_pos = ext
            max_pos = np.atleast_1d(np.asarray(max_pos, dtype=float))
            min_pos = np.atleast_1d(np.asarray(min_pos, dtype=float))
            if max_pos.size < 3 or min_pos.size < 3:
                break

            windows[:, imf_idx] = filter_size1D(
                imax=max_pos, imin=min_pos, window_type=self.window_type
            )
            w_sz = int(windows[self.window_type, imf_idx])

            sift_stop = False
            while not sift_stop:
                sift_count[imf_idx] += 1
                for i in range(num_vars):
                    h1[:, i] = self.sift(h[:, i], w_sz=w_sz)
                    mse[i] = immse(h1[:, i], h[:, i])

                if np.all(mse < tol) and sift_count[imf_idx] != 1:
                    sift_stop = True
                h = h1.copy()

            imfs[:, :, imf_idx] = h
            residue = residue - h
            imf_idx += 1

        if np.any(sift_count >= 5):
            selected = windows[self.window_type, :imf_idx]
            if selected.size >= 2 and np.any(np.diff(selected) <= 0):
                print(
                    "Decomposition may be oversifted; "
                    "filter window size does not increase monotonically."
                )

        imfs[:, :, -1] = residue
        out = check_outputs(imfs, inputs_shape)

        self.imfs = out[:-1] if out.ndim >= 2 else out
        self.residue = residue.T if len(inputs_shape) == 2 else residue[:, 0]

        if return_all:
            return out, self.residue, windows, sift_count
        return out

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.imfs is None or self.residue is None:
            raise ValueError(
                "No IMF found. Please run `fit_transform` method first."
            )
        return self.imfs, self.residue


def check_inputs(signal: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    """Normalize to internal layout ``(seq_len, n_channels)``."""
    signal = np.asarray(signal, dtype=float)
    inputs_shape = signal.shape
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    elif signal.ndim == 2:
        pass
    else:
        raise ValueError(
            "signal must have shape [seq_len] or [n_channels, seq_len]"
        )
    return signal.T, inputs_shape


def check_outputs(imfs: np.ndarray, inputs_shape: Tuple) -> np.ndarray:
    """Map ``(seq_len, n_channels, K)`` back to PySDKit layout."""
    if len(inputs_shape) == 1:
        return np.transpose(imfs[:, 0, :], (1, 0))
    return np.transpose(imfs, (2, 0, 1))


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from pysdkit.data import test_emd, test_multivariate_signal
    from pysdkit.plot import plot_IMFs

    faemd = FAEMD(max_imfs=3)
    _, signal = test_emd()
    imfs = faemd.fit_transform(signal)
    plot_IMFs(signal, imfs)
    plt.show()
