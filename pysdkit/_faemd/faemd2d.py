# -*- coding: utf-8 -*-
"""
Bidimensional Fast and Adaptive Empirical Mode Decomposition (FAEMD2D)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from pysdkit._faemd.filter import (
    filter_size_2d,
    identify_max_min_2d,
    immse,
    mean_envelope_2d,
)


class FAEMD2D(object):
    """
    Bidimensional Fast and Adaptive Empirical Mode Decomposition (FAEMD2D)

    Thirumalaisamy, Mruthun R., and Phillip J. Ansell.
    “Fast and Adaptive Empirical Mode Decomposition for Multidimensional,
    Multivariate Signals.” IEEE Signal Processing Letters, 25(10):1550–1554, 2018.

    Extends FA-EMD to images / 2-D fields.  Extrema are detected with an
    8-neighbour ring; the adaptive OSF window comes from Delaunay nearest
    neighbour distances (Bhuiyan et al.).  Multivariate channels share one
    window estimated on the equal-weight projection ``Σ H_i / √n``.
    """

    def __init__(
        self,
        max_imfs: int = 3,
        tol: float = 0.05,
        window_type: int = 5,
    ) -> None:
        """
        :param max_imfs: Number of returned modes (last mode = residue)
        :param tol: Per-channel sifting MSE tolerance (MATLAB 2-D default 0.05)
        :param window_type: Adaptive window selector in ``0..6``
            (MATLAB default type 6 → index 5)
        """
        if not isinstance(max_imfs, (int, np.integer)) or max_imfs < 1:
            raise ValueError("`max_imfs` must be a positive integer")
        if window_type not in range(7):
            raise ValueError("`window_type` must be an integer in 0..6")
        if tol <= 0:
            raise ValueError("`tol` must be positive")

        self.max_imfs = int(max_imfs)
        self.tol = float(tol)
        self.window_type = int(window_type)

        self.imfs: Optional[np.ndarray] = None
        self.residue: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return "Bidimensional Fast and Adaptive Empirical Mode Decomposition (FAEMD2D)"

    def __call__(
        self, signal: np.ndarray, max_imfs: Optional[int] = None
    ) -> np.ndarray:
        return self.fit_transform(signal, max_imfs=max_imfs)

    @staticmethod
    def _as_channels(signal: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Normalize input to ``(n_channels, H, W)``.

        Returns the array and a flag ``univariate`` (True if original was 2-D).
        """
        x = np.asarray(signal, dtype=float)
        if x.ndim == 2:
            return x[np.newaxis, ...], True
        if x.ndim == 3:
            return x, False
        raise ValueError("FAEMD2D expects shape (H, W) or (n_channels, H, W)")

    def fit_transform(
        self,
        signal: np.ndarray,
        max_imfs: Optional[int] = None,
        return_all: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Decompose a 2-D image or multi-channel 2-D field.

        :param signal: ``(H, W)`` or ``(n_channels, H, W)``
        :param max_imfs: Override number of modes
        :param return_all: Also return residue / windows / sift counts
        :return: ``(K, H, W)`` or ``(K, n_channels, H, W)`` (last mode = residue)
        """
        channels, univariate = self._as_channels(signal)
        n_ch, height, width = channels.shape
        max_imfs = self.max_imfs if max_imfs is None else int(max_imfs)
        if max_imfs < 1:
            raise ValueError("max_imfs must be a positive integer")

        imfs = np.zeros((max_imfs, n_ch, height, width), dtype=float)
        windows = np.zeros((7, max_imfs), dtype=float)
        sift_count = np.zeros(max_imfs, dtype=int)

        residue = channels.copy()
        imf_idx = 0

        while imf_idx < max_imfs - 1:
            h = residue.copy()
            combined = np.sum(h / np.sqrt(n_ch), axis=0)

            maxima, minima = identify_max_min_2d(combined)
            if np.count_nonzero(maxima) < 3 or np.count_nonzero(minima) < 3:
                break

            windows[:, imf_idx] = filter_size_2d(
                maxima, minima, window_type=self.window_type
            )
            w_sz = int(windows[self.window_type, imf_idx])
            if w_sz == 0:
                # Delaunay failed (collinear extrema)
                break

            sift_stop = False
            while not sift_stop:
                sift_count[imf_idx] += 1
                h1 = np.empty_like(h)
                mse = np.empty(n_ch, dtype=float)
                for i in range(n_ch):
                    mean_env = mean_envelope_2d(h[i], w_sz)
                    h1[i] = h[i] - mean_env
                    mse[i] = immse(h1[i], h[i])
                if np.all(mse < self.tol) and sift_count[imf_idx] != 1:
                    sift_stop = True
                h = h1

            imfs[imf_idx] = h
            residue = residue - h
            imf_idx += 1

        imfs[-1] = residue
        self.residue = residue[0] if univariate else residue
        out = imfs[:, 0, :, :] if univariate else imfs
        self.imfs = out[:-1]

        if return_all:
            return out, self.residue, windows, sift_count
        return out
