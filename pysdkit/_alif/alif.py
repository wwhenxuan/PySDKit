# -*- coding: utf-8 -*-
"""
Adaptive Local Iterative Filtering (ALIF).

MATLAB reference: https://github.com/Cicone/ALIF
Paper: Cicone, A., Liu, J., Zhou, H. Adaptive Local Iterative Filtering for Signal
Decomposition and Instantaneous Frequency analysis. ACHA, 41(2):384-411, 2016.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

from pysdkit._alif._helpers import (
    adaptive_average,
    load_prefixed_filter,
    maxmins,
)
from pysdkit._alif.iterative_filtering import IterativeFiltering


class ALIF(object):
    """
    Adaptive Local Iterative Filtering.

    Decomposes a (approximately) periodic signal into IMFs plus a trend:
        signal ≈ IMF_1 + ... + IMF_K + trend
    """

    def __init__(
        self,
        delta: float = 1e-4,
        ext_points: int = 3,
        max_imfs: int = 100,
        xi: float = 1.6,
        max_inner: int = 500,
        max_time: float = np.inf,
        verbose: int = 0,
    ) -> None:
        """
        :param delta: relative energy stopping criterion for the inner loop
        :param ext_points: stop when fewer than this many extrema remain
        :param max_imfs: maximum number of IMFs (excluding the trend)
        :param xi: mask-length scale factor (``ALIF.xi`` in the MATLAB code)
        :param max_inner: maximum inner-loop iterations per IMF
        :param max_time: wall-clock time budget in seconds
        :param verbose: print progress when > 0
        """
        self.delta = delta
        self.ext_points = ext_points
        self.max_imfs = max_imfs
        self.xi = xi
        self.max_inner = max_inner
        self.max_time = max_time
        self.verbose = verbose

        self.imfs: Optional[np.ndarray] = None
        self.mask_lengths: Optional[np.ndarray] = None

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return self.fit_transform(signal)

    def __str__(self) -> str:
        return "Adaptive Local Iterative Filtering (ALIF)"

    def _instantaneous_periods(
        self, signal: np.ndarray, extrema: np.ndarray
    ) -> np.ndarray:
        """Cubic-interpolate inter-extrema spacings onto the sample grid."""
        n = signal.size
        # MATLAB: T_f = [diff(maxmins) (maxmins(1)+N-maxmins(end))]
        # extrema are 0-based; convert spacings consistently
        ext = extrema.astype(int)
        t_f = np.diff(ext).astype(np.float64)
        t_f = np.append(t_f, ext[0] + n - ext[-1])

        # Replicate extrema / periods for smooth periodic interpolation
        temp_t = np.tile(t_f, 11)
        temp_ext = np.concatenate([ext + k * n for k in range(11)]).astype(np.float64)
        # Query 1..11N in MATLAB → 0..11N-1 in Python, then take center block
        query = np.arange(1, 11 * n + 1, dtype=np.float64)
        # Interpolate with cubic; MATLAB interp1(...,'cubic')
        # Use 1-based style abscissa: extrema positions as MATLAB indices
        temp_ext_m = temp_ext + 1.0
        interp = interp1d(
            temp_ext_m,
            temp_t,
            kind="cubic",
            fill_value="extrapolate",
            assume_sorted=False,
        )
        temp_i = interp(query)
        return np.asarray(temp_i[5 * n : 6 * n], dtype=np.float64)

    def _mask_from_periods(self, i_t: np.ndarray) -> np.ndarray:
        """
        Smooth the instantaneous-period curve with IF and scale by ``2 * xi``.
        """
        n_try = 1
        i_t0 = i_t.copy()
        while True:
            iff = IterativeFiltering(
                delta=0.001,
                ext_points=3,
                max_imfs=n_try,
                xi=1.6,
                alpha=1.0,
                max_inner=200,
                verbose=0,
            )
            imf_it = iff.fit_transform(i_t0)
            trend = imf_it[-1]
            n_imfs = imf_it.shape[0] - 1
            if trend.min() <= 0 and n_imfs == n_try:
                n_try += 1
                if n_try > 20:
                    # Fall back to a positive smooth version of i_t
                    trend = np.maximum(i_t, 1.0)
                    break
            elif trend.min() <= 0:
                trend = np.maximum(trend, 1.0)
                break
            else:
                break

        return 2.0 * self.xi * np.asarray(trend, dtype=np.float64)

    def fit_transform(self, signal: np.ndarray, return_masks: bool = False):
        """
        Run ALIF on a 1-D signal.

        :param signal: input samples (treated as one period of a periodic signal)
        :param return_masks: if True, also return the mask-length functions
        :return: IMFs with shape ``(n_imfs + 1, N)`` (last row = trend),
                 optionally ``mask_lengths`` with shape ``(n_imfs, N)``
        """
        f = np.asarray(signal, dtype=np.float64).ravel().copy()
        n = f.size
        mm = load_prefixed_filter()

        imfs = []
        masks = []
        t0 = time.perf_counter()

        extrema = maxmins(f, extension_type="p")
        while (
            extrema.size > self.ext_points
            and len(imfs) < self.max_imfs
            and (time.perf_counter() - t0) < self.max_time
        ):
            if self.verbose > 0:
                print("ALIF: extracting IMF #{}".format(len(imfs) + 1))

            i_t = self._instantaneous_periods(f, extrema)
            mask_len = self._mask_from_periods(i_t)

            if np.ceil(np.max(mask_len)) >= np.floor(n / 2.0):
                if self.verbose > 0:
                    print(
                        "Mask length exceeds half the signal; "
                        "try reducing xi. Stopping IMF extraction."
                    )
                break

            masks.append(mask_len.copy())
            h = f.copy()
            sd = np.inf
            in_step = 0
            while sd > self.delta and in_step < self.max_inner:
                in_step += 1
                ave = adaptive_average(h, mask_len, mm)
                denom = np.linalg.norm(h) ** 2
                sd = (np.linalg.norm(ave) ** 2 / denom) if denom > 0 else 0.0
                h = h - ave
                if self.verbose > 1 and in_step % 10 == 0:
                    print("  step {:3d}  SD={:.6e}".format(in_step, sd))

            imfs.append(h)
            f = f - h
            extrema = maxmins(f, extension_type="p")

        imfs.append(f)
        self.imfs = np.vstack(imfs)
        self.mask_lengths = np.vstack(masks) if masks else np.zeros((0, n))

        if return_masks:
            return self.imfs, self.mask_lengths
        return self.imfs
