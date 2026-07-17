# -*- coding: utf-8 -*-
"""
Iterative Filtering (IF) — subroutine used by Adaptive Local Iterative Filtering.

MATLAB reference: IF_v8_3.m from https://github.com/Cicone/ALIF
"""
from __future__ import annotations

import time
from typing import Optional, Tuple, Union

import numpy as np

from pysdkit._alif._helpers import (
    build_ifft_kernel,
    get_mask_v1,
    load_prefixed_filter,
    maxmins,
)


class IterativeFiltering(object):
    """
    Iterative Filtering for periodic signals.

    Cicone, A., Liu, J., Zhou, H. Adaptive Local Iterative Filtering for Signal
    Decomposition and Instantaneous Frequency analysis. ACHA, 2016.
    """

    def __init__(
        self,
        delta: float = 0.001,
        ext_points: int = 3,
        max_imfs: int = 200,
        xi: float = 1.6,
        alpha: Union[str, float] = "Almost_min",
        max_inner: int = 200,
        max_time: float = np.inf,
        verbose: int = 0,
    ) -> None:
        self.delta = delta
        self.ext_points = ext_points
        self.max_imfs = max_imfs
        self.xi = xi
        self.alpha = alpha
        self.max_inner = max_inner
        self.max_time = max_time
        self.verbose = verbose
        self.log_m: Optional[np.ndarray] = None

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return self.fit_transform(signal)

    def __str__(self) -> str:
        return "Iterative Filtering (IF)"

    def _mask_length(
        self,
        n_pp: int,
        k_pp: int,
        diff_maxmins: np.ndarray,
        count_imfs: int,
        log_m: np.ndarray,
        m_override: Optional[float],
    ) -> float:
        if m_override is not None:
            return float(m_override)

        avg = 2 * round(n_pp / max(k_pp, 1) * self.xi)
        if isinstance(self.alpha, str):
            if self.alpha == "ave":
                m = avg
            elif self.alpha == "Almost_min":
                p30 = 2 * round(self.xi * float(np.percentile(diff_maxmins, 30)))
                m = p30 if p30 < avg else avg
                if count_imfs > 1 and m <= log_m[count_imfs - 2]:
                    m = float(np.ceil(log_m[count_imfs - 2] * 1.1))
            else:
                raise ValueError("Unrecognized IF.alpha value: {}".format(self.alpha))
        else:
            alpha = float(self.alpha)
            m = 2 * round(
                self.xi
                * (
                    float(np.max(diff_maxmins)) * alpha
                    + float(np.min(diff_maxmins)) * (1.0 - alpha)
                )
            )
        return float(m)

    def fit_transform(
        self,
        signal: np.ndarray,
        mask_lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Decompose a periodic signal into IMFs + trend.

        :return: array of shape ``(n_imfs + 1, N)``; last row is the trend.
        """
        f = np.asarray(signal, dtype=np.float64).ravel().copy()
        n = f.size
        mm = load_prefixed_filter()

        f_pp = f[np.abs(f) > 1e-18]
        maxmins_pp = maxmins(f_pp, extension_type="p")
        if maxmins_pp.size < 2:
            return f.reshape(1, -1)
        diff_maxmins_pp = np.diff(maxmins_pp)
        # Periodic wrap for spacing (MATLAB uses plain diff on extrema of zero-stripped signal)
        n_pp = f_pp.size
        k_pp = maxmins_pp.size

        imfs = []
        log_m = np.zeros(self.max_imfs, dtype=np.float64)
        count = 0
        t0 = time.perf_counter()

        while (
            count < self.max_imfs
            and k_pp >= self.ext_points
            and (time.perf_counter() - t0) < self.max_time
        ):
            count += 1
            h = f.copy()
            m_override = None
            if mask_lengths is not None and len(mask_lengths) >= count:
                m_override = float(mask_lengths[count - 1])

            m = self._mask_length(n_pp, k_pp, diff_maxmins_pp, count, log_m, m_override)
            log_m[count - 1] = m
            a = get_mask_v1(mm, m)

            # Tile signal if shorter than the mask
            n_work = n
            n_old = n
            tiled = False
            if n < a.size:
                tiled = True
                nxs = int(np.ceil(a.size / float(n)))
                if nxs % 2 == 0:
                    nxs += 1
                h = np.tile(h, nxs)
                n_work = nxs * n_old

            ifft_a = build_ifft_kernel(a, n_work)

            sd = 1.0
            in_step = 0
            while sd > self.delta and in_step < self.max_inner:
                in_step += 1
                h_ave = np.real(np.fft.ifft(ifft_a * np.fft.fft(h)))
                denom = np.linalg.norm(h) ** 2
                sd = (np.linalg.norm(h_ave) ** 2 / denom) if denom > 0 else 0.0
                h = h - h_ave

            if tiled:
                start = int(n_old * (nxs - 1) / 2)
                h = h[start : start + n_old]

            imfs.append(h)
            f = f - h

            f_pp = f[np.abs(f) > 1e-18]
            maxmins_pp = maxmins(f_pp, extension_type="p")
            if maxmins_pp.size < 2:
                break
            diff_maxmins_pp = np.diff(maxmins_pp)
            n_pp = f_pp.size
            k_pp = maxmins_pp.size

        self.log_m = log_m[:count]
        if not imfs:
            return f.reshape(1, -1)
        return np.vstack([np.asarray(imfs), f])
