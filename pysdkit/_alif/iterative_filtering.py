# -*- coding: utf-8 -*-
"""
Iterative Filtering (IF) — subroutine used by Adaptive Local Iterative Filtering.

MATLAB reference: IF_v8_3.m from https://github.com/Cicone/ALIF
"""
from __future__ import annotations

import os
import time
from typing import Optional, Tuple, Union

import numpy as np

_FILTER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "prefixed_double_filter.npy",
)

_MM_CACHE: Optional[np.ndarray] = None


def load_prefixed_filter() -> np.ndarray:
    """Load the prefixed double filter shipped with the package."""
    global _MM_CACHE
    if _MM_CACHE is None:
        if not os.path.isfile(_FILTER_PATH):
            raise FileNotFoundError("Missing ALIF filter data: {}".format(_FILTER_PATH))
        _MM_CACHE = np.load(_FILTER_PATH).astype(np.float64).ravel()
    return _MM_CACHE


def get_mask_v1(y: np.ndarray, k: float) -> np.ndarray:
    """
    Resample the prefixed filter ``y`` to a mask of half-width ``k``.

    Port of MATLAB ``get_mask_v1`` from ALIFv5_4.m / IF_v8_3.m.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    m = (n - 1) / 2.0

    if k < 0:
        return np.array([], dtype=np.float64)

    if k <= m:
        if abs(k - round(k)) < 1e-12:
            k_int = int(round(k))
            a = np.zeros(2 * k_int + 1, dtype=np.float64)
            for i in range(1, 2 * k_int + 2):
                s = (i - 1) * (2 * m + 1) / (2 * k_int + 1) + 1
                t = i * (2 * m + 1) / (2 * k_int + 1)
                s2 = np.ceil(s) - s
                t1 = t - np.floor(t)
                cs = int(np.ceil(s))
                ft = int(np.floor(t))
                mid = y[cs - 1 : ft].sum() if ft >= cs else 0.0
                a[i - 1] = mid + s2 * y[cs - 1] + t1 * y[ft - 1]
            return a

        new_k = int(np.floor(k))
        extra = k - new_k
        c = (2 * m + 1) / (2 * new_k + 1 + 2 * extra)
        a = np.zeros(2 * new_k + 3, dtype=np.float64)

        t = extra * c + 1
        t1 = t - np.floor(t)
        ft = int(np.floor(t))
        a[0] = y[:ft].sum() + t1 * y[ft - 1]

        for i in range(2, 2 * new_k + 3):
            s = extra * c + (i - 2) * c + 1
            t = extra * c + (i - 1) * c
            s2 = np.ceil(s) - s
            t1 = t - np.floor(t)
            cs = int(np.ceil(s))
            ft = int(np.floor(t))
            mid = y[cs - 1 : ft].sum() if ft >= cs else 0.0
            a[i - 1] = mid + s2 * y[cs - 1] + t1 * y[ft - 1]

        t2 = np.ceil(t) - t
        ct = int(np.ceil(t))
        a[-1] = y[ct - 1 :].sum() + t2 * y[ct - 1]
        return a

    dx = 0.01
    f = y / dx
    dy = m * dx / k
    x_src = np.arange(0.0, m + 1e-12, 1.0)
    x_query = np.arange(0.0, m + 1e-12, m / k)
    b = np.interp(x_query, x_src, f[int(m) : 2 * int(m) + 1])
    a = np.concatenate([b[:0:-1], b]) * dy
    if abs(np.abs(a).sum() - 1.0) > 1e-14:
        a = a / np.abs(a).sum()
    return a


def maxmins(
    f: np.ndarray,
    extension_type: str = "p",
    tol: float = 1e-15,
) -> np.ndarray:
    """
    Locate extrema of ``f`` (0-based indices).

    Port of MATLAB ``Maxmins_v3_3`` (periodic / constant cases used by IF & ALIF).
    """
    f = np.asarray(f, dtype=np.float64).ravel()
    n_old = f.size
    if n_old < 3:
        return np.array([], dtype=int)

    df = np.diff(f)
    h = 0

    if extension_type == "p":
        while h < n_old - 1 and abs(df[h]) <= tol:
            h += 1
        if h >= n_old - 1:
            return np.array([], dtype=int)
        matlab_h = h + 1
        df = np.diff(np.concatenate([f, f[1 : matlab_h + 1]]))
        n_ext = n_old + matlab_h
        loop_start = matlab_h
    else:
        n_ext = n_old
        loop_start = 1
        if extension_type == "c" and abs(df[0]) <= tol:
            while h < n_old - 1 and abs(df[h]) <= tol:
                h += 1
            loop_start = h + 1

    maxs = []
    mins = []
    c = 0
    last_df = 0
    posc = 0

    def _mod_to_0based(matlab_idx: int) -> int:
        m = matlab_idx % n_old
        if m == 0:
            m = n_old
        return m - 1

    for i_mat in range(loop_start, n_ext - 1):
        i0 = i_mat - 1
        if i0 + 1 >= df.size:
            break
        prod = df[i0] * df[i0 + 1]

        if -tol <= prod <= tol:
            if df[i0] < -tol:
                last_df = -1
                posc = i_mat
            elif df[i0] > tol:
                last_df = 1
                posc = i_mat
            c += 1
            if df[i0 + 1] < -tol:
                if last_df == 1:
                    maxs.append(_mod_to_0based(posc + (c - 1) // 2 + 1))
                c = 0
            if df[i0 + 1] > tol:
                if last_df == -1:
                    mins.append(_mod_to_0based(posc + (c - 1) // 2 + 1))
                c = 0

        if prod < -tol:
            if df[i0] < -tol and df[i0 + 1] > tol:
                m = (i_mat + 1) % n_old
                if m == 0:
                    m = 1
                mins.append(m - 1)
                last_df = -1
            elif df[i0] > tol and df[i0 + 1] < -tol:
                m = (i_mat + 1) % n_old
                if m == 0:
                    m = 1
                maxs.append(m - 1)
                last_df = 1

    if c > 0 and extension_type == "c":
        if last_df > 0:
            maxs.append(posc)
        else:
            mins.append(posc)

    if not maxs and not mins:
        return np.array([], dtype=int)

    maxmins_arr = np.sort(np.unique(np.array(maxs + mins, dtype=int)))
    if extension_type == "c" and maxmins_arr.size > 0:
        if maxmins_arr[0] != 0 and maxmins_arr[-1] != n_old - 1:
            maxmins_arr = np.unique(np.concatenate([[0], maxmins_arr, [n_old - 1]]))
    return maxmins_arr.astype(int)


def build_ifft_kernel(mask: np.ndarray, n: int) -> np.ndarray:
    """Embed a short mask into an FFT multiplier of length ``n`` (periodic IF)."""
    a = np.asarray(mask, dtype=np.float64).ravel()
    nza = n - a.size
    if nza < 0:
        raise ValueError("Signal shorter than mask; tile the signal first.")

    if nza % 2 == 0:
        a_pad = np.concatenate([np.zeros(nza // 2), a, np.zeros(nza // 2)])
        half = (a_pad.size - 1) // 2
        a_centered = np.concatenate([a_pad[half:], a_pad[:half]])
    else:
        a_pad = np.concatenate(
            [np.zeros((nza - 1) // 2), a, np.zeros((nza - 1) // 2 + 1)]
        )
        mid = a_pad.size // 2
        a_centered = np.concatenate([a_pad[mid - 1 :], a_pad[: mid - 1]])

    return np.real(np.fft.fft(a_centered))


def adaptive_average(
    h: np.ndarray, mask_lengths: np.ndarray, mm: np.ndarray
) -> np.ndarray:
    """
    Position-dependent moving average used by ALIF (``ave = W * h``).

    Implemented without forming the dense ``N x N`` matrix.
    """
    h = np.asarray(h, dtype=np.float64).ravel()
    mask_lengths = np.asarray(mask_lengths, dtype=np.float64).ravel()
    n = h.size
    ave = np.zeros(n, dtype=np.float64)

    for i in range(n):
        k = float(mask_lengths[i])
        if k <= 0:
            continue
        wn = get_mask_v1(mm, k)
        if wn.size == 0:
            continue
        norm1 = np.abs(wn).sum()
        if norm1 <= 0:
            continue
        wn = wn / norm1
        half = (wn.size - 1) // 2
        idxs = np.arange(i - half, i + half + 1) % n
        if idxs.size != wn.size:
            continue
        ave[i] = float(np.dot(wn, h[idxs]))
    return ave


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
