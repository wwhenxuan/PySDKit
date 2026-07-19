# -*- coding: utf-8 -*-
"""
Created on 2025/07/20
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Adaptive Polymorphic Mode Decomposition (APMD).

Huang Z. and Liu J., Digital Signal Processing, 161:104913, 2025.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm
from scipy.signal import hilbert
from scipy.stats import kurtosis

try:
    from scipy.integrate import cumulative_trapezoid, trapezoid
except ImportError:  # pragma: no cover
    from scipy.integrate import cumtrapz as cumulative_trapezoid
    from scipy.integrate import trapz as trapezoid


class APMD(object):
    """
    Adaptive Polymorphic Mode Decomposition.

    APMD is a TFR-based framework that alternately extracts
    frequency-dominant (AM-FM / chirp) and time-dominant (impulsive /
    dispersive) modes. Each step:

    1. chooses an optimal STFT window length by Rényi entropy;
    2. detects an initial ridge and bandwidth on the TFR;
    3. jointly refines ridges / bandwidths by iteration;
    4. restores the mode by integrating the TFR inside the bandwidth;
    5. squeezes the mode energy onto the ridge for a sharper TFR.

    Huang Z. and Liu J., "Adaptive Polymorphic Mode Decomposition",
    Digital Signal Processing, 161:104913, 2025.
    """

    def __init__(
        self,
        n_modes: int = 10,
        d: Optional[int] = None,
        thd: float = 0.05,
        thd2: float = 0.05,
        fs: Optional[float] = None,
        max_inner_iter: int = 10,
    ) -> None:
        """
        :param n_modes: maximum number of modes (including the residual)
        :param d: initial half bandwidth in TFR bins; defaults to ``round(0.02 * N)``
        :param thd: residual-energy threshold for stopping mode separation
        :param thd2: relative mode-change threshold for the inner iteration
        :param fs: default sampling frequency (overridable in ``fit_transform``)
        :param max_inner_iter: maximum iterations inside ``single_mod_ext``
        """
        self.n_modes = int(n_modes)
        self.d = d
        self.thd = float(thd)
        self.thd2 = float(thd2)
        self.fs = fs
        self.max_inner_iter = int(max_inner_iter)

        self.modes: Optional[np.ndarray] = None
        self.tfr_modes: Optional[np.ndarray] = None
        self.tfr_squeezed: Optional[np.ndarray] = None
        self.if_mode: Optional[np.ndarray] = None
        self.it_mode: Optional[np.ndarray] = None
        self.u_mode: Optional[np.ndarray] = None
        self.v_mode: Optional[np.ndarray] = None
        self.wl_opt: Optional[np.ndarray] = None
        self.voh: Optional[np.ndarray] = None
        self.elapsed: Optional[float] = None

    def __call__(
        self,
        signal: np.ndarray,
        fs: Optional[float] = None,
        return_all: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Allow instances to be called like functions."""
        return self.fit_transform(signal=signal, fs=fs, return_all=return_all)

    def __str__(self) -> str:
        return "Adaptive Polymorphic Mode Decomposition (APMD)"

    def fit_transform(
        self,
        signal: np.ndarray,
        fs: Optional[float] = None,
        return_all: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run APMD on a univariate signal.

        :param signal: 1-D time-domain signal
        :param fs: sampling frequency; required unless set in ``__init__``
        :param return_all: if True, return a dict with modes, TFRs, ridges, etc.
        :return: modes of shape ``(n_extracted, N)`` or a result dictionary
        """
        signal = np.asarray(signal, dtype=float).ravel()
        n = signal.size
        if n < 8:
            raise ValueError("signal length must be at least 8 samples")

        fs = self.fs if fs is None else fs
        if fs is None:
            raise ValueError("fs must be provided either in __init__ or fit_transform")
        fs = float(fs)

        d = self.d if self.d is not None else max(1, int(round(0.02 * n)))
        d = max(int(d), 1)

        n_modes = max(int(self.n_modes), 1)
        t0 = time.perf_counter()

        # Frequency axis length matches sub_tfrstft positive bins.
        lf = n // 2
        x_mode = np.zeros((n, n_modes), dtype=float)
        x_tfr = np.zeros((lf, n, n_modes), dtype=complex)
        if_mode = np.zeros((n, n_modes), dtype=float)
        it_mode = np.zeros((n, n_modes), dtype=float)
        u_mode = np.zeros((n, n_modes), dtype=float)
        v_mode = np.zeros((n, n_modes), dtype=float)
        wl_opt = np.zeros(n_modes, dtype=float)
        voh = np.zeros(n_modes, dtype=float)

        x_iter = signal.copy()
        i = 0
        while norm_sq_ratio(signal, x_mode) > self.thd and i < n_modes - 1:
            wl = optwin(x_iter)
            wl_opt[i] = wl
            x_plane, _ = sub_tfrstft(x_iter, wl, fs)
            xm, if_i, it_i, u_i, v_i, voh_i = single_mod_ext(
                x_iter,
                x_plane,
                fs,
                wl_opt=int(wl),
                d=d,
                thd=self.thd2,
                max_iter=self.max_inner_iter,
            )
            x_mode[:, i] = xm
            if_mode[:, i] = _pad_to(if_i, n)
            it_mode[:, i] = _pad_to(it_i, n)
            u_mode[:, i] = _pad_to(u_i, n)
            v_mode[:, i] = _pad_to(v_i, n)
            voh[i] = voh_i
            x_iter = x_iter - xm
            x_tfr[:, :, i], _ = sub_tfrstft(xm, int(wl), fs)
            i += 1

        # Residual as the last mode
        wl_opt[i] = 5 * (2 ** (nextpow2(n / 5.0) - 1))
        x_mode[:, i] = signal - np.sum(x_mode[:, :i], axis=1)
        x_tfr[:, :, i], _ = sub_tfrstft(x_mode[:, i], int(wl_opt[i]), fs)
        n_keep = i + 1

        # Drop unused slots (wl_opt == 0)
        keep = wl_opt[:n_keep] != 0
        # Residual window is never zero; extracted modes with wl==0 are empty.
        x_mode = x_mode[:, :n_keep][:, keep]
        x_tfr = x_tfr[:, :, :n_keep][:, :, keep]
        if_mode = if_mode[:, :n_keep][:, keep]
        it_mode = it_mode[:, :n_keep][:, keep]
        u_mode = u_mode[:, :n_keep][:, keep]
        v_mode = v_mode[:, :n_keep][:, keep]
        wl_opt = wl_opt[:n_keep][keep]
        voh = voh[:n_keep][keep]

        n_out = x_mode.shape[1]
        x_s = np.zeros((lf, n), dtype=float)
        for k in range(max(n_out - 1, 0)):
            x_s += tf_squeezing(
                x_tfr[:, :, k],
                if_mode[:, k],
                it_mode[:, k],
                int(voh[k]),
            )

        elapsed = time.perf_counter() - t0

        # Store with PySDKit convention: modes as (K, N)
        self.modes = x_mode.T.copy()
        self.tfr_modes = np.transpose(x_tfr, (2, 0, 1)).copy()
        self.tfr_squeezed = x_s
        self.if_mode = if_mode.T.copy()
        self.it_mode = it_mode.T.copy()
        self.u_mode = u_mode.T.copy()
        self.v_mode = v_mode.T.copy()
        self.wl_opt = wl_opt.copy()
        self.voh = voh.copy()
        self.elapsed = elapsed

        if return_all:
            return {
                "modes": self.modes,
                "tfr_modes": self.tfr_modes,
                "tfr_squeezed": self.tfr_squeezed,
                "if_mode": self.if_mode,
                "it_mode": self.it_mode,
                "u_mode": self.u_mode,
                "v_mode": self.v_mode,
                "wl_opt": self.wl_opt,
                "voh": self.voh,
                "elapsed": self.elapsed,
            }
        return self.modes


def norm_sq_ratio(x: np.ndarray, modes: np.ndarray) -> float:
    """``||x - sum(modes)||^2 / ||x||^2`` used as the outer stopping criterion."""
    resid = x - np.sum(modes, axis=1)
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(resid, resid) / denom)


def _pad_to(arr: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).ravel()
    out = np.zeros(n, dtype=float)
    m = min(n, arr.size)
    out[:m] = arr[:m]
    return out


def nextpow2(n: float) -> int:
    """Smallest integer ``p`` such that ``2**p >= n`` (MATLAB ``nextpow2``)."""
    n = float(n)
    if n <= 0:
        return 0
    return int(np.ceil(np.log2(n)))


def matlab_smooth(x: np.ndarray, span: int) -> np.ndarray:
    """Centered moving-average smoother matching MATLAB ``smooth(x, span)``."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    span = int(span)
    if span <= 0:
        raise ValueError("span must be a positive integer")
    if span % 2 == 0:
        span += 1
    if span > n:
        span = n if n % 2 == 1 else max(1, n - 1)

    k = (span - 1) // 2
    y = np.empty_like(x)
    for i in range(n):
        left = min(i, k)
        right = min(n - 1 - i, k)
        y[i] = x[i - left : i + right + 1].mean()
    return y


def gausswin(length: int, alpha: float = 2.5) -> np.ndarray:
    """MATLAB-compatible Gaussian window (default ``alpha=2.5``)."""
    length = int(length)
    if length < 1:
        raise ValueError("window length must be >= 1")
    if length == 1:
        return np.ones(1, dtype=float)
    n = np.arange(length, dtype=float)
    center = (length - 1) / 2.0
    return np.exp(-0.5 * (alpha * (n - center) / center) ** 2)


def tfrstft(
    x: np.ndarray,
    n_fft: int,
    window: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Short-time Fourier transform matching Time-Frequency Toolbox ``tfrstft``.

    :return: ``(tfr, freqs_norm)`` with ``tfr`` shape ``(n_fft, len(x))``.
    """
    x = np.asarray(x, dtype=float).ravel()
    x_row = x.size
    h = np.asarray(window, dtype=float).ravel()
    lh = (h.size - 1) / 2.0
    tfr = np.zeros((n_fft, x_row), dtype=complex)

    half_n = int(np.round(n_fft / 2.0)) - 1
    for icol in range(x_row):
        ti = icol
        tau_min = -min(half_n, int(np.floor(lh)), ti)
        tau_max = min(half_n, int(np.floor(lh)), x_row - ti - 1)
        tau = np.arange(tau_min, tau_max + 1)
        indices = np.mod(n_fft + tau, n_fft)
        tfr[indices, icol] = x[ti + tau] * np.conj(h[int(lh) + tau])

    tfr = np.fft.fft(tfr, axis=0)
    freqs = np.arange(n_fft, dtype=float) / float(n_fft)
    return tfr, freqs


def sub_tfrstft(
    x: np.ndarray, length_win: int, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """STFT used by APMD (``sub_tfrstft.m``)."""
    x = np.asarray(x, dtype=float).ravel()
    len_w = x.size
    n2 = len_w // 2
    length_win = int(length_win)
    if length_win % 2 == 0:
        length_win += 1
    h = gausswin(length_win)
    sx, fr = tfrstft(x, n_fft=len_w, window=h)
    sfs = fr[:n2] * fs
    sx = sx[:n2, :len_w] / fs
    return sx, sfs


def renyi_entropy(x: np.ndarray, alpha: float = 2.5) -> float:
    """Rényi entropy of a TFR magnitude grid."""
    xr2a = np.sum(np.abs(x) ** (2.0 * alpha))
    xr2 = np.sum(np.abs(x) ** 2.0)
    if xr2 <= 0.0 or xr2a <= 0.0:
        return np.inf
    return float((alpha * np.log(xr2) - np.log(xr2a)) / (alpha - 1.0))


def awft_cubic_b_spline(wl: int) -> np.ndarray:
    """Cubic B-spline window used by AWFT scale-one STFT."""
    m = int(np.log2(wl - 1) - 1)
    bs = []
    for k in range(2 ** (m + 1) + 1):
        xv = k / (2 ** (m - 1))
        ax = abs(xv - 2.0)
        if 0.0 <= ax < 1.0:
            beta = 2.0 / 3.0 - ax**2 + (ax**3) / 2.0
        elif 1.0 <= ax < 2.0:
            beta = ((2.0 - ax) ** 3) / 6.0
        else:
            beta = 0.0
        bs.append(beta)
    return np.asarray(bs, dtype=float)


def awft_scale_one(x: np.ndarray, wl1: int) -> np.ndarray:
    """B-spline STFT at scale 1 (Liu / Shen adaptive TFR)."""
    x = np.asarray(x, dtype=float).ravel()
    x_l = x.size
    h = awft_cubic_b_spline(wl1)
    h = h / norm(h)
    lh = (h.size - 1) // 2
    tfr = np.zeros((x_l, x_l), dtype=complex)
    tau = np.arange(-lh, lh + 1)
    for ti in range(x_l):
        indices = np.mod(ti + tau + x_l, x_l)
        tfr[indices, ti] = x[indices] * np.conj(h[lh + tau])
    tfr = np.fft.fft(tfr, axis=0)
    if x_l % 2:
        select = np.arange((x_l + 1) // 2)
    else:
        select = np.arange(x_l // 2 + 1)
    return tfr[select, :]


def optwin(x: np.ndarray) -> int:
    """Optimal STFT window length via Rényi entropy (``optwin.m``)."""
    x = np.asarray(x, dtype=float).ravel()
    lx = x.size
    wl0 = 5
    env = np.abs(hilbert(x))
    if np.allclose(env, env.flat[0]):
        kh = 1.0
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            kh = float(kurtosis(env, fisher=False, bias=True) - 2.0)
    if not np.isfinite(kh) or kh <= 0.0:
        kh = 1.0
    m = nextpow2(5000.0 / kh / wl0) - 1
    m_cap = nextpow2(lx / float(wl0)) - 1
    if m > m_cap:
        m = m_cap
    m = max(int(m), 0)

    x_awft = awft_scale_one(x, wl0)
    r = np.zeros(m + 1, dtype=float)
    r[0] = renyi_entropy(x_awft, 2.5)
    for d in range(1, m + 1):
        shift = 2 ** (d - 1)
        indices = np.concatenate([np.arange(lx - shift, lx), np.arange(lx - shift)])
        for _ in range(4):
            x_awft = x_awft + x_awft[:, indices]
        # MATLAB: X(:, [mod(2^d-1,lx):lx, 1:mod(2^d-2,lx)])
        # Use circular roll by (2^d - 1) which matches the intended dyadic
        # re-indexing used in the AWFT scale recursion for practical cases.
        x_awft = np.roll(x_awft, -int(np.mod(2**d - 1, lx)), axis=1)
        r[d] = renyi_entropy(x_awft, 2.5)

    if m == 0:
        return wl0
    m_opt = int(np.argmin(r[1:]) + 1)
    return int(wl0 * (2 ** (m_opt - 1)))


def tf_squeezing(
    x_tf: np.ndarray, if_ridge: np.ndarray, it_ridge: np.ndarray, voh: int
) -> np.ndarray:
    """Synchro-squeezing of one mode TFR (``TFsqueezing.m``)."""
    lf, lt = x_tf.shape
    x_s = np.zeros((lf, lt), dtype=float)
    if_ridge = np.asarray(if_ridge).ravel()
    it_ridge = np.asarray(it_ridge).ravel()
    if voh == 0:
        energy = trapezoid(np.abs(x_tf) ** 2, axis=0)
        for i in range(lt):
            idx = int(np.clip(np.round(if_ridge[i]), 0, lf - 1))
            x_s[idx, i] = energy[i]
    else:
        energy = trapezoid(np.abs(x_tf) ** 2, axis=1)
        for i in range(min(lf, it_ridge.size)):
            idx = int(np.clip(np.round(it_ridge[i]), 0, lt - 1))
            x_s[i, idx] = energy[i]
    return x_s


def ifit_find(x_tf: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Initial ridge detection (``IFITfind``). Indices are 0-based."""
    x_abs = np.abs(x_tf)
    lf, lt = x_abs.shape
    d = int(d)

    flat_idx = int(np.argmax(x_abs))
    m_row, m_col = np.unravel_index(flat_idx, x_abs.shape)

    it = np.zeros(lf, dtype=int)
    it_x = np.zeros(lf, dtype=float)
    it[m_row] = m_col

    if m_row != 0:
        for i in range(m_row - 1, -1, -1):
            lo = max(it[i + 1] - d, 0)
            hi = min(it[i + 1] + d, lt - 1)
            mk = np.arange(lo, hi + 1)
            it_x[i] = np.max(x_abs[i + 1, mk])
            weights = x_abs[i + 1, mk]
            it[i] = int(np.round(np.sum(weights * mk) / (np.sum(weights) + 1e-30)))

    if m_row != lf - 1:
        for i in range(m_row + 1, lf):
            lo = max(it[i - 1] - d, 0)
            hi = min(it[i - 1] + d, lt - 1)
            mk = np.arange(lo, hi + 1)
            it_x[i] = np.max(x_abs[i - 1, mk])
            weights = x_abs[i - 1, mk]
            it[i] = int(np.round(np.sum(weights * mk) / (np.sum(weights) + 1e-30)))

    if_ridge = np.zeros(lt, dtype=int)
    if_x = np.zeros(lt, dtype=float)
    if_ridge[m_col] = m_row

    if m_col != 0:
        for i in range(m_col - 1, -1, -1):
            lo = max(if_ridge[i + 1] - d, 0)
            hi = min(if_ridge[i + 1] + d, lf - 1)
            mk = np.arange(lo, hi + 1)
            if_x[i] = np.max(x_abs[mk, i + 1])
            weights = x_abs[mk, i + 1]
            if_ridge[i] = int(
                np.round(np.sum(weights * mk) / (np.sum(weights) + 1e-30))
            )

    if m_col != lt - 1:
        for i in range(m_col + 1, lt):
            lo = max(if_ridge[i - 1] - d, 0)
            hi = min(if_ridge[i - 1] + d, lf - 1)
            mk = np.arange(lo, hi + 1)
            if_x[i] = np.max(x_abs[mk, i - 1])
            weights = x_abs[mk, i - 1]
            if_ridge[i] = int(
                np.round(np.sum(weights * mk) / (np.sum(weights) + 1e-30))
            )

    if norm(it_x) > norm(if_x):
        if_ridge = np.arange(lf, dtype=int)
        voh = 1
    else:
        it = np.arange(lt, dtype=int)
        voh = 0
    return if_ridge, it, voh


def ifit_opt(
    x_tf: np.ndarray,
    if0: np.ndarray,
    it0: np.ndarray,
    um: np.ndarray,
    vm: np.ndarray,
    voh: int,
    d: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Update ridges given bandwidths (``IFITopt``)."""
    x_abs = np.abs(x_tf)
    lf, lt = x_abs.shape
    if_ridge = np.zeros(lt, dtype=int)
    it = np.zeros(lf, dtype=int)
    if0 = np.asarray(if0).ravel()
    it0 = np.asarray(it0).ravel()
    um = np.asarray(um).ravel()
    vm = np.asarray(vm).ravel()

    if voh == 0:
        for i in range(lt):
            lo = int(np.round(if0[i] - vm[i]))
            hi = int(np.round(if0[i] + um[i]))
            mk = np.arange(max(0, lo), min(lf - 1, hi) + 1)
            if mk.size == 0:
                if_ridge[i] = int(if0[i])
                continue
            weights = x_abs[mk, i]
            denom = np.sum(weights)
            if denom <= 0:
                if_ridge[i] = int(if0[i])
            else:
                if_ridge[i] = int(np.round(np.sum(weights * mk) / denom))
        if_ridge = np.round(
            matlab_smooth(if_ridge.astype(float), max(3, 2 * d))
        ).astype(int)
        it = it0[:lf].astype(int).copy()
    else:
        for i in range(lf):
            lo = int(np.round(it0[i] - vm[i]))
            hi = int(np.round(it0[i] + um[i]))
            mk = np.arange(max(0, lo), min(lt - 1, hi) + 1)
            if mk.size == 0:
                it[i] = int(it0[i])
                continue
            weights = x_abs[i, mk]
            denom = np.sum(weights)
            if denom <= 0:
                it[i] = int(it0[i])
            else:
                it[i] = int(np.round(np.sum(weights * mk) / denom))
        it = np.round(matlab_smooth(it.astype(float), max(3, d))).astype(int)
        if_ridge = if0[:lt].astype(int).copy() if if0.size >= lt else if0.astype(int)
        if if_ridge.size < lt:
            tmp = np.zeros(lt, dtype=int)
            tmp[: if_ridge.size] = if_ridge
            if_ridge = tmp
    return if_ridge, it


def _optbw_time(
    x: np.ndarray,
    x_tf: np.ndarray,
    f: np.ndarray,
    if_ridge: np.ndarray,
    wl_opt: int,
    w0: float,
    d: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bandwidth optimization for frequency-dominant (time-ridge) modes."""
    lf, lt = x_tf.shape
    d = max(int(d), 1)
    if_ridge = np.asarray(if_ridge, dtype=int).ravel()

    x0 = np.zeros(lt, dtype=float)
    for i in range(lt):
        lo = max(0, int(if_ridge[i]) - d)
        hi = min(int(if_ridge[i]) + d, lf - 1)
        mk = np.arange(lo, hi + 1)
        x0[i] = np.real(2.0 / w0 * trapezoid(x_tf[mk, i], f[mk]))
    u0 = d * np.ones(lt)
    resid_norm = norm(x - x0)
    tau = (norm(u0) ** 2 + norm(u0) ** 2) / (resid_norm**2 + np.finfo(float).eps)

    xm_cube = np.full((lt, d, d), np.inf, dtype=float)
    for i in range(lt):
        for k in range(1, d + 1):
            lo = max(0, int(if_ridge[i]) - k)
            hi = min(int(if_ridge[i]) + d, lf - 1)
            segment = np.real(x_tf[lo : hi + 1, i])
            tmp = cumulative_trapezoid(segment, initial=0.0)
            vals = (2.0 / w0) * tmp[(k + 1) :]
            if vals.size == 0:
                continue
            start = d - vals.size
            if start < 0:
                vals = vals[-d:]
                start = 0
            xm_cube[i, start : start + vals.size, k - 1] = vals

    dt = int(np.floor(wl_opt / 2.0 + 1.0))
    if dt > lt // 2:
        dt = max(1, lt // 2)

    uu = np.arange(1, d + 1, dtype=float)
    u2v2 = np.add.outer(uu**2, uu**2)
    l1 = np.zeros((lt, d, d), dtype=float)
    l2 = np.zeros((lt, d, d), dtype=float)

    l1[0] = dt * u2v2
    dt_use = min(dt, lt)
    diff0 = x[:dt_use, None, None] - xm_cube[:dt_use]
    l2[0] = np.sum(np.abs(diff0) ** 2, axis=0)

    for i in range(1, lt):
        i_m = i + 1  # 1-based MATLAB index
        if i_m <= dt:
            l1[i] = (i_m + dt - 1) * u2v2
            j = i_m + dt - 2  # 0-based
            if 0 <= j < lt:
                l2[i] = l2[i - 1] + np.abs(x[j] - xm_cube[j]) ** 2
            else:
                l2[i] = l2[i - 1]
        elif i_m <= lt - dt + 1:
            l1[i] = (wl_opt + 1) * u2v2
            j_add = i_m + dt - 2
            j_sub = i_m - dt - 1
            l2[i] = l2[i - 1]
            if 0 <= j_add < lt:
                l2[i] = l2[i] + np.abs(x[j_add] - xm_cube[j_add]) ** 2
            if 0 <= j_sub < lt:
                l2[i] = l2[i] - np.abs(x[j_sub] - xm_cube[j_sub]) ** 2
        else:
            l1[i] = (lt - i_m + dt) * u2v2
            j_sub = i_m - dt - 1
            if 0 <= j_sub < lt:
                l2[i] = l2[i - 1] - np.abs(x[j_sub] - xm_cube[j_sub]) ** 2
            else:
                l2[i] = l2[i - 1]

    loss = l1 + tau * l2
    um = np.zeros(lt, dtype=float)
    vm = np.zeros(lt, dtype=float)
    for i in range(lt):
        flat = int(np.nanargmin(loss[i]))
        ii, jj = np.unravel_index(flat, (d, d))
        um[i] = ii + 1
        vm[i] = jj + 1

    um = np.round(matlab_smooth(um, max(3, 2 * d)))
    vm = np.round(matlab_smooth(vm, max(3, 2 * d)))
    xm = np.zeros(lt, dtype=float)
    for i in range(lt):
        if if_ridge[i] - vm[i] <= 0:
            vm[i] = float(if_ridge[i])
        if if_ridge[i] + um[i] >= lf - 1:
            um[i] = float(lf - 1 - if_ridge[i])
        lo = int(max(0, if_ridge[i] - vm[i]))
        hi = int(min(lf - 1, if_ridge[i] + um[i]))
        xm[i] = np.real(2.0 / w0 * trapezoid(x_tf[lo : hi + 1, i]))
    return xm, um, vm


def _optbw_freq(
    x_tf: np.ndarray,
    f: np.ndarray,
    it: np.ndarray,
    w0: float,
    d: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bandwidth optimization for time-dominant (frequency-ridge / GD) modes."""
    lf, lt = x_tf.shape
    d_time = max(2 * int(d), 2)
    it = np.asarray(it, dtype=int).ravel()

    x0 = x_tf.copy()
    for i in range(lf):
        x0[i, int(np.clip(it[i], 0, lt - 1))] = 0.0
    mean_abs = np.mean(np.abs(x0), axis=0)
    tau = 2.0 * (d_time**2) / (norm(mean_abs) ** 2 + np.finfo(float).eps)

    uu = np.arange(1, d_time + 1, dtype=float)
    u2v2 = np.add.outer(uu**2, uu**2)
    l1 = np.broadcast_to(u2v2, (lf, d_time, d_time)).copy()
    l2 = np.zeros((lf, d_time, d_time), dtype=float)

    for i in range(lf):
        x_block = np.concatenate(
            [np.zeros(d_time + 1), np.abs(x0[i]) ** 2, np.zeros(d_time + 1)]
        )
        ridge = int(np.clip(it[i], 0, lt - 1)) + d_time + 1  # 0-based ridge in x_block
        left_seg = x_block[ridge - d_time : ridge]
        xl = np.cumsum(left_seg)[::-1] + np.sum(x_block[: ridge - d_time])
        right_seg = x_block[ridge + 1 : ridge + 1 + d_time]
        xr = np.cumsum(right_seg[::-1])[::-1]
        # MATLAB sums X_block(IT_tmp+D+1 : lt) with original lt as upper bound
        upper = min(lt, x_block.size)
        xr = xr + np.sum(x_block[ridge + d_time + 1 : upper])
        l2[i] = np.add.outer(xr, xl)

    loss = l1 + tau * l2
    um = np.zeros(lf, dtype=float)
    vm = np.zeros(lf, dtype=float)
    for i in range(lf):
        flat = int(np.nanargmin(loss[i]))
        ii, jj = np.unravel_index(flat, (d_time, d_time))
        um[i] = ii + 1
        vm[i] = jj + 1

    um = np.round(matlab_smooth(um, max(3, d_time)))
    vm = np.round(matlab_smooth(vm, max(3, d_time)))

    x_masked = x_tf.copy()
    for i in range(lf):
        if it[i] - vm[i] <= 0:
            vm[i] = float(it[i])
        if it[i] + um[i] >= lt - 1:
            um[i] = float(lt - 1 - it[i])
        lo = int(max(0, it[i] - vm[i]))
        hi = int(min(lt - 1, it[i] + um[i]))
        x_masked[i, :lo] = 0.0
        x_masked[i, hi + 1 :] = 0.0

    xm = np.zeros(lt, dtype=float)
    t_lo = int(np.min(it - vm))
    t_hi = int(np.max(it + um))
    t_lo = max(0, t_lo)
    t_hi = min(lt - 1, t_hi)
    for i in range(t_lo, t_hi + 1):
        xm[i] = np.real(2.0 / w0 * trapezoid(x_masked[:, i], f))
    return xm, um, vm


def optbw(
    x: np.ndarray,
    x_tf: np.ndarray,
    f: np.ndarray,
    if_ridge: np.ndarray,
    it: np.ndarray,
    wl_opt: int,
    w0: float,
    d: int,
    voh: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bandwidth optimization (``optbw``)."""
    if voh == 0:
        return _optbw_time(x, x_tf, f, if_ridge, wl_opt, w0, d)
    return _optbw_freq(x_tf, f, it, w0, d)


def single_mod_ext(
    x: np.ndarray,
    x_tf: np.ndarray,
    fs: float,
    wl_opt: int,
    d: int,
    thd: float,
    max_iter: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract one polymorphic mode (``single_mod_ext.m``)."""
    x = np.asarray(x, dtype=float).ravel()
    lf, lt = x_tf.shape
    f = np.linspace(0.0, fs / 2.0, lf)
    win = gausswin(int(wl_opt))
    w0 = float(np.max(win / (norm(win) + np.finfo(float).eps)))

    if0, it0, voh = ifit_find(x_tf, d)
    if voh == 0:
        xm0 = np.zeros(lt, dtype=float)
        for i in range(lt):
            lo = max(0, int(if0[i]) - d)
            hi = min(int(if0[i]) + d, lf - 1)
            mk = np.arange(lo, hi + 1)
            xm0[i] = np.real(2.0 / w0 * trapezoid(x_tf[mk, i], f[mk]))
        um0 = d * np.ones(lt, dtype=float)
        vm0 = d * np.ones(lt, dtype=float)
    else:
        xm0 = np.zeros(lt, dtype=float)
        x0 = x_tf.copy()
        for i in range(lf):
            lo = max(0, int(it0[i]) - d)
            hi = min(int(it0[i]) + d, lt - 1)
            x0[i, :lo] = 0.0
            x0[i, hi + 1 :] = 0.0
        for i in range(lt):
            xm0[i] = np.real(2.0 / w0 * trapezoid(x0[:, i], f))
        um0 = d * np.ones(lf, dtype=float)
        vm0 = d * np.ones(lf, dtype=float)

    um_tau = d * np.ones(lt, dtype=float)
    tau = (norm(um_tau) ** 2 + norm(um_tau) ** 2) / (
        norm(x - xm0) ** 2 + np.finfo(float).eps
    )

    if_cur = if0.copy()
    it_cur = it0.copy()
    um, vm, xm = um0.copy(), vm0.copy(), xm0.copy()

    for _ in range(max_iter):
        xm_new, um_new, vm_new = optbw(
            x, x_tf, f, if_cur, it_cur, int(wl_opt), w0, d, voh
        )
        if_new, it_new = ifit_opt(x_tf, if_cur, it_cur, um_new, vm_new, voh, d)

        cost0 = norm(um) ** 2 + norm(vm) ** 2 + tau * norm(xm - x) ** 2
        cost1 = norm(um_new) ** 2 + norm(vm_new) ** 2 + tau * norm(xm_new - x) ** 2
        if cost0 > cost1:
            rel = norm(xm_new - xm) ** 2 / (norm(xm) ** 2 + np.finfo(float).eps)
            if_cur, it_cur = if_new, it_new
            um, vm, xm = um_new, vm_new, xm_new
            if rel <= thd:
                break
            continue
        break

    if voh == 0:
        u = np.clip(if_cur + um, 0, lf - 1)
        v = np.clip(if_cur - vm, 0, lf - 1)
        if_out = if_cur.astype(float)
        it_out = it_cur.astype(float)
        if it_out.size < lt:
            tmp = np.zeros(lt, dtype=float)
            tmp[: it_out.size] = it_out
            it_out = tmp
    else:
        u = np.zeros(lt, dtype=float)
        v = np.zeros(lt, dtype=float)
        u[:lf] = np.clip(it_cur + um, 0, lt - 1)
        v[:lf] = np.clip(it_cur - vm, 0, lt - 1)
        if_out = np.zeros(lt, dtype=float)
        it_out = np.zeros(lt, dtype=float)
        n_if = min(lf, if_cur.size)
        if_out[:n_if] = if_cur[:n_if]
        it_out[:lf] = it_cur[:lf]

    return xm, if_out, it_out, u, v, int(voh)
