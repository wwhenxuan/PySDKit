# -*- coding: utf-8 -*-
"""
Variational Generalized Nonlinear Mode Decomposition (VGNMD).

VGNMD jointly separates *chirp* modes (time-varying IF) and *dispersive* modes
(frequency-varying GD) from a generalized nonlinear signal (GNS).  The pipeline
is:

1. **ATFFC** — adaptive time–frequency fusion & clustering
2. **MTDC** — mode-type discrimination (chirp vs dispersive)
3. **VOA** — variational optimisation: ACMD (chirp) or GDMD (dispersive)

Wang H, Chen S, Zhai W.
Variational generalized nonlinear mode decomposition: Algorithm and applications.
Mechanical Systems and Signal Processing, 206:110913, 2024.
https://doi.org/10.1016/j.ymssp.2023.110913

MATLAB toolbox accompanying the paper (``VGNMD.m``, ``ATFFC.m``, ``MTDC.m``,
``VOA.m``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.linalg import norm
from scipy import ndimage
from scipy.integrate import cumulative_trapezoid
from scipy.signal import hilbert
from scipy.sparse import diags, eye, hstack, vstack
from scipy.sparse.linalg import spsolve

from pysdkit._gdmd.gdmd import (
    curve_smooth,
    differ,
    gdmd_core,
    second_order_difference,
    spectrum_to_time,
)


def stft_vgnmd(
    signal: np.ndarray,
    samp_freq: float,
    n_freq: Optional[int] = None,
    win_len: Optional[float] = None,
    sigma: float = 0.28,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    STFT used by the VGNMD toolbox (Gaussian window, analytic signal).

    Short-time Fourier transform (toolbox ``STFT.m``)

    :return: ``(Spec, f)`` with ``Spec`` shaped ``(n_freq_bins, n_time)``
    """
    x = np.asarray(signal, dtype=float).ravel()
    if np.isrealobj(x):
        x = hilbert(x)
    n = x.size
    if n_freq is None:
        n_freq = n
    if win_len is None:
        win_len = samp_freq / 2.0
    win_len = int(np.ceil(win_len / 2.0) * 2)
    win_len = max(win_len, 2)

    t_win = np.linspace(-1.0, 1.0, win_len)
    win = (np.pi * sigma**2) ** (-0.25) * np.exp(-(t_win**2) / (2.0 * sigma**2))
    lh = (win_len - 1) // 2

    spec = np.zeros((n_freq, n), dtype=complex)
    for i in range(n):
        tau_lo = -min(n_freq // 2 - 1, lh, i)
        tau_hi = min(n_freq // 2 - 1, lh, n - 1 - i)
        tau = np.arange(tau_lo, tau_hi + 1)
        if tau.size == 0:
            continue
        r_sig = x[i + tau] * np.conj(win[lh + tau])
        spec[: tau.size, i] = r_sig

    spec = np.fft.fftshift(np.fft.fft(spec, axis=0), axes=0)
    n_level = spec.shape[0]
    half = n_level // 2
    # positive frequencies (match MATLAB round(end/2)+1:end)
    spec_pos = np.abs(spec[half:, :])
    f = np.linspace(0.0, samp_freq / 2.0, spec_pos.shape[0])
    return spec_pos, f


def tfc(
    spec: np.ndarray, min_frac: float = 0.001
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Time-frequency clustering via connected components (``TFC`` in ``ATFFC.m``).

    ATFFC — adaptive time-frequency fusion & clustering

    Small components (< ``min_frac`` of the TF plane) are treated as noise.
    """
    s = np.asarray(spec, dtype=float)
    labeled, n_lab = ndimage.label(s > 0)
    m, n = s.shape
    min_size = min_frac * m * n
    spec_fc = np.zeros_like(s)
    clusters: List[np.ndarray] = []
    for i in range(1, n_lab + 1):
        mask = labeled == i
        if mask.sum() > min_size:
            cluster = mask.astype(float) * s
            clusters.append(cluster)
            spec_fc += cluster
    return spec_fc, clusters


def atffc(
    signal: np.ndarray,
    samp_freq: float,
    n_windows: int = 5,
    min_frac: float = 0.001,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Adaptive time–frequency fusion and clustering (``ATFFC.m``).

    :return: ``(spec_clusters, f)`` — list of per-mode TF maps and frequency axis
    """
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    th = np.zeros(n_windows + 1)
    spec_fc = None
    f = None
    clusters: List[np.ndarray] = []

    for i in range(n_windows):
        win_len = samp_freq / ((i + 1) * 2.0)
        spec, f = stft_vgnmd(x, samp_freq, n_freq=n, win_len=win_len)
        spec = spec / (spec.max() + 1e-30)
        spec1 = spec.copy()
        if i == 0:
            th[0] = float(np.mean(spec))
        spec_thr = spec.copy()
        spec_thr[spec_thr < th[i]] = 0.0
        spec_c, _ = tfc(spec_thr, min_frac=min_frac)

        if i == 0:
            spec_fc = spec1
            spec_f = spec_c
        else:
            both = (spec_fc > 0) & (spec_c > 0)
            spec_f = np.zeros_like(spec_fc)
            spec_f[both] = np.maximum(spec_fc[both], spec_c[both])

        m1 = spec_fc[spec_fc > 0]
        m2 = spec_f[spec_f > 0]
        if m1.size == 0 or m2.size == 0:
            th[i + 1] = 0.0
        else:
            r = float(np.mean(m1) / (np.mean(m2) + 1e-30))
            if r < 0.8:
                th[i + 1] = float(np.mean(m2) - r * 1.5 * np.std(m2))
            else:
                th[i + 1] = 0.0

        spec_fc, clusters = tfc(spec_f, min_frac=min_frac)

    if f is None:
        raise RuntimeError("ATFFC failed to compute a TF representation")
    if not clusters:
        # fall back: single cluster = final fused map
        if spec_fc is not None and np.any(spec_fc > 0):
            clusters = [spec_fc]
        else:
            raise RuntimeError(
                "ATFFC found no TF clusters; try a higher SNR or different fs"
            )
    return clusters, f


def find_ridges_peak(spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Peak ridge along columns (toolbox ``findridges.m``).

    Ridge helpers & mode-type discrimination

    For each time column with nonzero energy, take the frequency argmax.
    """
    s = np.abs(np.asarray(spec, dtype=float))
    emax = np.max(s, axis=0)
    index_f_all = np.argmax(s, axis=0)
    indext = np.flatnonzero(emax > 0)
    indexf = index_f_all[indext]
    return indext.astype(int), indexf.astype(int)


def _avg_repeat_spacing(coord: np.ndarray) -> float:
    """Average spacing of positions where a coordinate value repeats."""
    vals, counts = np.unique(coord, return_counts=True)
    repeated = vals[counts > 1]
    if repeated.size == 0:
        return np.inf
    total = 0.0
    n_rep = 0
    for v in repeated:
        pos = np.flatnonzero(coord == v)
        if pos.size < 2:
            continue
        total += float(np.sum(np.diff(pos)) / (pos.size - 1))
        n_rep += 1
    if n_rep == 0:
        return np.inf
    return total / n_rep


def mtdc(spec: np.ndarray, ad_thresh: float = 10.0) -> Tuple[int, np.ndarray]:
    """
    Mode-type discrimination criterion (``MTDC.m``).

    :return: ``(type, index)`` where ``type`` is 1 (chirp) or 2 (dispersive),
             and ``index`` is ``(P, 2)`` ridge indices.
             - chirp: columns ``(time_idx, freq_idx)``
             - dispersive: columns ``(freq_idx, time_idx)``
    """
    indext1, indexf1 = find_ridges_peak(spec)
    if indext1.size < 2:
        # too short — default to chirp with whatever points exist
        idx = np.column_stack([indext1, indexf1]) if indext1.size else np.zeros((0, 2))
        return 1, idx.astype(int)

    ad1 = _avg_repeat_spacing(indexf1)
    if ad1 < ad_thresh:
        indext2, indexf2 = find_ridges_peak(spec.T)
        if indext2.size < 2:
            return 2, np.column_stack([indext2, indexf2]).astype(int)
        ad2 = _avg_repeat_spacing(indexf2)
        if ad2 < ad_thresh:
            denom = float(indext1[0] - indext1[-1])
            if abs(denom) < 1e-12:
                cr = np.inf
            else:
                cr = abs((indexf1[0] - indexf1[-1]) / denom)
            if cr <= 1.0:
                return 1, np.column_stack([indext1, indexf1]).astype(int)
            return 2, np.column_stack([indext2, indexf2]).astype(int)
        return 2, np.column_stack([indext2, indexf2]).astype(int)
    return 1, np.column_stack([indext1, indexf1]).astype(int)


def acmd_single(
    signal: np.ndarray,
    t: np.ndarray,
    init_if: np.ndarray,
    alpha: float = 1e-4,
    beta: float = 1e-7,
    tol: float = 1e-8,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single-mode ACMD on a (possibly truncated) time support.

    Single-mode ACMD / GDMD used by VOA

    :return: ``(mode, if_est, ia_est)``
    """
    s = np.asarray(signal, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    e_if = np.asarray(init_if, dtype=float).ravel().copy()
    n = s.size
    if t.size != n or e_if.size != n:
        raise ValueError("signal, t and init_if must share the same length")
    if n < 4:
        return s.copy(), e_if, np.abs(s)

    dt = float(t[1] - t[0]) if n > 1 else 1.0
    oper = second_order_difference(n)
    opedoub = (oper.T @ oper).tocsc()
    spzeros = diags([np.zeros(n)], [0], shape=(n - 2, n))
    phim = vstack([hstack([oper, spzeros]), hstack([spzeros, oper])]).tocsc()
    phidoubm = (phim.T @ phim).tocsc()

    s_prev = None
    s_dif = tol + 1.0
    it = 0
    si = np.zeros(n)
    ia = np.zeros(n)

    while s_dif > tol and it < max_iter:
        phase = cumulative_trapezoid(e_if, t, initial=0.0)
        cosm = np.cos(2.0 * np.pi * phase)
        sinm = np.sin(2.0 * np.pi * phase)
        cm = diags([cosm], [0], shape=(n, n))
        sm = diags([sinm], [0], shape=(n, n))
        kerm = hstack([cm, sm]).tocsc()
        kerdoubm = (kerm.T @ kerm).tocsc()
        ym = spsolve((1.0 / alpha) * phidoubm + kerdoubm, kerm.T @ s)
        si = np.asarray(kerm @ ym).ravel()
        ycm = np.asarray(ym[:n]).ravel()
        ysm = np.asarray(ym[n:]).ravel()
        ycmbar = differ(ycm, dt)
        ysmbar = differ(ysm, dt)
        denom = ycm**2 + ysm**2 + 1e-30
        delta_if = (ycm * ysmbar - ysm * ycmbar) / denom / (2.0 * np.pi)
        delta_if = spsolve((1.0 / beta) * opedoub + eye(n, format="csc"), delta_if)
        e_if = e_if - delta_if
        ia = np.sqrt(np.maximum(ycm**2 + ysm**2, 0.0))

        if s_prev is not None:
            den = norm(s_prev)
            s_dif = (norm(si - s_prev) / den) ** 2 if den > 1e-30 else 0.0
        s_prev = si
        it += 1

    return si, e_if, ia


def gdmd_on_frequency(
    spectrum: np.ndarray,
    freq: np.ndarray,
    init_gd: np.ndarray,
    alpha: float = 1e-4,
    beta: float = 1e-7,
    tol: float = 1e-8,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GDMD on an arbitrary frequency grid (toolbox ``GDMD.m`` in VGNMD).

    Uses the true frequency axis for phase integration; ``df = freq[1]-freq[0]``.
    """
    s = np.asarray(spectrum, dtype=complex).ravel()
    freq = np.asarray(freq, dtype=float).ravel()
    e_gd = np.atleast_2d(np.asarray(init_gd, dtype=float)).astype(float).copy()
    if e_gd.shape[0] != 1:
        # VOA uses a single mode; allow (1, N) or (N,)
        if e_gd.shape[1] == 1 and e_gd.shape[0] == s.size:
            e_gd = e_gd.T
    if e_gd.shape != (1, s.size):
        e_gd = np.atleast_2d(e_gd.ravel())
    if s.size != freq.size:
        raise ValueError("spectrum and freq length mismatch")

    # Map to duration-style gdmd_core when the grid is uniform from 0
    df = float(freq[1] - freq[0]) if freq.size > 1 else 1.0
    if np.allclose(freq, np.arange(freq.size) * df):
        duration = 1.0 / df
        gd, modes, _, _ = gdmd_core(
            s, duration, e_gd, alpha=alpha, beta=beta, tol=tol, max_iter=max_iter
        )
        return gd[0], modes[0]

    # General uniform / mildly non-uniform grid
    n = s.size
    oper = second_order_difference(n)
    opedoub = (oper.T @ oper).tocsc()
    s_prev = None
    s_dif = tol + 1.0
    it = 0
    mode = np.zeros(n, dtype=complex)
    gd = e_gd[0].copy()

    while s_dif > tol and it < max_iter:
        phase = cumulative_trapezoid(gd, freq, initial=0.0)
        kern = np.exp(-1j * 2.0 * np.pi * phase)
        kerm = diags([kern], [0], shape=(n, n)).tocsc()
        kerdoubm = (kerm.T.conj() @ kerm).tocsc()
        phidoubm = opedoub  # single mode
        ym = spsolve((1.0 / alpha) * phidoubm + kerdoubm, kerm.T.conj() @ s)
        mode = np.asarray(kerm @ ym).ravel()
        delta_phase = np.unwrap(np.angle(ym))
        delta_gd = differ(delta_phase, df) / (2.0 * np.pi)
        delta_gd = spsolve((1.0 / beta) * opedoub + eye(n, format="csc"), delta_gd)
        gd = gd - delta_gd
        if s_prev is not None:
            den = norm(s_prev)
            s_dif = (norm(mode - s_prev) / den) ** 2 if den > 1e-30 else 0.0
        s_prev = mode
        it += 1

    return gd, mode


def voa(
    signal: np.ndarray,
    mode_type: int,
    ridge_index: np.ndarray,
    t: np.ndarray,
    f: np.ndarray,
    alpha: float = 1e-4,
    beta: float = 1e-7,
    tol: float = 1e-8,
    max_iter: int = 300,
    smooth_beta: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Extract one mode with ACMD (type 1) or GDMD (type 2) — ``VOA.m``.

    VOA — variational optimisation algorithm

    :return: ``(mode_t, mode_f, if_or_gd, type)``
    """
    x = np.asarray(signal, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    f = np.asarray(f, dtype=float).ravel()
    n = x.size
    nf = f.size
    ridge_index = np.asarray(ridge_index, dtype=int)
    if ridge_index.ndim != 2 or ridge_index.shape[1] != 2:
        raise ValueError("ridge_index must have shape (P, 2)")

    mode_t = np.zeros(n, dtype=float)
    mode_f = np.zeros(nf, dtype=float)
    eif_gd = np.zeros(n if mode_type == 1 else nf, dtype=float)

    if mode_type == 1:
        indext = np.sort(np.unique(ridge_index[:, 0]))
        # map each time to a frequency (first occurrence)
        freq_of_t = {}
        for ti, fi in ridge_index:
            freq_of_t.setdefault(int(ti), int(fi))
        indext = np.array([ti for ti in indext if ti in freq_of_t], dtype=int)
        if indext.size < 4:
            return mode_t, mode_f, eif_gd, mode_type
        indexf = np.array([freq_of_t[int(ti)] for ti in indext], dtype=int)
        indexf = np.clip(indexf, 0, nf - 1)
        sig1 = x[indext]
        iif = curve_smooth(f[indexf], smooth_beta).ravel()
        sest, if_est, _ = acmd_single(
            sig1, t[indext], iif, alpha=alpha, beta=beta, tol=tol, max_iter=max_iter
        )
        mode_t[indext] = sest
        eif_gd = np.zeros(n, dtype=float)
        eif_gd[indext] = if_est
        spec_full = 2.0 * np.abs(np.fft.fft(mode_t)) / max(n, 1)
        mode_f = spec_full[:nf]
        return mode_t, mode_f, eif_gd, mode_type

    if mode_type == 2:
        # ridge columns: (freq_idx, time_idx)
        indexf = np.sort(np.unique(ridge_index[:, 0]))
        time_of_f = {}
        for fi, ti in ridge_index:
            time_of_f.setdefault(int(fi), int(ti))
        indexf = np.array([fi for fi in indexf if fi in time_of_f], dtype=int)
        if indexf.size < 4:
            return mode_t, mode_f, eif_gd, mode_type
        indext = np.array([time_of_f[int(fi)] for fi in indexf], dtype=int)
        indexf = np.clip(indexf, 0, nf - 1)
        indext = np.clip(indext, 0, n - 1)

        fft_full = np.fft.fft(x)
        # unilateral length matching STFT frequency axis ≈ n//2
        dsn = fft_full[:nf]
        dsn1 = dsn[indexf]
        ini_gd = curve_smooth(t[indext], smooth_beta).ravel()
        gd_est, des_est = gdmd_on_frequency(
            dsn1,
            f[indexf],
            ini_gd,
            alpha=alpha,
            beta=beta,
            tol=tol,
            max_iter=max_iter,
        )
        mode_f_c = np.zeros(nf, dtype=complex)
        mode_f_c[indexf] = des_est
        eif_gd = np.zeros(nf, dtype=float)
        eif_gd[indexf] = gd_est
        # rebuild full FFT length for IFFT
        n_uni = n // 2 + 1
        full_uni = np.zeros(n_uni, dtype=complex)
        n_copy = min(nf, n_uni)
        full_uni[:n_copy] = mode_f_c[:n_copy]
        mode_t = spectrum_to_time(full_uni, n)
        mode_f = np.real(mode_f_c)
        return mode_t, mode_f, eif_gd, mode_type

    raise ValueError(f"unknown mode type {mode_type}; expected 1 or 2")


def make_vgnmd_demo_signal(
    samp_freq: float = 1000.0,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Generalized nonlinear signal from the VGNMD MATLAB ``Test.m`` demo.

    Demo signal (toolbox ``Test.m``)

    Three chirp modes + four dispersive modes on ``t ∈ [0, 3)`` s.
    """
    fs = float(samp_freq)
    t1 = np.arange(0.0, 1.5, 1.0 / fs)
    t2 = np.arange(1.5, 3.0, 1.0 / fs)
    t = np.concatenate([t1, t2])
    nt = t.size
    nf = nt // 2 + 1
    duration = nt / fs
    f_axis = np.arange(nf) / duration

    sig1 = np.cos(2 * np.pi * (160 * t + 20 * t**2 + 3 * np.cos(3 * np.pi * t)))
    if1 = 160 + 40 * t - 9 * np.pi * np.sin(3 * np.pi * t)

    sig2 = np.concatenate(
        [np.cos(2 * np.pi * (70 * t1 + 20 * t1**2)), np.zeros(t2.size)]
    )
    if2 = 70 + 40 * t1

    sig3 = np.concatenate(
        [
            np.zeros(t1.size),
            np.cos(2 * np.pi * (20 * t2 + 20 * t2**2 + 3 * np.cos(3 * np.pi * t2))),
        ]
    )
    if3 = 20 + 40 * t2 - 9 * np.pi * np.sin(3 * np.pi * t2)

    def _dispersive(f_cut_frac, amp_phase_fn, gd_fn):
        split = int(f_cut_frac * nf)
        split = min(max(split, 1), nf - 1)
        f_hi = f_axis[split:]
        ds = np.zeros(nf, dtype=complex)
        ds[split:] = 30.0 * np.exp(-1j * 2 * np.pi * amp_phase_fn(f_hi))
        mode_t = spectrum_to_time(ds, nt)
        gd = np.full(nf, np.nan)
        gd[split:] = gd_fn(f_hi)
        return mode_t, ds, gd, f_hi

    sig4, _, gd4, f42 = _dispersive(
        0.5,
        lambda fq: 0.4 * fq + 2 * np.cos(2 * np.pi * fq / 100.0),
        lambda fq: 0.4 - 4 * np.pi / 100.0 * np.sin(2 * np.pi * fq / 100.0),
    )
    sig5, _, gd5, f52 = _dispersive(
        0.6,
        lambda fq: 0.8 * fq + 0.0005 * fq**2,
        lambda fq: 0.8 + 0.001 * fq,
    )
    sig6, _, gd6, f62 = _dispersive(
        0.7,
        lambda fq: 1.8 * fq + 2 * np.cos(2 * np.pi * fq / 100.0),
        lambda fq: 1.8 - 4 * np.pi / 100.0 * np.sin(2 * np.pi * fq / 100.0),
    )
    sig7, _, gd7, f72 = _dispersive(
        0.8,
        lambda fq: 2.2 * fq + 0.0005 * fq**2,
        lambda fq: 2.2 + 0.001 * fq,
    )

    clean = sig1 + sig2 + sig3 + sig4 + sig5 + sig6 + sig7
    if noise_std > 0:
        rng = np.random.default_rng() if rng is None else rng
        noise = rng.standard_normal(nt)
        noise = (noise - noise.mean()) / (noise.std() + 1e-30)
        noise = noise_std * noise
        observed = clean + noise
    else:
        noise = np.zeros(nt)
        observed = clean.copy()

    return {
        "t": t,
        "t1": t1,
        "t2": t2,
        "fs": np.array([fs]),
        "signal": observed,
        "clean": clean,
        "noise": noise,
        "modes_true": np.vstack([sig1, sig2, sig3, sig4, sig5, sig6, sig7]),
        "if1": if1,
        "if2": if2,
        "if3": if3,
        "f42": f42,
        "gd4": gd4,
        "f52": f52,
        "gd5": gd5,
        "f62": f62,
        "gd6": gd6,
        "f72": f72,
        "gd7": gd7,
    }


class VGNMD(object):
    """
    Variational Generalized Nonlinear Mode Decomposition (VGNMD).

    Automatically detects chirp / dispersive modes via ATFFC + MTDC, then
    reconstructs each mode with ACMD or GDMD (VOA).
    """

    def __init__(
        self,
        alpha: float = 1e-4,
        beta: float = 1e-7,
        tol: float = 1e-8,
        max_iter: int = 300,
        n_windows: int = 5,
        min_frac: float = 0.001,
        smooth_beta: float = 1e-6,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.n_windows = int(n_windows)
        self.min_frac = float(min_frac)
        self.smooth_beta = float(smooth_beta)

        self.modes_time_: Optional[np.ndarray] = None
        self.modes_freq_: Optional[np.ndarray] = None
        self.types_: Optional[np.ndarray] = None
        self.init_ridges_: Optional[List[np.ndarray]] = None
        self.features_: Optional[List[np.ndarray]] = None  # IF or GD
        self.f_: Optional[np.ndarray] = None
        self.t_: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return "Variational Generalized Nonlinear Mode Decomposition (VGNMD)"

    def __call__(
        self, signal: np.ndarray, fs: float, return_all: bool = False
    ) -> Union[np.ndarray, Tuple]:
        return self.fit_transform(signal, fs, return_all=return_all)

    def fit_transform(
        self,
        signal: np.ndarray,
        fs: float,
        return_all: bool = False,
    ) -> Union[np.ndarray, Tuple]:
        """
        Decompose a generalized nonlinear signal.

        :param signal: 1-D real signal
        :param fs: sampling frequency (Hz)
        :param return_all: if True, also return frequency-domain modes, types,
                           initial ridges, refined IF/GD features, and axes
        :return: time-domain modes ``(K, N)``, or a rich tuple when ``return_all``
        """
        x = np.asarray(signal, dtype=float).ravel()
        if x.size < 16:
            raise ValueError("signal length must be >= 16")
        fs = float(fs)
        if fs <= 0:
            raise ValueError("fs must be positive")

        n = x.size
        t = np.arange(n, dtype=float) / fs
        clusters, f = atffc(x, fs, n_windows=self.n_windows, min_frac=self.min_frac)
        k = len(clusters)
        modes_t = np.zeros((k, n), dtype=float)
        modes_f = np.zeros((k, f.size), dtype=float)
        types = np.zeros(k, dtype=int)
        init_ridges: List[np.ndarray] = []
        features: List[np.ndarray] = []

        for i, clu in enumerate(clusters):
            mtype, ridge = mtdc(clu)
            init_ridges.append(ridge)
            mt, mf, feat, mtype = voa(
                x,
                mtype,
                ridge,
                t,
                f,
                alpha=self.alpha,
                beta=self.beta,
                tol=self.tol,
                max_iter=self.max_iter,
                smooth_beta=self.smooth_beta,
            )
            modes_t[i] = np.real(mt)
            modes_f[i] = np.real(mf) if np.isrealobj(mf) else np.abs(mf)
            types[i] = int(mtype)
            features.append(np.asarray(feat, dtype=float).ravel())

        self.modes_time_ = modes_t
        self.modes_freq_ = modes_f
        self.types_ = types
        self.init_ridges_ = init_ridges
        self.features_ = features
        self.f_ = f
        self.t_ = t

        if return_all:
            return modes_t, modes_f, types, init_ridges, features, f, t
        return modes_t


def vgnmd(
    signal: np.ndarray,
    fs: float,
    alpha: float = 1e-4,
    beta: float = 1e-7,
    tol: float = 1e-8,
    max_iter: int = 300,
    return_all: bool = False,
) -> Union[np.ndarray, Tuple]:
    """Functional wrapper around :class:`VGNMD`."""
    return VGNMD(alpha=alpha, beta=beta, tol=tol, max_iter=max_iter).fit_transform(
        signal, fs, return_all=return_all
    )
