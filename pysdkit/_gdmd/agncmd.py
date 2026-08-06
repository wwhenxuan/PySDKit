# -*- coding: utf-8 -*-
"""
Created on 2025/08/06
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Adaptive Generalized Dispersive Mode Decomposition (AGDMD / AGNCMD).

AGDMD extends GDMD with a fully data-driven pipeline that removes the need
for hand-crafted initial group delays and bandwidth parameters:

1. **AGDI** — Adaptive Group-Delay Initialization
   (zero-pad → data-driven GD via IF-DN + TVLP → FIR decimate)
2. **BE** — Bandwidth Estimation via dispersion compensation + envelope peaks
3. **Adaptive GDMD** — recursive single-mode extraction with online ``alpha``
   refinement (non-increasing)

Paper (MATLAB toolbox name ``AGDMD``; package alias ``AGNCMD``)::

    Wang H., Chen S., Zhai W.
    Adaptive generalized dispersive mode decomposition: A data-driven
    approach for nonlinear dispersive component extraction in mechanical
    systems. Journal of Sound and Vibration, 2025.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm
from scipy.integrate import cumulative_trapezoid
from scipy.signal import decimate, find_peaks, firwin, hilbert, resample
from scipy.sparse import block_diag, diags, eye, lil_matrix
from scipy.sparse.linalg import spsolve

from pysdkit._gdmd.gdmd import (
    curve_smooth,
    differ,
    second_order_difference,
    tf_spec_from_gd,
)


# ---------------------------------------------------------------------------
# Small helpers matching the MATLAB toolbox
# ---------------------------------------------------------------------------


def spectrum_to_time_agdmd(spectrum: np.ndarray, n_time: int) -> np.ndarray:
    """
    Real IFFT reconstruction used by MATLAB ``AGDMD.m``::

        full = [S, conj(fliplr(S(2:ceil(Nt/2))))];  real(ifft(full))
    """
    s = np.asarray(spectrum, dtype=complex).ravel()
    nf = n_time // 2 + 1
    if s.size != nf:
        raise ValueError(
            f"unilateral spectrum length {s.size} incompatible with n_time={n_time} "
            f"(expected {nf})"
        )
    mid = int(np.ceil(n_time / 2.0))
    full = np.concatenate([s, np.conj(s[1:mid][::-1])])
    if full.size != n_time:
        # Rare rounding edge: fall back to conjugate-symmetric fill
        full = np.empty(n_time, dtype=complex)
        full[:nf] = s
        full[nf:] = np.conj(s[-2 : n_time - nf - 1 : -1])
    return np.real(np.fft.ifft(full))


def differ_complex(y: np.ndarray, delta: float) -> np.ndarray:
    """Discrete derivative supporting complex series (MATLAB ``Differ.m``)."""
    y = np.asarray(y).ravel()
    l = y.size
    if l < 2:
        return np.zeros_like(y)
    ybar = np.empty(l, dtype=np.result_type(y.dtype, float))
    ybar[0] = (y[1] - y[0]) / delta
    ybar[-1] = (y[-1] - y[-2]) / delta
    if l > 2:
        ybar[1:-1] = (y[2:] - y[:-2]) / (2.0 * delta)
    return ybar


def findev(sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local maxima extraction (MATLAB ``findev.m``).

    Extrema are detected on ``real(sig)`` (MATLAB's ``>`` on complex uses the
    real part); returned peak *values* keep the original dtype (possibly complex).

    :return: ``(max_vals, max_indices, upper_envelope)`` — indices are 0-based.
    """
    x = np.asarray(sig).ravel()
    n = x.size
    if n < 3:
        idx = np.array([0], dtype=int) if n else np.array([], dtype=int)
        vals = x[idx] if idx.size else np.array([])
        up = np.full(n, vals[0] if vals.size else 0.0, dtype=x.dtype)
        return vals, idx, up

    xr = np.real(x)
    d1 = np.diff(xr)
    d2 = np.diff(d1 > 0)
    max_vec = np.where(d2 < 0)[0] + 1

    if xr[0] > xr[1]:
        max_vec = np.r_[0, max_vec]
    if xr[-1] > xr[-2]:
        max_vec = np.r_[max_vec, n - 1]

    max_vec = np.unique(max_vec.astype(int))
    if max_vec.size == 0:
        up = np.full(n, x[int(np.argmax(xr))], dtype=x.dtype)
        return np.array([]), max_vec, up

    max_vals = x[max_vec]
    up = np.empty(n, dtype=x.dtype)
    if max_vec.size == 1:
        up[:] = max_vals[0]
    else:
        for i in range(max_vec.size - 1):
            up[max_vec[i] : max_vec[i + 1] + 1] = max_vals[i]
        up[: max_vec[0] + 1] = max_vals[0]
        up[max_vec[-1] :] = max_vals[-1]
    return max_vals, max_vec, up


def arccos_phase(g: np.ndarray) -> np.ndarray:
    """
    Piecewise inverse-cosine phase accumulation (MATLAB ``arccos.m``).

    Supports complex ``g`` as in the AGDI spectral IF-DN path.
    """
    g = np.asarray(g).ravel()
    s = g.size
    if s == 0:
        return g.copy()

    f1 = g.copy()
    # Real-valued path: clip for acos domain; complex: leave for np.arccos
    if not np.iscomplexobj(f1):
        f1 = np.clip(f1, -1.0, 1.0)
    theta = (0.5 / np.pi) * np.arccos(f1)
    temp = 0.0 + 0.0j if np.iscomplexobj(g) else 0.0

    for i in range(2, s):  # MATLAB i = 3:s
        # Turning-point detection (MATLAB sign / > on complex → real part)
        d1 = np.real(g[i] - g[i - 1])
        d0 = np.real(g[i - 1] - g[i - 2])
        if np.sign(d1) != np.sign(d0):
            temp = theta[i - 1]
        if np.real(g[i] - g[i - 1]) > 0:
            f1[i] = -g[i]
        else:
            f1[i] = g[i]
        if not np.iscomplexobj(f1):
            f1[i] = float(np.clip(np.real(f1[i]), -1.0, 1.0))
        theta[i] = (0.5 / np.pi) * np.arccos(f1[i]) + temp

    return theta


def _curve_smooth_maybe_complex(curves: np.ndarray, beta: float) -> np.ndarray:
    """``curve_smooth`` with complex support (smooth Re/Im separately)."""
    c = np.asarray(curves)
    if np.iscomplexobj(c):
        return curve_smooth(np.real(c), beta) + 1j * curve_smooth(np.imag(c), beta)
    return curve_smooth(c, beta)


def if_dn(sig: np.ndarray, samp_freq: float, beta: float) -> np.ndarray:
    """
    Instantaneous-frequency / group-delay estimate via derivative
    normalisation (MATLAB ``IF_DN.m``).

    When called from AGDI, ``sig`` is a unilateral *frequency-domain* spectrum
    (complex) treated as a 1-D series and ``samp_freq`` is the *duration* ``T``
    of the zero-padded time signal (MATLAB naming quirk in ``DDGDI``).
    """
    x = np.asarray(sig).ravel()
    delta = 1.0 / float(samp_freq)
    dx = differ_complex(x, delta)

    _, max_vec, _ = findev(dx)
    _, min_vec, _ = findev(-dx)
    mv = np.unique(np.sort(np.concatenate([max_vec, min_vec]).astype(int)))

    g = np.zeros(dx.size, dtype=np.result_type(dx.dtype, float))
    if mv.size >= 2:
        for i in range(mv.size - 1):
            a, b = int(mv[i]), int(mv[i + 1])
            lav = np.abs(dx[a] - dx[b]) / 2.0
            lm = (dx[a] + dx[b]) / 2.0
            if lav < 1e-30:
                g[a : b + 1] = 0.0
            else:
                g[a : b + 1] = (dx[a : b + 1] - lm) / lav

    theta = arccos_phase(g)
    inst = differ_complex(theta, delta)
    inst = _curve_smooth_maybe_complex(inst, beta).ravel()
    return np.real(inst)


def low_filter(sig: np.ndarray, cut_freq: float, samp_freq: float) -> np.ndarray:
    """
    Linear-phase FIR low-pass (MATLAB ``low_filter.m`` / ``fir1``).

    Filter length ≈ ``0.8 * N``; phase delay is corrected by cropping the
    convolution result.
    """
    x = np.asarray(sig, dtype=complex).ravel()
    n0 = x.size
    if n0 < 4:
        return x.copy()

    n = int(np.floor(n0 * 0.8))
    # Normalised cutoff (Nyquist = 1), matching MATLAB fir1
    w1 = 2.0 * cut_freq / float(samp_freq)
    w1 = float(np.clip(w1, 1e-6, 0.99))

    l_ord = n if (n % 2 == 0) else (n + 1)
    # fir1(L, …) → order L → L+1 taps
    b = firwin(l_ord + 1, w1, pass_zero="lowpass")
    y = np.convolve(b, x)
    start = l_ord // 2
    return y[start : start + n0]


def tvlp(
    sig: np.ndarray, samp_freq: float, eif: np.ndarray, c_pass: float
) -> np.ndarray:
    """
    Time-varying low-pass filter (MATLAB ``TVLP.m``).

    Demodulate with ``EIF - Cpass``, FIR low-pass at ``1.1 * Cpass``, remodulate.
    """
    x = np.asarray(sig).ravel()
    eif = np.asarray(eif, dtype=float).ravel()
    if eif.size != x.size:
        raise ValueError("EIF length must match signal length")

    t = np.arange(x.size, dtype=float) / float(samp_freq)
    phase = 2.0 * np.pi * cumulative_trapezoid(eif - c_pass, t, initial=0.0)

    # MATLAB hilbert ignores imag of complex inputs
    xa = hilbert(np.real(x))
    demod = xa * np.exp(-1j * phase)
    filt = low_filter(demod, 1.1 * c_pass, samp_freq)
    remode = filt * np.exp(1j * phase)
    return np.real(remode)


def ddgdi(
    sig: np.ndarray,
    samp_freq: float,
    beta: float,
    max_iter: int = 30,
    tol: float = 0.01,
) -> np.ndarray:
    """
    Data-driven group-delay initialisation (MATLAB ``DDGDI`` in ``AGDI.m``).

    IF-DN is applied to the (extended) unilateral spectrum; TVLP refinements
    are accepted only while the GD estimate remains stable.  If TVLP causes
    the estimate to drift (common with close residual modes), the first
    IF-DN result — which emphasises the largest-GD component — is kept.
    """
    x = np.asarray(sig).ravel()
    gd0 = if_dn(x, samp_freq, beta)
    gd = gd0.copy()
    igd_norm = float(norm(gd0))

    for _ in range(int(max_iter) - 1):
        c_pass = 1.2 * float(np.max(gd))
        nyq = 0.49 * float(samp_freq)
        if c_pass <= 0:
            c_pass = 0.1 * float(samp_freq)
        c_pass = min(max(c_pass, 1e-6), nyq)
        x = tvlp(x, samp_freq, gd, c_pass)
        gd_new = if_dn(x, samp_freq, beta)
        gd_n = float(norm(gd_new))
        delta = abs(gd_n - float(norm(gd))) / (float(norm(gd)) + 1e-30)
        # Drift relative to the original IF-DN estimate
        drift = float(norm(gd_new - gd0)) / (igd_norm + 1e-30)
        if delta < tol:
            return gd_new
        if drift > 0.35:
            # TVLP moved away from the largest-GD solution — stop
            return gd0
        gd = gd_new

    return gd0


def agdi(sig: np.ndarray, samp_freq: float, beta: float) -> np.ndarray:
    """
    Adaptive group-delay initialisation (MATLAB ``AGDI.m``).

    Zero-pads the time signal to ``3N``, estimates GD on the unilateral FFT
    via :func:`ddgdi`, then FIR-decimates by 3 back to the original frequency
    grid length.
    """
    x = np.asarray(sig, dtype=float).ravel()
    n = x.size
    # [Sig, zeros(N), zeros(N)]
    x_ext = np.concatenate([x, np.zeros(n), np.zeros(n)])
    nt = x_ext.size
    t2 = nt / float(samp_freq)
    dsn = np.fft.fft(x_ext)[: nt // 2 + 1]

    ini_gd = ddgdi(dsn, t2, beta)
    # MATLAB: decimate(iniGD, 3, 'fir')
    igd = decimate(np.real(ini_gd), 3, ftype="fir", zero_phase=True)

    nf = n // 2 + 1
    if igd.size != nf:
        # Align to the original unilateral FFT length
        igd = np.interp(
            np.linspace(0.0, 1.0, nf),
            np.linspace(0.0, 1.0, igd.size),
            igd,
        )
    return igd.astype(float)


def dispersion_compensation(
    sig: np.ndarray, samp_freq: float, egd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dispersion compensation (MATLAB ``DC`` in ``BE.m``).

    :return: ``(sig_d, dsn_d)`` — compensated *time* signal (length ``N``) and
             real part of the compensated extended unilateral spectrum.
    """
    x = np.asarray(sig, dtype=float).ravel()
    egd = np.asarray(egd, dtype=float).ravel()
    nt1 = x.size
    t1 = nt1 / float(samp_freq)

    x_ext = np.concatenate([x, np.zeros(nt1), np.zeros(nt1)])
    nt = x_ext.size
    t_ext = nt / float(samp_freq)

    dsn = np.fft.fft(x_ext)[: nt // 2 + 1]
    n = dsn.size

    # MATLAB resample(EGD, T, T1) → rate change T/T1 (= 3)
    n_up = int(round(egd.size * t_ext / t1))
    e_gd1 = resample(egd, n_up)
    if e_gd1.size < n:
        e_gd1 = np.pad(e_gd1, (0, n - e_gd1.size), mode="edge")
    e_gd1 = e_gd1[:n]

    ff = np.arange(n, dtype=float) / t_ext
    phase = (
        2.0 * np.pi * cumulative_trapezoid(e_gd1 - float(np.max(egd)), ff, initial=0.0)
    )

    sh = hilbert(np.real(dsn))
    dsnd = sh * np.exp(-1j * phase)

    mid = int(np.ceil(nt / 2.0))
    full = np.concatenate([dsnd, np.conj(dsnd[1:mid][::-1])])
    if full.size != nt:
        full = np.empty(nt, dtype=complex)
        nf = nt // 2 + 1
        full[:nf] = dsnd[:nf]
        full[nf:] = np.conj(dsnd[nf - 2 : nt - nf - 1 : -1])
    iffte = np.fft.ifft(full)

    dsnd_r = np.real(dsnd)
    sig_d = np.real(iffte[-nt1:])
    sig_d = sig_d[::-1]  # MATLAB flip
    return sig_d, dsnd_r


def bandwidth_estimation(
    sig: np.ndarray, samp_freq: float, egd: np.ndarray
) -> Tuple[float, float]:
    """
    Bandwidth / ``alpha`` estimation (MATLAB ``BE.m``).

    Uses the 4th-power envelope of the dispersion-compensated signal and the
    last valley before ``max(GD)`` to estimate a normalised bandwidth, then
    maps it to the GDMD penalty through the paper's empirical power law::

        alpha = ((BW - 0.008791) / 0.2906) ** 4.363
    """
    x = np.asarray(sig, dtype=float).ravel()
    egd = np.asarray(egd, dtype=float).ravel()
    nt = x.size
    duration = nt / float(samp_freq)
    t = np.arange(nt, dtype=float) / float(samp_freq)

    sig_d, _ = dispersion_compensation(x, samp_freq, egd)

    shde = np.abs(hilbert(sig_d**4))
    shde = curve_smooth(shde, 1e-2).ravel()

    it = float(np.max(egd))
    itv = int(np.fix(it / duration * nt))
    itv = int(np.clip(itv, 1, nt - 2))

    # Valleys = peaks of -envelope
    loc_f, _ = find_peaks(-shde[:itv])
    loc_b, _ = find_peaks(-shde[itv:])  # noqa: F841 — mirrored MATLAB

    if loc_f.size == 0:
        # Fallback: use a mild default bandwidth
        bw = 0.05
    else:
        it_f = float(t[int(loc_f[-1])])
        bw = 2.0 * (it - it_f) / duration

    bw = max(float(bw), 0.008791 + 1e-6)
    alpha = ((bw - 0.008791) / 0.2906) ** 4.363
    # Keep alpha in a numerically safe / practically useful range for GDMD
    alpha = float(np.clip(alpha, 1e-4, 10.0))
    return float(bw), alpha


def agdmd_core(
    spectrum: np.ndarray,
    duration: float,
    init_gd: np.ndarray,
    alpha0: float,
    beta: float,
    signal: np.ndarray,
    samp_freq: float,
    tol: float = 1e-8,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive single-/multi-mode GDMD with online ``alpha`` updates
    (MATLAB ``GDMD.m`` in the AGDMD toolbox — *not* the plain GDMD solver).

    ``alpha`` is re-estimated by :func:`bandwidth_estimation` each iteration and
    is only allowed to **decrease** (narrower bandwidth as GD improves).

    :return: ``(gd_final, mode_freq, alpha_history)``
        - ``gd_final`` shape ``(N,)`` (first mode)
        - ``mode_freq`` shape ``(N,)`` complex
        - ``alpha_history`` 1-D array of length ``n_iters`` (values *before*
          each update, matching MATLAB ``alpha`` cell contents used for plots)
    """
    s = np.asarray(spectrum, dtype=complex).ravel()
    e_gd = np.atleast_2d(np.asarray(init_gd, dtype=float)).copy()
    num, n = e_gd.shape
    if s.size != n:
        raise ValueError(f"spectrum length {s.size} must match init_gd columns {n}")
    if duration <= 0:
        raise ValueError("duration (T) must be positive")

    x_time = np.asarray(signal, dtype=float).ravel()
    freq = np.arange(n, dtype=float) / duration
    df = 1.0 / duration

    oper = second_order_difference(n)
    opedoub = (oper.T @ oper).tocsc()
    phim = block_diag([oper] * num).tocsc()
    phidoubm = (phim.T @ phim).tocsc()

    alpha_hist: List[float] = [float(alpha0)]
    s_dif = tol + 1.0
    it = 0
    s_prev = None
    modes_iter = np.zeros((num, n), dtype=complex)

    while s_dif > tol and it < max_iter:
        alpha = alpha_hist[it]

        kerm = lil_matrix((n, n * num), dtype=complex)
        for kk in range(num):
            phase = cumulative_trapezoid(e_gd[kk], freq, initial=0.0)
            kern = np.exp(-1j * 2.0 * np.pi * phase)
            kerm[:, kk * n : (kk + 1) * n] = diags([kern], [0], shape=(n, n))
        kerm = kerm.tocsc()
        kerdoubm = (kerm.T.conj() @ kerm).tocsc()

        a_mat = (1.0 / alpha) * phidoubm + kerdoubm
        ym_all = spsolve(a_mat, kerm.T.conj() @ s)

        modes_iter = np.empty((num, n), dtype=complex)
        for kk in range(num):
            ym = ym_all[kk * n : (kk + 1) * n]
            delta_phase = np.unwrap(np.angle(ym))
            delta_gd = differ(delta_phase, df) / (2.0 * np.pi)
            delta_gd = spsolve((1.0 / beta) * opedoub + eye(n, format="csc"), delta_gd)
            e_gd[kk] = e_gd[kk] - delta_gd
            modes_iter[kk] = kerm[:, kk * n : (kk + 1) * n] @ ym

        # Adaptive alpha (non-increasing)
        _, alpha1 = bandwidth_estimation(x_time, samp_freq, e_gd[0])
        alpha_next = alpha1 if alpha1 < alpha else alpha
        alpha_hist.append(float(alpha_next))

        if s_prev is not None:
            s_dif = 0.0
            for kk in range(num):
                den = norm(s_prev[kk])
                if den < 1e-30:
                    continue
                s_dif += (norm(modes_iter[kk] - s_prev[kk]) / den) ** 2

        s_prev = modes_iter
        it += 1

    if it == 0:
        raise RuntimeError("Adaptive GDMD failed to perform any iteration")

    # MATLAB returns alpha used during iterations (length ≈ it); drop the
    # trailing unused next-value for plotting consistency when desired.
    alpha_arr = np.asarray(alpha_hist[:it], dtype=float)
    return e_gd[0].copy(), modes_iter[0].copy(), alpha_arr


def make_agncmd_demo_signal(
    samp_freq: float = 100.0,
    duration: float = 10.0,
) -> Dict[str, np.ndarray]:
    """
    Three close dispersive modes from MATLAB ``Example1.m``.

    Group delays::

        gd_k(f) = (1/125) f^2 - (2/5) f + (6.5 + 0.5*(k-1)),  k = 1,2,3

    :return: dict with ``t``, ``signal``, ``f``, ``fs``, ``true_gds``,
             ``true_modes_time``, ``true_modes_freq``.
    """
    fs = float(samp_freq)
    t_dur = float(duration)
    nt = int(round(fs * t_dur))
    nf = nt // 2 + 1
    t = np.arange(nt, dtype=float) / fs
    f = np.arange(nf, dtype=float) / t_dur

    offsets = (6.5, 7.0, 7.5)
    true_gds = []
    true_freq = []
    true_time = []
    for off in offsets:
        gd = (1.0 / 125.0) * f**2 - (2.0 / 5.0) * f + off
        # Phase = ∫ gd df  →  (1/375)f^3 - (1/5)f^2 + off*f + 0.5
        ds = np.exp(-0.02 * f) * np.exp(
            -1j
            * 2
            * np.pi
            * ((1.0 / 375.0) * f**3 - (1.0 / 5.0) * f**2 + off * f + 0.5)
        )
        true_gds.append(gd)
        true_freq.append(ds)
        true_time.append(spectrum_to_time_agdmd(ds, nt))

    modes_t = np.vstack(true_time)
    modes_f = np.vstack(true_freq)
    gds = np.vstack(true_gds)
    signal = np.real(modes_t.sum(axis=0))

    return {
        "t": t,
        "signal": signal,
        "f": f,
        "fs": np.array([fs]),
        "true_gds": gds,
        "true_modes_time": np.real(modes_t),
        "true_modes_freq": modes_f,
    }


def stft_agncmd(
    sig: np.ndarray,
    samp_freq: float,
    n_freq: int = 1002,
    win_len: int = 108,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compact STFT matching MATLAB ``STFT.m`` in the AGDMD toolbox
    (Gaussian window, ``sigma=0.28``).

    :return: ``(spec_abs, freq_axis)`` with ``spec_abs`` shaped
             ``(n_pos_freq, N)``.
    """
    x = np.asarray(sig, dtype=float).ravel()
    xa = hilbert(x)
    n = xa.size
    win_len = int(np.ceil(win_len / 2.0) * 2)
    tt = np.linspace(-1.0, 1.0, win_len)
    sigma = 0.28
    win = (np.pi * sigma**2) ** (-0.25) * np.exp(-(tt**2) / (2.0 * sigma**2))
    lh = (win_len - 1) / 2.0

    spec = np.zeros((n_freq, n), dtype=complex)
    half = int(round(n_freq / 2.0))
    lh_i = int(lh)
    for i in range(n):
        tau_lo = -min(half - 1, lh_i, i)
        tau_hi = min(half - 1, lh_i, n - 1 - i)
        tau = np.arange(tau_lo, tau_hi + 1)
        rsig = xa[i + tau] * np.conj(win[lh_i + tau])
        spec[: rsig.size, i] = rsig

    spec = np.fft.fftshift(np.fft.fft(spec, axis=0), axes=0)
    n_level = spec.shape[0]
    f = np.linspace(0.0, samp_freq / 2.0, int(round(n_level / 2.0)))
    spec_pos = np.abs(spec[int(round(n_level / 2.0)) :, :])
    return spec_pos, f


# ---------------------------------------------------------------------------
# Public estimator
# ---------------------------------------------------------------------------


class AGNCMD(object):
    """
    Adaptive Generalized Dispersive Mode Decomposition (AGDMD / AGNCMD).

    Recursively extracts ``max_modes`` nonlinear dispersive components.  Each
    iteration:

    1. estimates an initial GD with :func:`agdi`
    2. estimates bandwidth ``alpha`` with :func:`bandwidth_estimation`
    3. runs adaptive GDMD (:func:`agdmd_core`)
    4. subtracts the recovered time-domain mode from the residual
    """

    def __init__(
        self,
        beta: float = 1e-7,
        max_modes: int = 3,
        tol: float = 1e-8,
        max_iter: int = 300,
    ) -> None:
        """
        :param beta: GD-increment smoothness (smaller → smoother)
        :param max_modes: maximum number of dispersive modes (MATLAB ``Kmax``)
        :param tol: adaptive-GDMD relative convergence tolerance
        :param max_iter: maximum demodulation iterations per mode
        """
        self.beta = float(beta)
        self.max_modes = int(max_modes)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.freq_: Optional[np.ndarray] = None
        self.init_gds_: Optional[np.ndarray] = None
        self.group_delays_: Optional[np.ndarray] = None
        self.modes_freq_: Optional[np.ndarray] = None
        self.modes_time_: Optional[np.ndarray] = None
        self.alphas_: Optional[List[np.ndarray]] = None
        self.residual_: Optional[np.ndarray] = None

    def __call__(
        self,
        signal: np.ndarray,
        fs: float,
        return_all: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        return self.fit_transform(signal, fs=fs, return_all=return_all)

    def __str__(self) -> str:
        return "Adaptive Generalized Dispersive Mode Decomposition " "(AGDMD / AGNCMD)"

    def fit_transform(
        self,
        signal: np.ndarray,
        fs: float,
        return_all: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Decompose a real dispersive signal.

        :param signal: 1-D real time series
        :param fs: sampling frequency (Hz)
        :param return_all: if True, also return ``(f, IGD, EGD, modes_freq,
                           alphas)``
        :return: time-domain modes ``(K, N)``, or the extended tuple when
                 ``return_all`` is True
        """
        x = np.asarray(signal, dtype=float).ravel()
        if x.ndim != 1 or x.size < 16:
            raise ValueError("signal must be a 1-D array with length >= 16")
        fs = float(fs)
        if fs <= 0:
            raise ValueError("fs must be positive")
        if self.max_modes < 1:
            raise ValueError("max_modes must be >= 1")

        nt = x.size
        duration = nt / fs
        nf = nt // 2 + 1
        freq = np.arange(nf, dtype=float) / duration

        residual = x.copy()
        init_gds = np.zeros((self.max_modes, nf), dtype=float)
        est_gds = np.zeros((self.max_modes, nf), dtype=float)
        modes_f = np.zeros((self.max_modes, nf), dtype=complex)
        modes_t = np.zeros((self.max_modes, nt), dtype=float)
        alphas: List[np.ndarray] = []

        k_eff = 0
        for i in range(self.max_modes):
            igd = agdi(residual, fs, self.beta)
            _, ialpha = bandwidth_estimation(residual, fs, igd)

            dsn = np.fft.fft(residual)[:nf]
            egd, mode_f, alpha_hist = agdmd_core(
                dsn,
                duration,
                igd[np.newaxis, :],
                ialpha,
                self.beta,
                residual,
                fs,
                tol=self.tol,
                max_iter=self.max_iter,
            )
            mode_t = spectrum_to_time_agdmd(mode_f, nt)

            init_gds[i] = igd
            est_gds[i] = egd
            modes_f[i] = mode_f
            modes_t[i] = mode_t
            alphas.append(alpha_hist)
            residual = residual - mode_t
            k_eff = i + 1

            # Early stop if residual energy collapses
            if norm(residual) < 1e-10 * (norm(x) + 1e-30):
                break

        init_gds = init_gds[:k_eff]
        est_gds = est_gds[:k_eff]
        modes_f = modes_f[:k_eff]
        modes_t = modes_t[:k_eff]

        self.freq_ = freq
        self.init_gds_ = init_gds
        self.group_delays_ = est_gds
        self.modes_freq_ = modes_f
        self.modes_time_ = modes_t
        self.alphas_ = alphas
        self.residual_ = residual

        if return_all:
            # Match MATLAB order: f, IGD, EGD, modef, modet, alpha
            return modes_t, freq, init_gds, est_gds, modes_f, alphas
        return modes_t


# Paper / MATLAB name
AGDMD = AGNCMD


def agncmd(
    signal: np.ndarray,
    fs: float,
    beta: float = 1e-7,
    max_modes: int = 3,
    tol: float = 1e-8,
    max_iter: int = 300,
    return_all: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """Functional wrapper around :class:`AGNCMD`."""
    return AGNCMD(
        beta=beta, max_modes=max_modes, tol=tol, max_iter=max_iter
    ).fit_transform(signal, fs=fs, return_all=return_all)


agdmd = agncmd
