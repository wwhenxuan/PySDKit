# -*- coding: utf-8 -*-
"""
Bandwidth-Aware Adaptive Chirp Mode Decomposition (BA-ACMD).

Chen S, Guo L, Fan J, Yi C, Wang K, Zhai W.
Bandwidth-aware adaptive chirp mode decomposition for railway bearing
fault diagnosis. Structural Health Monitoring, 2024; 23(2):876-902.

MATLAB reference (File Exchange):
https://www.mathworks.com/matlabcentral/fileexchange/132792
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks, hilbert
from scipy.sparse import eye as speye
from scipy.sparse import hstack, spdiags, vstack
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Low-level helpers (MATLAB: Differ / SNR / addnoise / GN_SE / IFIAextrc / …)
# ---------------------------------------------------------------------------


def differ(y: np.ndarray, delta: float) -> np.ndarray:
    """Central difference of a 1-D series (MATLAB ``Differ``)."""
    y = np.asarray(y, dtype=float).ravel()
    L = y.size
    if L < 2:
        return np.zeros_like(y)
    mid = np.zeros(max(L - 2, 0), dtype=float)
    for i in range(1, L - 1):
        mid[i - 1] = (y[i + 1] - y[i - 1]) / (2.0 * delta)
    return np.concatenate(
        (
            np.array([(y[1] - y[0]) / delta], dtype=float),
            mid,
            np.array([(y[-1] - y[-2]) / delta], dtype=float),
        )
    )


def add_noise(
    n: int,
    mean: float = 0.0,
    std: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Unit-normalized Gaussian noise with given mean / std (MATLAB ``addnoise``)."""
    if rng is None:
        rng = np.random.default_rng()
    y = rng.standard_normal(int(n))
    y = y / (np.std(y) + np.finfo(float).eps)
    y = y - np.mean(y)
    return mean + std * y


def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR in dB between clean and noisy records (MATLAB ``SNR``)."""
    clean = np.asarray(clean, dtype=float).ravel()
    noisy = np.asarray(noisy, dtype=float).ravel()
    ps = np.sum((clean - np.mean(clean)) ** 2)
    pn = np.sum((clean - noisy) ** 2) + np.finfo(float).eps
    return float(10.0 * np.log10(ps / pn))


def gini_squared_envelope(signal: np.ndarray) -> float:
    """Gini index of the squared Hilbert envelope (MATLAB ``GN_SE``)."""
    signal = np.asarray(signal, dtype=float).ravel()
    se = np.abs(hilbert(signal)) ** 2
    nn = se.size
    se_sorted = np.sort(se)
    weights = (nn - np.arange(1, nn + 1) + 0.5) / nn
    temp = float(np.sum(weights * se_sorted))
    return float(1.0 - 2.0 * temp / (np.linalg.norm(se, 1) + np.finfo(float).eps))


def extract_if_ia(
    y: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Instantaneous frequency / amplitude via padded analytic signal
    (MATLAB ``IFIAextrc``).
    """
    y = np.asarray(y, dtype=float).ravel()
    n_pad = int(round(0.5 * y.size))
    if n_pad < 1:
        analytic = hilbert(y)
        phase = np.unwrap(np.angle(analytic))
        return differ(phase, 1.0 / fs) / (2.0 * np.pi), np.abs(analytic)

    # MATLAB: [fliplr(y(2:2+num_padding-1)) y fliplr(y(end-num_padding:end-1))]
    left = y[1 : n_pad + 1][::-1]
    right = y[-(n_pad + 1) : -1][::-1]
    yp = np.concatenate([left, y, right])
    analytic = hilbert(yp)
    phase = np.unwrap(np.angle(analytic))
    inst_f = differ(phase, 1.0 / fs) / (2.0 * np.pi)
    inst_a = np.abs(analytic)
    lo = n_pad
    hi = yp.size - n_pad
    return inst_f[lo:hi], inst_a[lo:hi]


def impulse_times(phi: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample indices / times where phase ``phi`` crosses multiples of ``2π``
    (MATLAB ``impultime``).
    """
    phi = np.asarray(phi, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    n = t.size
    index = [0]
    pre_phi = phi[0]
    increment = np.zeros(n, dtype=float)
    for jj in range(1, n):
        increment[jj] = phi[jj] - pre_phi
        if increment[jj - 1] <= 2.0 * np.pi < increment[jj]:
            if (increment[jj] - 2.0 * np.pi) < (2.0 * np.pi - increment[jj - 1]):
                index.append(jj)
                pre_phi = phi[jj]
            else:
                index.append(jj - 1)
                pre_phi = phi[jj - 1]
    index_arr = np.asarray(index, dtype=int)
    return t[index_arr], index_arr


def bandwidth_to_alpha(bw: float, ac: float = 0.3) -> float:
    """
    Map normalized bandwidth to ACMD penalty ``alpha0`` (paper / BAACMD.m).

    Default coefficients correspond to amplitude threshold ``Ac = 0.3``.
    """
    bw = float(bw)
    if ac >= 0.5:
        # MATLAB commented alternative for Ac = 0.5
        delta = max(bw - 0.001458, 1e-12)
        return float((3.7092 * delta) ** 3.9231)
    delta = max(bw - 0.002137, 1e-12)
    return float((2.9868 * delta) ** 3.9323)


# ---------------------------------------------------------------------------
# Spectrum trend / Fourier fitting (MATLAB: coef_ovefour / Spectrendgene)
# ---------------------------------------------------------------------------


def coef_overcomplete_fourier(
    f: np.ndarray,
    samp_freq: float,
    order_amp: int,
    alpha: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Over-complete Fourier fit of a 1-D series (MATLAB ``coef_ovefour``).

    :return: ``(fitf, finte)`` fitted series and its analytic integral basis
             projection (kept for MATLAB parity).
    """
    f = np.asarray(f, dtype=float).ravel()
    n = f.size
    order_amp = int(order_amp)
    order_amp = 2 * order_amp + 1
    dt = np.arange(n, dtype=float) / float(samp_freq)
    f0 = float(samp_freq) / 2.0 / n

    tmatrix = np.zeros((n, order_amp), dtype=float)
    tmatrix[:, 0] = 1.0
    half = (order_amp + 1) / 2.0
    for j in range(1, order_amp):
        # MATLAB 1-based j → Python j+1; freq index (j) for cos branch
        tmatrix[:, j] = np.cos(2.0 * np.pi * f0 * j * dt)
        if (j + 1) > half:
            k = (j + 1) - half
            tmatrix[:, j] = np.sin(2.0 * np.pi * f0 * k * dt)

    tmatrix_inte = np.zeros((n, order_amp), dtype=float)
    tmatrix_inte[:, 0] = dt
    for j in range(1, order_amp):
        tmatrix_inte[:, j] = (
            1.0 / (2.0 * np.pi * f0 * j) * np.sin(2.0 * np.pi * f0 * j * dt)
        )
        if (j + 1) > half:
            k = (j + 1) - half
            tmatrix_inte[:, j] = (
                -1.0
                / (2.0 * np.pi * f0 * k)
                * np.cos(2.0 * np.pi * f0 * k * dt)
            )

    eye_m = np.eye(order_amp)
    coeff = np.linalg.solve(
        alpha * eye_m + tmatrix.T @ tmatrix, tmatrix.T @ f
    )
    fitf = tmatrix @ coeff
    finte = tmatrix_inte @ coeff
    return fitf, finte


def spectrum_trend_generate(
    sig: np.ndarray,
    fs: float,
    offset: float = 0.01,
    cut_pfreq: float = 0.0015,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted spectrum trend and ranked frequency intervals
    (MATLAB ``Spectrendgene``).

    :return: ``Spec, weight, Spec_trend, WeSpec_trend, sortInter``
             ``sortInter`` has shape ``(n_bands, 2)`` with ``[f_lo, f_hi]`` in Hz,
             ordered by descending weighted-trend peak.
    """
    sig = np.asarray(sig, dtype=float).ravel()
    n_sig = sig.size
    spec_full = 2.0 * np.abs(np.fft.fft(sig)) / n_sig
    n_half = int(round(spec_full.size / 2.0))
    spec = spec_full[:n_half]
    freq_bin = np.linspace(0.0, fs / 2.0, spec.size)

    order_amp = int(round(2 * n_sig * cut_pfreq))
    order_amp = max(order_amp, 1)
    spec_trend, _ = coef_overcomplete_fourier(spec, fs, order_amp)
    spec_trend = spec_trend + float(offset)

    # Local minima of the spectrum trend
    minima, _ = find_peaks(-spec_trend)
    temp_index = np.concatenate(
        ([0], minima.astype(int), [freq_bin.size - 1])
    )

    temp_frn = temp_index[:-1]
    temp_end = temp_index[1:]
    diff_index = temp_end - temp_frn
    delet = np.where(diff_index <= 0.005 * n_sig)[0]
    if delet.size and delet[0] == 0:
        delet = delet.copy()
        delet[0] = 1
    if delet.size:
        keep = np.ones(temp_index.size, dtype=bool)
        keep[delet] = False
        # Never drop the very first / last boundaries if that would empty bands
        if keep.sum() < 2:
            keep[:] = True
        temp_index = temp_index[keep]

    weight = np.zeros_like(freq_bin)
    for kk in range(temp_index.size - 1):
        a = int(temp_index[kk])
        b = int(temp_index[kk + 1])
        den = max(b - a, 1)
        weight[a : b + 1] = np.sum(spec[a : b + 1] ** 2) / den

    we_spec_trend = weight * spec_trend

    n_band = temp_index.size - 1
    freq_interval = np.zeros((n_band, 2), dtype=float)
    weight_max = np.zeros(n_band, dtype=float)
    for dd in range(n_band):
        a = int(temp_index[dd])
        b = int(temp_index[dd + 1])
        hi = max(b - 1, a)
        weight_max[dd] = float(np.max(we_spec_trend[a : hi + 1]))
        freq_interval[dd, 0] = freq_bin[a]
        freq_interval[dd, 1] = freq_bin[hi]

    order = np.argsort(-weight_max)
    sort_inter = freq_interval[order]
    return spec, weight, spec_trend, we_spec_trend, sort_inter


# ---------------------------------------------------------------------------
# Single-mode ACMD used by BA-ACMD (MATLAB folder ACMD.m — mean ΔIF update)
# ---------------------------------------------------------------------------


def acmd_extract(
    signal: np.ndarray,
    fs: float,
    init_if: np.ndarray,
    alpha0: float,
    beta: float = 1e-10,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract one chirp mode with BA-ACMD's ACMD update rule.

    Unlike the generic ``pysdkit.ACMD`` class, IF is updated by
    ``eIF ← eIF - mean(ΔIF)`` (MATLAB ``ACMD.m`` in the BA-ACMD package).

    :return: ``(sest, IFest, IAest)``
    """
    from scipy.sparse import csr_matrix, diags

    s = np.asarray(signal, dtype=float).ravel()
    e_if = np.asarray(init_if, dtype=float).ravel().copy()
    if e_if.size != s.size:
        raise ValueError("init_if length must match the signal")

    n = s.size
    t = np.arange(n, dtype=float) / float(fs)
    e = np.ones(n)
    e2 = -2.0 * e
    oper = spdiags(
        np.vstack((e[:-2], e2[1:-1], e[2:])),
        np.array([0, 1, 2]),
        n - 2,
        n,
    ).tocsc()
    spzeros = csr_matrix((n - 2, n))
    opedoub = (oper.T @ oper).tocsc()
    phim = vstack((hstack((oper, spzeros)), hstack((spzeros, oper))))
    phidoubm = (phim.T @ phim).tocsc()

    if_hist = np.zeros((max_iter, n))
    s_hist = np.zeros((max_iter, n))
    y_hist = np.zeros((max_iter, 2 * n))

    s_dif = tol + 1.0
    it = 0
    alpha = float(max(alpha0, 1e-12))
    ridge = 1e-10

    while s_dif > tol and it < max_iter:
        phase = cumulative_trapezoid(e_if, t, initial=0.0)
        cosm = np.cos(2.0 * np.pi * phase)
        sinm = np.sin(2.0 * np.pi * phase)
        cm = diags(cosm, 0, shape=(n, n), format="csc")
        sm = diags(sinm, 0, shape=(n, n), format="csc")
        kerm = hstack((cm, sm))
        kerdoubm = (kerm.T @ kerm).tocsc()

        # Mild ridge ≈ MATLAB ``\`` least-squares behaviour on ill-conditioned systems
        a_mat = (1.0 / alpha) * phidoubm + kerdoubm + ridge * speye(2 * n, format="csc")
        ym = spsolve(a_mat, kerm.T @ s)
        if not np.all(np.isfinite(ym)):
            ym = np.linalg.lstsq(a_mat.toarray(), (kerm.T @ s), rcond=None)[0]
        si = np.asarray(kerm @ ym).ravel()
        s_hist[it, :] = si
        y_hist[it, :] = ym

        ycm = ym[:n]
        ysm = ym[n:]
        ycm_bar = differ(ycm, 1.0 / fs)
        ysm_bar = differ(ysm, 1.0 / fs)
        denom = ycm**2 + ysm**2 + np.finfo(float).eps
        delta_if = (ycm * ysm_bar - ysm * ycm_bar) / denom / (2.0 * np.pi)
        smooth = (1.0 / max(beta, 1e-16)) * opedoub + speye(n, format="csc")
        delta_if = spsolve(smooth, delta_if)
        if not np.all(np.isfinite(delta_if)):
            delta_if = np.linalg.lstsq(smooth.toarray(), delta_if, rcond=None)[0]
        e_if = e_if - float(np.mean(delta_if))
        if_hist[it, :] = e_if

        if it > 0:
            s_dif = (
                norm(s_hist[it] - s_hist[it - 1])
                / (norm(s_hist[it - 1]) + np.finfo(float).eps)
            ) ** 2
        it += 1

    it = max(it - 1, 0)
    sest = s_hist[it]
    if_est = if_hist[it]
    ycm = y_hist[it, :n]
    ysm = y_hist[it, n:]
    ia_est = np.sqrt(ycm**2 + ysm**2)
    return sest, if_est, ia_est


# ---------------------------------------------------------------------------
# Demo signal (MATLAB: genersig1 + Example2)
# ---------------------------------------------------------------------------


def generate_demo_components(
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Four synthetic components from MATLAB ``genersig1``.

    :return: ``impulse_w, impulse_random, impulse_bear, harmonic``
    """
    t = np.asarray(t, dtype=float).ravel()
    phi_rot = 2.0 * np.pi * (170.0 * t)
    phi_fcf1 = 2.0 * np.pi * 10.0 * t
    phi_fcf2 = 2.0 * np.pi * 80.0 * t

    fr1, fr2, fr3 = 700.0, 1300.0, 2000.0
    beta = beta1 = beta2 = 500.0

    _, index = impulse_times(phi_fcf1, t)
    kernel = np.exp(-beta * t) * np.sin(2.0 * np.pi * fr1 * t)
    impulse_w = np.zeros_like(t)
    for idx in index:
        sub = np.zeros_like(t)
        sub[idx:] = kernel[: t.size - idx]
        impulse_w = impulse_w + sub

    index_random = np.round(np.array([0.15, 0.25, 0.75]) * t.size).astype(int)
    amp_random = np.array([2.0, 5.0, 4.0])
    impulse_random = np.zeros_like(t)
    kernel_r = np.exp(-beta1 * t) * np.sin(2.0 * np.pi * fr2 * t)
    for amp, idx in zip(amp_random, index_random):
        idx = int(np.clip(idx, 0, t.size - 1))
        sub = np.zeros_like(t)
        sub[idx:] = kernel_r[: t.size - idx]
        impulse_random = impulse_random + amp * sub

    _, index2 = impulse_times(phi_fcf2, t)
    kernel_b = np.exp(-beta2 * t) * np.sin(2.0 * np.pi * fr3 * t)
    impulse_bear = np.zeros_like(t)
    for idx in index2:
        sub = np.zeros_like(t)
        sub[idx:] = kernel_b[: t.size - idx]
        impulse_bear = impulse_bear + 0.4 * sub

    harmonic = 0.04 * np.cos(phi_rot)
    return impulse_w, impulse_random, impulse_bear, harmonic


def generate_demo_signal(
    fs: float = 5000.0,
    duration: float = 1.0,
    noise_std: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Build the noisy mixture used in MATLAB ``Example2.m``.

    :return: ``t, noisy_signal, snr_db, components``
    """
    # MATLAB: t = 0:1/SampFreq:1  →  fs*duration + 1 samples
    n = int(round(duration * fs)) + 1
    t = np.arange(n, dtype=float) / float(fs)

    c1, c2, c3, c4 = generate_demo_components(t)
    clean = c1 + c2 + c3 + c4
    noise = add_noise(t.size, 0.0, noise_std, rng=rng)
    noisy = clean + noise
    snr_db = compute_snr(clean, noisy)
    components = {
        "impulse_w": c1,
        "impulse_random": c2,
        "impulse_bear": c3,
        "harmonic": c4,
        "clean": clean,
        "noise": noise,
    }
    return t, noisy, snr_db, components


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class BA_ACMD(object):
    """
    Bandwidth-Aware Adaptive Chirp Mode Decomposition.

    1. Build a weighted spectrum trend (WST) and rank frequency bands.
    2. For each band (strongest first), map bandwidth → ``alpha0`` and run
       single-mode ACMD with a constant mid-band IF initialization.
    3. Subtract the mode and stop when its correlation with the *original*
       signal falls below ``ce``.
    """

    def __init__(
        self,
        fs: float,
        beta: float = 1e-10,
        tol: float = 1e-7,
        ce: float = 0.3,
        offset: float = 0.01,
        cut_pfreq: float = 0.0015,
        max_iter: int = 100,
        ac: float = 0.3,
    ) -> None:
        """
        :param fs: sampling frequency (Hz)
        :param beta: IF-smoothing penalty in ACMD
        :param tol: ACMD convergence tolerance
        :param ce: correlation threshold for early stopping
        :param offset: additive offset on the fitted spectrum trend
        :param cut_pfreq: controls Fourier order of the spectrum-trend fit
        :param max_iter: max ACMD iterations per mode
        :param ac: amplitude coefficient selecting the Bw→α map (0.3 or 0.5)
        """
        if float(fs) <= 0:
            raise ValueError("fs must be positive")
        self.fs = float(fs)
        self.beta = float(beta)
        self.tol = float(tol)
        self.ce = float(ce)
        self.offset = float(offset)
        self.cut_pfreq = float(cut_pfreq)
        self.max_iter = int(max_iter)
        self.ac = float(ac)

        self.imfs: Optional[np.ndarray] = None
        self.instantaneous_frequencies: Optional[np.ndarray] = None
        self.instantaneous_amplitudes: Optional[np.ndarray] = None
        self.sort_intervals: Optional[np.ndarray] = None
        self.spec: Optional[np.ndarray] = None
        self.spec_trend: Optional[np.ndarray] = None
        self.weighted_spec_trend: Optional[np.ndarray] = None
        self.weight_factor: Optional[np.ndarray] = None

    def __call__(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self.fit_transform(signal, return_all=return_all)

    def __str__(self) -> str:
        return "Bandwidth-Aware Adaptive Chirp Mode Decomposition (BA-ACMD)"

    def compute_spectrum_trend(
        self, signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Instance wrapper around :func:`spectrum_trend_generate`."""
        return spectrum_trend_generate(
            signal, self.fs, offset=self.offset, cut_pfreq=self.cut_pfreq
        )

    def fit_transform(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Decompose ``signal`` with BA-ACMD.

        :param signal: 1-D real vibration record
        :param return_all: if True, also return IF and IA arrays
        :return: modes ``(K, N)`` [, IFs ``(K, N)``, IAs ``(K, N)``]
        """
        sig0 = np.asarray(signal, dtype=float).ravel()
        if sig0.size < 8:
            raise ValueError("signal length must be >= 8")

        (
            spec,
            weight,
            spec_trend,
            we_spec_trend,
            sort_inter,
        ) = self.compute_spectrum_trend(sig0)

        self.spec = spec
        self.weight_factor = weight
        self.spec_trend = spec_trend
        self.weighted_spec_trend = we_spec_trend
        self.sort_intervals = sort_inter

        n = sig0.size
        max_num = max(int(sort_inter.shape[0]), 1)
        comp_set = np.zeros((max_num, n), dtype=float)
        if_set = np.zeros((max_num, n), dtype=float)
        ia_set = np.zeros((max_num, n), dtype=float)

        residual = sig0.copy()
        last_ii = 0
        for ii in range(max_num):
            f_lo, f_hi = float(sort_inter[ii, 0]), float(sort_inter[ii, 1])
            bw = (f_hi - f_lo) / self.fs
            alpha0 = bandwidth_to_alpha(bw, ac=self.ac)
            init_if = 0.5 * (f_lo + f_hi) * np.ones(n, dtype=float)

            sest, if_est, ia_est = acmd_extract(
                residual,
                self.fs,
                init_if,
                alpha0=alpha0,
                beta=self.beta,
                tol=self.tol,
                max_iter=self.max_iter,
            )
            comp_set[ii] = sest
            if_set[ii] = if_est
            ia_set[ii] = ia_est
            residual = residual - sest
            last_ii = ii

            corr = np.corrcoef(sest, sig0)[0, 1]
            if not np.isfinite(corr) or corr < self.ce:
                break

        modes = comp_set[: last_ii + 1]
        ifs = if_set[: last_ii + 1]
        ias = ia_set[: last_ii + 1]

        self.imfs = modes
        self.instantaneous_frequencies = ifs
        self.instantaneous_amplitudes = ias

        if return_all:
            return modes, ifs, ias
        return modes


# Friendly aliases matching MATLAB names
BAACMD = BA_ACMD
Spectrendgene = spectrum_trend_generate
coef_ovefour = coef_overcomplete_fourier
ACMD_ba = acmd_extract
Differ = differ
GN_SE = gini_squared_envelope
IFIAextrc = extract_if_ia
impultime = impulse_times
addnoise = add_noise
SNR = compute_snr
