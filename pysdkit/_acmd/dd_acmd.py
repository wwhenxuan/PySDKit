# -*- coding: utf-8 -*-
"""
Data-driven Adaptive Chirp Mode Decomposition (DD-ACMD).

Wang H, Chen S, Zhai W. Data-driven adaptive chirp mode decomposition with
application to machine fault diagnosis under non-stationary conditions.
Mechanical Systems and Signal Processing, 2023.

MATLAB reference (File Exchange 121373):
https://www.mathworks.com/matlabcentral/fileexchange/121373

Relationship to ``pysdkit._acmd.acmd.ACMD``
------------------------------------------
DD-ACMD **reuses classical ACMD as the inner mode extractor**.  What changes
is the *outer* recursion:

* **ACMD** needs a user-provided (or Fourier-peak / ridge) initial IF and a
  fixed mode count ``K``.
* **DD-ACMD** (1) extracts a zero-IF trend first, (2) builds the next IF seed
  by *derivative-normalization* (IF-DN) + iterative time-varying low-pass
  (DDIFI), (3) calls ACMD with that seed, (4) filters the residual with TVLP,
  and (5) stops when residual energy falls below a fraction of the original
  signal — no STFT ridge or spectrum peak is required.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import firwin, hilbert

from pysdkit._acmd.acmd import ACMD, curve_smooth, differ
from pysdkit._vmd.base import Base


# ---------------------------------------------------------------------------
# Low-level helpers (MATLAB: findev / arccos / low_filter / IF_DN / DDIFI / TVLP)
# ---------------------------------------------------------------------------


def find_extrema(sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local maxima, their indices, and a piecewise-constant upper envelope
    (MATLAB ``findev``).

    :return: ``(max_values, max_indices, upper_piecewise)``
    """
    sig = np.asarray(sig, dtype=float).ravel()
    L = sig.size
    if L < 3:
        idx = np.array([0], dtype=int) if L else np.array([], dtype=int)
        vals = sig[idx] if L else np.array([], dtype=float)
        up = np.full(L, float(sig[0]) if L else 0.0)
        return vals, idx, up

    # MATLAB: MaxVec = find(diff(diff(Sig) > 0) < 0) + 1
    interior = np.where(np.diff((np.diff(sig) > 0).astype(int)) < 0)[0] + 1
    max_vec = interior.astype(int)

    if sig[0] > sig[1]:
        max_vec = np.concatenate(([0], max_vec))
    if sig[-1] > sig[-2]:
        max_vec = np.concatenate((max_vec, [L - 1]))

    # unique sorted (endpoints may duplicate rare plateaus)
    max_vec = np.unique(max_vec)
    maxima = sig[max_vec]

    up = np.empty(L, dtype=float)
    if max_vec.size == 1:
        up[:] = maxima[0]
    else:
        for i in range(max_vec.size - 1):
            up[max_vec[i] : max_vec[i + 1] + 1] = maxima[i]
        up[: max_vec[0] + 1] = maxima[0]
        up[max_vec[-1] :] = maxima[-1]
    return maxima, max_vec, up


def phase_arccos(g: np.ndarray) -> np.ndarray:
    """
    Cumulative phase via piecewise ``acos`` with slope-based folding
    (MATLAB custom ``arccos`` — *not* ``np.arccos`` alone).

    Returns phase in units of cycles (already scaled by ``0.5/π · acos``).
    """
    g = np.asarray(g, dtype=float).ravel()
    s = g.size
    if s == 0:
        return g.copy()
    g_clip = np.clip(g, -1.0, 1.0)
    f1 = g_clip.copy()
    theta = (0.5 / np.pi) * np.arccos(f1)
    temp = 0.0
    for i in range(2, s):
        if np.sign(g[i] - g[i - 1]) != np.sign(g[i - 1] - g[i - 2]):
            temp = float(theta[i - 1])
        if g[i] - g[i - 1] > 0.0:
            f1[i] = -g_clip[i]
        else:
            f1[i] = g_clip[i]
        # keep acos domain
        f1[i] = float(np.clip(f1[i], -1.0, 1.0))
        theta[i] = (0.5 / np.pi) * np.arccos(f1[i]) + temp
    return theta


def low_filter(
    sig: np.ndarray, cut_freq: float, samp_freq: float
) -> np.ndarray:
    """
    Linear-phase FIR low-pass with delay compensation (MATLAB ``low_filter``).

    Works for real or complex inputs (FIR applied via ``np.convolve``).
    """
    sig = np.asarray(sig)
    n0 = sig.size
    if n0 < 4:
        return sig.copy()

    n = int(np.floor(n0 * 0.8))
    L = n if (n % 2 == 0) else n + 1
    # MATLAB fir1(L, Wn) with Wn = cut/(fs/2); L is filter *order*
    nyq = 0.5 * float(samp_freq)
    wn = float(cut_freq) / nyq
    wn = float(np.clip(wn, 1e-6, 0.999999))
    numtaps = L + 1
    b = firwin(numtaps, wn, pass_zero="lowpass")
    filtered = np.convolve(b, sig, mode="full")
    start = L // 2
    return filtered[start : start + n0]


def if_derivative_normalization(
    sig: np.ndarray, samp_freq: float, beta: float = 1e-11
) -> np.ndarray:
    """
    Instantaneous-frequency estimate via derivative normalization (MATLAB ``IF_DN``).

    Steps: differentiate → normalize between consecutive extrema → custom
    ``arccos`` phase → differentiate → Tikhonov smooth.
    """
    sig = np.asarray(sig, dtype=float).ravel()
    ds = differ(sig, 1.0 / float(samp_freq))
    _, max_vec, _ = find_extrema(ds)
    _, min_vec, _ = find_extrema(-ds)
    mv = np.sort(np.concatenate((max_vec, min_vec)))
    mv = np.unique(mv)
    g = np.zeros_like(ds)
    for i in range(mv.size - 1):
        i0, i1 = int(mv[i]), int(mv[i + 1])
        lav = abs(ds[i0] - ds[i1]) / 2.0
        lm = (ds[i0] + ds[i1]) / 2.0
        if lav < np.finfo(float).eps:
            g[i0 : i1 + 1] = 0.0
        else:
            g[i0 : i1 + 1] = (ds[i0 : i1 + 1] - lm) / lav
    g = np.clip(g, -1.0, 1.0)
    theta = phase_arccos(g)
    inst_f = differ(theta, 1.0 / float(samp_freq))
    inst_f = np.real(curve_smooth(inst_f, beta))
    return inst_f


def time_varying_lowpass(
    sig: np.ndarray,
    samp_freq: float,
    e_if: np.ndarray,
    c_pass: float,
    cutoff_scale: float = 1.1,
) -> np.ndarray:
    """
    Time-varying low-pass filter (MATLAB ``TVLP``).

    Demodulate by ``exp(-j ∫ 2π (IF − Cpass) dt)``, FIR-lowpass at
    ``cutoff_scale * Cpass``, then remodulate.  ``cutoff_scale=1.1`` matches
    the fully open-source MATLAB package (``low_filter`` path).
    """
    sig = np.asarray(sig, dtype=float).ravel()
    e_if = np.asarray(e_if, dtype=float).ravel()
    n = sig.size
    fs = float(samp_freq)
    t = np.arange(n, dtype=float) / fs
    phase = 2.0 * np.pi * cumulative_trapezoid(e_if - float(c_pass), t, initial=0.0)
    analytic = hilbert(sig)
    shifted = analytic * np.exp(-1j * phase)
    cut = max(float(cutoff_scale) * float(c_pass), 1e-6)
    # keep cutoff below Nyquist
    cut = min(cut, 0.49 * fs)
    filtered = low_filter(shifted, cut, fs)
    restored = filtered * np.exp(1j * phase)
    return np.real(restored)


def data_driven_if_init(
    sig: np.ndarray,
    samp_freq: float,
    beta: float = 1e-11,
    max_iter: int = 10,
    tol: float = 0.01,
) -> np.ndarray:
    """
    Data-driven IF initialization (MATLAB ``DDIFI``).

    Iterate IF-DN → TVLP until the IF-norm change falls below ``tol``.
    """
    residual = np.asarray(sig, dtype=float).ravel().copy()
    n = residual.size
    iif = np.full(n, float(samp_freq), dtype=float)
    last = iif.copy()
    for _ in range(int(max_iter)):
        inst_f = if_derivative_normalization(residual, samp_freq, beta=beta)
        residual = time_varying_lowpass(
            residual, samp_freq, inst_f, float(np.max(inst_f))
        )
        denom = norm_safe(iif)
        delta = abs(norm_safe(inst_f) - denom) / denom
        last = inst_f
        if delta < tol:
            break
        iif = inst_f
    return last


def norm_safe(x: np.ndarray) -> float:
    nrm = float(np.linalg.norm(np.asarray(x, dtype=float).ravel()))
    return nrm if nrm > 0.0 else np.finfo(float).eps


def add_noise(
    n: int,
    mean: float = 0.0,
    std: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Gaussian noise matching MATLAB ``addnoise``."""
    rng = np.random.default_rng() if rng is None else rng
    y = rng.standard_normal(int(n))
    y = y / (np.std(y) + np.finfo(float).eps)
    y = y - np.mean(y)
    return mean + std * y


def generate_stationary_demo(
    fs: float = 300.0,
    duration: float = 1.0,
    noise_std: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """MATLAB Test (stationary signal) mixture."""
    t = np.arange(0.0, duration, 1.0 / fs)
    sig1 = 4.0 * t**2
    sig2 = 0.5 * np.cos(40.0 * np.pi * t)
    sig3 = 0.5 * np.cos(50.0 * np.pi * t)
    sig4 = 0.5 * np.cos(60.0 * np.pi * t)
    clean = sig1 + sig2 + sig3 + sig4
    noisy = clean + add_noise(clean.size, 0.0, noise_std, rng=rng)
    return {
        "t": t,
        "fs": fs,
        "clean": clean,
        "signal": noisy,
        "modes": np.vstack([sig1, sig2, sig3, sig4]),
        "ifs": np.vstack(
            [
                np.zeros_like(t),
                np.full_like(t, 20.0),
                np.full_like(t, 25.0),
                np.full_like(t, 30.0),
            ]
        ),
    }


def generate_nonstationary_demo(
    fs: float = 1500.0,
    duration: float = 3.0,
    noise_std: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """MATLAB Test (noisy non-stationary) mixture."""
    t = np.arange(0.0, duration, 1.0 / fs)
    a1 = 1.0 + 0.3 * np.cos(2.0 * np.pi * t)
    sig1 = a1 * np.cos(2.0 * np.pi * (8 * t**3 + 16 * t**2 + 80 * t) + np.sin(10 * np.pi * t))
    if1 = 24 * t**2 + 32 * t + 80 + 5 * np.cos(10 * np.pi * t)
    a2 = 1.0 + 0.3 * np.cos(2.0 * np.pi * t + np.pi / 4)
    sig2 = a2 * np.cos(2.0 * np.pi * (6 * t**3 + 12 * t**2 + 60 * t) + np.sin(10 * np.pi * t))
    if2 = 18 * t**2 + 24 * t + 60 + 5 * np.cos(10 * np.pi * t)
    a3 = 1.0 + 0.3 * np.cos(2.0 * np.pi * t + np.pi / 2)
    sig3 = a3 * np.cos(2.0 * np.pi * (4 * t**3 + 8 * t**2 + 40 * t) + np.sin(10 * np.pi * t))
    if3 = 12 * t**2 + 16 * t + 40 + 5 * np.cos(10 * np.pi * t)
    clean = sig1 + sig2 + sig3
    noisy = clean + add_noise(clean.size, 0.0, noise_std, rng=rng)
    return {
        "t": t,
        "fs": fs,
        "clean": clean,
        "signal": noisy,
        "modes": np.vstack([sig1, sig2, sig3]),
        "ifs": np.vstack([if1, if2, if3]),
    }


def generate_close_modes_demo(
    fs: float = 800.0,
    duration: float = 2.0,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """MATLAB Test (close chirp modes) mixture."""
    t = np.arange(0.0, duration, 1.0 / fs)
    sig1 = np.exp(-0.1 * t) * np.cos(2.0 * np.pi * (-60 * t**3 + 180 * t**2 + 100 * t))
    if1 = -180 * t**2 + 360 * t + 100
    sig2 = np.exp(-0.2 * t) * np.cos(2.0 * np.pi * (-60 * t**3 + 180 * t**2 + 90 * t))
    if2 = -180 * t**2 + 360 * t + 90
    sig3 = np.exp(-0.3 * t) * np.cos(2.0 * np.pi * (-60 * t**3 + 180 * t**2 + 80 * t))
    if3 = -180 * t**2 + 360 * t + 80
    clean = sig1 + sig2 + sig3
    noisy = clean + add_noise(clean.size, 0.0, noise_std, rng=rng) if noise_std > 0 else clean.copy()
    return {
        "t": t,
        "fs": fs,
        "clean": clean,
        "signal": noisy,
        "modes": np.vstack([sig1, sig2, sig3]),
        "ifs": np.vstack([if1, if2, if3]),
    }


# ---------------------------------------------------------------------------
# DD-ACMD class
# ---------------------------------------------------------------------------


class DD_ACMD(Base):
    """
    Data-driven Adaptive Chirp Mode Decomposition.

    Outer loop around classical :class:`~pysdkit._acmd.acmd.ACMD`:

    1. Extract a **trend** with zero initial IF.
    2. Estimate the next IF by **DDIFI** (IF-DN + TVLP).
    3. Run ACMD; subtract the mode; stop if
       ``‖residual‖² / ‖signal‖² < energy_tol``.
    4. Otherwise apply TVLP to the residual and repeat.

    MATLAB: ``DDACMD.m`` (File Exchange 121373).
    """

    def __init__(
        self,
        fs: float,
        k_max: int = 20,
        alpha0: float = 1e-7,
        beta: float = 1e-10,
        tol: float = 1e-30,
        energy_tol: float = 0.01,
        ddifi_beta: float = 1e-11,
        ddifi_max_iter: int = 10,
        ddifi_tol: float = 0.01,
        max_iter: int = 300,
        tvlp_cutoff_scale: float = 1.1,
    ) -> None:
        """
        :param fs: sampling frequency (Hz)
        :param k_max: maximum number of modes (incl. trend)
        :param alpha0: ACMD bandwidth penalty (MATLAB DDACMD default ``1e-7``)
        :param beta: ACMD IF-increment smoothness (MATLAB ``1e-10``)
        :param tol: ACMD convergence tolerance (MATLAB ``1e-30`` → ~max_iter)
        :param energy_tol: residual-energy stop ratio ``ε``
        :param ddifi_beta: IF-DN smoothing penalty
        :param ddifi_max_iter: max DDIFI iterations
        :param ddifi_tol: DDIFI relative IF-norm stop
        :param max_iter: max ACMD iterations per mode
        :param tvlp_cutoff_scale: FIR cutoff = scale × max(IF)
        """
        super().__init__()
        if float(fs) <= 0:
            raise ValueError("fs must be positive")
        self.fs = float(fs)
        self.k_max = int(k_max)
        self.alpha0 = float(alpha0)
        self.beta = float(beta)
        self.tol = float(tol)
        self.energy_tol = float(energy_tol)
        self.ddifi_beta = float(ddifi_beta)
        self.ddifi_max_iter = int(ddifi_max_iter)
        self.ddifi_tol = float(ddifi_tol)
        self.max_iter = int(max_iter)
        self.tvlp_cutoff_scale = float(tvlp_cutoff_scale)

        self.imfs_: Optional[np.ndarray] = None
        self.ifs_: Optional[np.ndarray] = None
        self.ias_: Optional[np.ndarray] = None
        self.ini_ifs_: Optional[np.ndarray] = None

    def __call__(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        return self.fit_transform(signal, return_all=return_all)

    def __str__(self) -> str:
        return "Data-driven Adaptive Chirp Mode Decomposition (DD-ACMD)"

    def _acmd_extractor(self) -> ACMD:
        return ACMD(
            K=1,
            fs=self.fs,
            alpha0=self.alpha0,
            beta=self.beta,
            tol=self.tol,
            max_iter=self.max_iter,
        )

    def extract_trend(
        self, signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Mode 1: ACMD with zero initial IF (MATLAB trend step)."""
        signal = np.asarray(signal, dtype=float).ravel()
        init_if = np.zeros(signal.size, dtype=float)
        sest, if_est, ia_est = self._acmd_extractor().extract_mode(signal, init_if)
        return sest, if_est, ia_est, init_if

    def estimate_init_if(self, signal: np.ndarray) -> np.ndarray:
        """Public wrapper for DDIFI."""
        return data_driven_if_init(
            signal,
            self.fs,
            beta=self.ddifi_beta,
            max_iter=self.ddifi_max_iter,
            tol=self.ddifi_tol,
        )

    def fit_transform(
        self,
        signal: np.ndarray,
        return_all: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Run DD-ACMD.

        :param signal: 1-D real record
        :param return_all: if True, also return ``(ini_IF, eIF, eIA)``
        :return: ``IMF`` of shape ``(K, N)``, or
                 ``(IMF, ini_IF, eIF, eIA)`` when ``return_all``
        """
        sig0 = np.asarray(signal, dtype=float).ravel()
        if sig0.size < 8:
            raise ValueError("signal length must be >= 8")

        n = sig0.size
        residual = sig0.copy()
        energy0 = float(np.sum(sig0**2)) + np.finfo(float).eps

        ini_ifs = np.zeros((self.k_max, n), dtype=float)
        e_ifs = np.zeros((self.k_max, n), dtype=float)
        e_ias = np.zeros((self.k_max, n), dtype=float)
        imfs = np.zeros((self.k_max, n), dtype=float)

        acmd = self._acmd_extractor()
        k_out = 0

        for i in range(self.k_max):
            if i == 0:
                init_if = np.zeros(n, dtype=float)
            else:
                init_if = data_driven_if_init(
                    residual,
                    self.fs,
                    beta=self.ddifi_beta,
                    max_iter=self.ddifi_max_iter,
                    tol=self.ddifi_tol,
                )

            sest, if_est, ia_est = acmd.extract_mode(residual, init_if)
            ini_ifs[i] = init_if
            e_ifs[i] = if_est
            e_ias[i] = ia_est
            imfs[i] = sest
            residual = residual - sest
            k_out = i + 1

            if i >= 1:
                epsilon = float(np.sum(residual**2)) / energy0
                if epsilon < self.energy_tol:
                    break
                c_pass = float(np.max(if_est))
                residual = time_varying_lowpass(
                    residual,
                    self.fs,
                    if_est,
                    c_pass,
                    cutoff_scale=self.tvlp_cutoff_scale,
                )

        self.ini_ifs_ = ini_ifs[:k_out]
        self.ifs_ = e_ifs[:k_out]
        self.ias_ = e_ias[:k_out]
        self.imfs_ = imfs[:k_out]

        if return_all:
            return self.imfs_, self.ini_ifs_, self.ifs_, self.ias_
        return self.imfs_


# Alias matching MATLAB naming
DDACMD = DD_ACMD
