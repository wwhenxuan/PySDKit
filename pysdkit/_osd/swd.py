# -*- coding: utf-8 -*-
"""
Created on 2025/07/23
@author: Kai Wu
@email: kwu@xidian.edu.cn

Swarm Decomposition (SWD / SwD).

A non-stationary signal decomposition method based on swarm intelligence.
Cornerstone is *swarm filtering* (SwF): the input is the trajectory of a
virtual “prey”, hunted by a swarm whose collective trajectory yields a
filtered output.  SWD applies iterative SwF under different hunting
parameters (tied to the current dominant frequency), aligns / subtracts
each extracted oscillatory component, and stops when the residual no
longer contains a significant spectral peak.

Apostolidis, G. K. and Hadjileontiadis, L. J. (2017).
Swarm decomposition: A novel signal analysis using swarm intelligence.
Signal Processing, 132, 40–50.
https://doi.org/10.1016/j.sigpro.2016.09.004

MATLAB reference (encrypted ``.p`` toolbox):
https://github.com/gkaposto/Swarm-Decomposition

Implementation note
-------------------
The official ``SwD.p`` binary is closed-source.  This module reconstructs
SWD from the published sifting architecture and the GA-fitted maps
``M(ω̂)``, ``δ(ω̂)`` (Eq. 9).  Because those maps were calibrated to produce
*particular frequency responses*, SwF is realised here as an adaptive
band-pass around the target ``ω̂`` (bandwidth from ``δ``) followed by a
stabilised swarm–prey hunting pass that refines the time-domain waveform.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import correlate, savgol_filter, welch


def swf_params_from_frequency(omega_hat: float) -> Tuple[int, float]:
    """
    Map normalised frequency ``ω̂ = ω/π ∈ (0, 1]`` to SwF parameters
    ``(M, δ)`` (Eq. 9 of the paper).
    """
    omega_hat = float(np.clip(omega_hat, 0.05, 1.0))
    M = int(np.rint(33.46 * (omega_hat ** (-0.735)) - 29.1))
    M = int(np.clip(M, 2, 80))
    delta = -1.5 * omega_hat**2 + 3.454 * omega_hat - 0.01
    delta = float(np.clip(delta, 0.05, 1.5))
    return M, delta


def _bandwidth_from_delta(delta: float, omega_hat: float) -> float:
    """Relative band-pass width derived from the flexibility ``δ``."""
    # Larger δ → more flexible swarm → wider spectral support
    return float(np.clip(0.15 + 0.35 * delta, 0.12, 0.6)) * max(omega_hat, 0.05)


def bandpass_swf(
    signal: np.ndarray,
    omega_hat: float,
    delta: float,
) -> np.ndarray:
    """
    Frequency-domain realisation of a SwF response centred at ``ω̂``.

    A raised-Gaussian window on the positive frequency axis isolates the
    oscillatory band that the GA-calibrated ``(M, δ)`` pair is designed to
    recover (Fig. 2 of the paper).
    """
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n) * 2.0  # Nyquist == 1
    center = float(np.clip(omega_hat, 0.02, 0.98))
    width = _bandwidth_from_delta(delta, center)
    window = np.exp(-0.5 * ((freqs - center) / width) ** 2)
    window[0] = 0.0
    return np.fft.irfft(spec * window, n=n)


def _cohesion_force(positions: np.ndarray, d_cr: float) -> np.ndarray:
    """Cohesion force (Eq. 2) — attract when far, repel when close."""
    m = positions.size
    if m < 2:
        return np.zeros_like(positions)
    diff = positions[:, None] - positions[None, :]
    abs_diff = np.maximum(np.abs(diff), 1e-12 * d_cr)
    f_mat = -np.sign(diff) * np.log(abs_diff / d_cr)
    np.fill_diagonal(f_mat, 0.0)
    return np.sum(f_mat, axis=1) / float(m - 1)


def swarm_hunt(
    signal: np.ndarray,
    M: int,
    delta: float,
    d_cr: float,
) -> np.ndarray:
    """
    Time-domain swarm–prey hunting (Eqs. 1–5) with a stabilised update.

    Used as a gentle refiner after the spectral SwF projection.
    """
    x = np.asarray(signal, dtype=float).ravel()
    L = x.size
    M = int(max(2, M))
    d_cr = float(max(d_cr, 1e-12))
    step = float(np.clip(delta, 0.05, 0.35))
    beta = 1.0 / M

    offsets = d_cr * (np.arange(1, M + 1, dtype=float) - (M + 1) / 2.0)
    positions = x[0] + offsets
    velocities = np.zeros(M, dtype=float)
    y = np.empty(L, dtype=float)
    y[0] = beta * np.sum(positions)

    damp = 0.55
    for n in range(1, L):
        f_dr = x[n] - positions
        f_coh = _cohesion_force(positions, d_cr)
        velocities = damp * velocities + step * (f_dr + f_coh)
        positions = positions + step * velocities
        y[n] = beta * np.sum(positions)
    return y


def swarm_filter(
    signal: np.ndarray,
    omega_hat: float,
    M: Optional[int] = None,
    delta: Optional[float] = None,
    refine: bool = True,
) -> np.ndarray:
    """
    Full SwF: spectral projection at ``ω̂`` (+ optional hunting refinement).
    """
    if M is None or delta is None:
        M, delta = swf_params_from_frequency(omega_hat)
    y = bandpass_swf(signal, omega_hat=omega_hat, delta=delta)
    if refine and np.sum(y**2) > 1e-12:
        d_cr = float(np.sqrt(np.mean(y**2)) + 1e-12)
        y_h = swarm_hunt(y, M=M, delta=delta, d_cr=d_cr)
        # Blend: keep spectral isolation, adopt hunting waveform shape
        if np.sum(y_h**2) > 1e-12:
            scale = np.sqrt(np.sum(y**2) / (np.sum(y_h**2) + 1e-12))
            y = 0.5 * y + 0.5 * scale * y_h
            y = bandpass_swf(y, omega_hat=omega_hat, delta=delta)
    return y


def iterative_swf(
    signal: np.ndarray,
    omega_hat: float,
    std_th: float = 0.05,
    max_sift: int = 20,
    refine: bool = True,
) -> np.ndarray:
    """Iterative SwF (Algorithm 1) until consecutive outputs agree."""
    M, delta = swf_params_from_frequency(omega_hat)
    y_prev = np.asarray(signal, dtype=float).ravel()
    y = swarm_filter(y_prev, omega_hat=omega_hat, M=M, delta=delta, refine=refine)

    for _ in range(1, max_sift):
        denom = float(np.sum(y**2) + 1e-12)
        std_val = float(np.sum((y - y_prev) ** 2) / denom)
        if (not np.isfinite(std_val)) or std_val < std_th:
            break
        y_prev = y
        y = swarm_filter(y_prev, omega_hat=omega_hat, M=M, delta=delta, refine=refine)
        if not np.all(np.isfinite(y)):
            return y_prev
    return y


def _align_and_subtract(
    residual: np.ndarray, component: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-correlation alignment + least-squares gain (Eq. 11)."""
    L = residual.size
    corr = correlate(residual, component, mode="full")
    lag = int(np.argmax(np.abs(corr)) - (L - 1))
    if corr[lag + (L - 1)] < 0:
        component = -component
        corr = correlate(residual, component, mode="full")
        lag = int(np.argmax(corr) - (L - 1))

    aligned = np.roll(component, lag)
    if lag > 0:
        aligned[:lag] = 0.0
    elif lag < 0:
        aligned[lag:] = 0.0

    denom = float(np.dot(aligned, aligned) + 1e-12)
    gain = float(np.dot(residual, aligned) / denom)
    aligned = gain * aligned
    return residual - aligned, aligned


class SWD(object):
    """
    Swarm Decomposition (SWD).

    Parameters
    ----------
    P_th : float
        Relative spectral-peak threshold (``≈ 0.05–0.2``). Larger → coarser.
    StD_th : float
        Iterative-SwF convergence threshold (``≈ 0.01–0.1``).
    spectrum : {'welch', 'sg', 'fft'}
        Estimator used to pick the dominant residual frequency.
    max_components : int
        Upper bound on the number of returned oscillatory components.
    max_sift : int
        Max SwF repetitions inside one iterative-SwF call.
    refine : bool
        If True, apply the time-domain hunting refiner after each spectral SwF.
    """

    def __init__(
        self,
        P_th: float = 0.05,
        StD_th: float = 0.05,
        spectrum: str = "welch",
        sg_order: int = 2,
        sg_window: int = 15,
        welch_window: Optional[int] = None,
        welch_noverlap: Optional[int] = None,
        welch_nfft: Optional[int] = None,
        max_components: int = 15,
        max_sift: int = 15,
        min_omega_hat: float = 0.05,
        freq_merge_tol: float = 0.03,
        refine: bool = True,
    ) -> None:
        self.P_th = float(P_th)
        self.StD_th = float(StD_th)
        self.spectrum = str(spectrum).lower()
        self.sg_order = int(sg_order)
        self.sg_window = int(sg_window)
        self.welch_window = welch_window
        self.welch_noverlap = welch_noverlap
        self.welch_nfft = welch_nfft
        self.max_components = int(max_components)
        self.max_sift = int(max_sift)
        self.min_omega_hat = float(min_omega_hat)
        self.freq_merge_tol = float(freq_merge_tol)
        self.refine = bool(refine)

        if self.spectrum not in ("welch", "sg", "fft"):
            raise ValueError("spectrum must be 'welch', 'sg' or 'fft'")

        self.signal: Optional[np.ndarray] = None
        self.components: Optional[np.ndarray] = None
        self.residue: Optional[np.ndarray] = None
        self.omegas: Optional[np.ndarray] = None

    def __call__(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self.fit_transform(signal=signal, return_all=return_all)

    def __str__(self) -> str:
        return "Swarm Decomposition (SWD)"

    def _power_spectrum(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        if self.spectrum == "welch":
            nperseg = self.welch_window or max(16, n // 8)
            nperseg = min(nperseg, n)
            noverlap = (
                self.welch_noverlap
                if self.welch_noverlap is not None
                else nperseg // 2
            )
            nfft = self.welch_nfft or int(2 ** np.ceil(np.log2(max(n, nperseg))))
            return welch(
                x,
                fs=2.0,
                nperseg=nperseg,
                noverlap=min(noverlap, nperseg - 1),
                nfft=nfft,
            )

        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.linspace(0.0, 1.0, spec.size)
        if self.spectrum == "sg":
            win = self.sg_window
            if win % 2 == 0:
                win += 1
            win = min(win, spec.size if spec.size % 2 == 1 else spec.size - 1)
            order = min(self.sg_order, max(1, win - 1))
            if win >= 3:
                spec = savgol_filter(spec, window_length=win, polyorder=order)
            spec = np.maximum(spec, 0.0)
        return freqs, spec

    def _dominant_frequency(self, x: np.ndarray) -> Optional[float]:
        freqs, power = self._power_spectrum(x)
        if power.size < 3:
            return None
        power = power.copy()
        power[0] = 0.0
        peak = float(np.max(power))
        if peak <= 0.0:
            return None
        mask = power >= self.P_th * peak
        if not np.any(mask):
            return None
        idx = int(np.argmax(np.where(mask, power, -np.inf)))
        omega_hat = float(freqs[idx])
        if omega_hat < self.min_omega_hat:
            return None
        return omega_hat

    def fit_transform(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Decompose ``signal``.

        Returns
        -------
        components : np.ndarray
            Shape ``(n_modes, N)``.
        residue, omegas : optional
            When ``return_all=True``.
        """
        x0 = np.asarray(signal, dtype=float).ravel()
        if x0.size < 8:
            raise ValueError("signal length must be at least 8 samples")
        self.signal = x0

        residual = x0.copy()
        buckets: Dict[float, np.ndarray] = {}
        omega_order: List[float] = []
        e0 = float(np.sum(x0**2) + 1e-12)

        for _ in range(self.max_components * 4):
            omega_hat = self._dominant_frequency(residual)
            if omega_hat is None:
                break

            extracted = iterative_swf(
                residual,
                omega_hat=omega_hat,
                std_th=self.StD_th,
                max_sift=self.max_sift,
                refine=self.refine,
            )
            if np.sum(extracted**2) < 1e-10 * e0:
                break

            new_residual, aligned = _align_and_subtract(residual, extracted)
            if np.sum(new_residual**2) >= 0.999 * np.sum(residual**2):
                break
            residual = new_residual

            key = None
            for existing in omega_order:
                if abs(existing - omega_hat) <= self.freq_merge_tol:
                    key = existing
                    break
            if key is None:
                key = omega_hat
                omega_order.append(key)
                buckets[key] = aligned.copy()
            else:
                buckets[key] = buckets[key] + aligned

            if len(omega_order) >= self.max_components:
                break
            if np.sum(residual**2) < (self.P_th**2) * e0:
                break

        if not omega_order:
            components = x0[None, :].copy()
            omegas = np.array([0.0])
            residual = np.zeros_like(x0)
        else:
            components = np.vstack([buckets[w] for w in omega_order])
            omegas = np.asarray(omega_order, dtype=float)

        self.components = components
        self.residue = residual
        self.omegas = omegas

        if return_all:
            return components, residual, omegas
        return components


def swd(
    signal: np.ndarray,
    P_th: float = 0.05,
    StD_th: float = 0.05,
    return_all: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Functional interface to :class:`SWD`."""
    return SWD(P_th=P_th, StD_th=StD_th, **kwargs).fit_transform(
        signal=signal, return_all=return_all
    )


SwD = SWD
SwarmDecomposition = SWD
