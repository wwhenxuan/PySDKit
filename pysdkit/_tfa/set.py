# -*- coding: utf-8 -*-
"""
Synchroextracting Transform (SET).

Faithful Python port of Yu, Yu and Xu's MATLAB ``SET_Y.m`` (IEEE TIE
2017) and the companion ridge extractors ``brevridge.m`` /
``brevridge_mult.m``.

SET is a post-processing of the short-time Fourier transform (STFT).
SST *squeezes* every STFT coefficient onto the instantaneous-frequency
(IF) trajectory; SET instead *extracts* only the coefficient that
already sits on that trajectory and discards the smeared energy.  The
result is closer to the ideal TFA

    ITFA(t, ω) = A(t) · δ(ω − φ'(t))

and, because the retained coefficients are the original STFT values,
a real-valued mode can be read off the ridge (MATLAB ``real(Te(I,t))``).

G. Yu, M. Yu, C. Xu, *Synchroextracting Transform*, IEEE Transactions
on Industrial Electronics 64(10):8042–8054, 2017.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np

_WINDOW_SIGMA = 0.32


def matlab_round(value: float) -> int:
    """MATLAB ``round`` (ties away from zero), used for ``round(N/2)``."""
    value = float(value)
    if value >= 0.0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))


def odd_window_length(hlength: int) -> int:
    """Force an odd Gaussian support (MATLAB ``hlength+1-rem(hlength,2)``)."""
    length = int(hlength)
    if length < 3:
        raise ValueError("hlength must be >= 3")
    return length + 1 - (length % 2)


def gaussian_window(
    hlength: int, sigma: float = _WINDOW_SIGMA
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unit-support Gaussian and its derivative (MATLAB ``SET_Y`` lines 32–38).

    ``ht = linspace(-0.5, 0.5, hlength)``,
    ``h = exp(-π/σ² ht²)``, ``h' = -2π/σ² ht · h``.

    :return: ``(h, dh)`` of odd length ``hlength``.
    """
    n_taps = odd_window_length(hlength)
    ht = np.linspace(-0.5, 0.5, n_taps)
    window = np.exp(-np.pi / (sigma**2) * ht**2)
    deriv = -2.0 * np.pi / (sigma**2) * ht * window
    return window, deriv


def frequency_axis_set(n_samples: int, fs: float = 1.0) -> np.ndarray:
    """
    Frequency axis of the SET / STFT map (cycles in ``fs`` units).

    MATLAB keeps FFT bins ``1:round(N/2)`` (DC through just below or at
    Nyquist).  Bin ``k`` (0-based) is ``k * fs / N`` Hz.
    """
    n_freq = matlab_round(n_samples / 2.0)
    return np.arange(n_freq, dtype=float) * (float(fs) / float(n_samples))


def _as_real_1d(signal: np.ndarray) -> np.ndarray:
    samples = np.asarray(signal, dtype=float)
    if samples.ndim == 2 and 1 in samples.shape:
        samples = samples.ravel()
    if samples.ndim != 1:
        raise ValueError(
            "SET expects a real 1-D record; got shape {}".format(
                np.asarray(signal).shape
            )
        )
    if samples.size < 8:
        raise ValueError("SET requires at least 8 samples")
    return np.ascontiguousarray(samples, dtype=float)


def stft_gaussian_pair(
    x: np.ndarray,
    hlength: int,
    sigma: float = _WINDOW_SIGMA,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    STFT of ``x`` with the SET Gaussian and with its derivative.

    Direct port of the ``tfr1`` / ``tfr2`` construction in ``SET_Y.m``:
    for each time ``ti`` the windowed samples ``x(ti+τ) h(τ)`` are
    placed at circular FFT indices and transformed.

    :return: ``(tfr, tfr_d, window, va)`` where both maps have shape
        ``(round(N/2), N)`` (positive frequencies only) and
        ``va = N / hlength`` is the SET time-scale factor.
    """
    samples = _as_real_1d(x)
    n_samples = int(samples.size)
    n_taps = odd_window_length(hlength)
    window, deriv = gaussian_window(n_taps, sigma=sigma)
    half_len = (n_taps - 1) // 2
    tau_cap = matlab_round(n_samples / 2.0) - 1

    buf = np.zeros((n_samples, n_samples), dtype=np.complex128)
    buf_d = np.zeros((n_samples, n_samples), dtype=np.complex128)
    for time_idx in range(n_samples):
        tau_lo = -min(tau_cap, half_len, time_idx)
        tau_hi = min(tau_cap, half_len, n_samples - 1 - time_idx)
        tau = np.arange(tau_lo, tau_hi + 1)
        fft_idx = np.mod(n_samples + tau, n_samples)
        buf[fft_idx, time_idx] = samples[time_idx + tau] * window[half_len + tau]
        buf_d[fft_idx, time_idx] = samples[time_idx + tau] * deriv[half_len + tau]

    spec = np.fft.fft(buf, axis=0)
    spec_d = np.fft.fft(buf_d, axis=0)
    n_freq = matlab_round(n_samples / 2.0)
    tfr = spec[:n_freq, :]
    tfr_d = spec_d[:n_freq, :]
    scale = float(n_samples) / float(n_taps)
    return tfr, tfr_d, window, scale


def synchroextracting_operator(
    tfr: np.ndarray,
    tfr_d: np.ndarray,
    va: float,
    energy: float,
    amp_thresh: float = 0.8,
    if_tol: float = 0.5,
) -> np.ndarray:
    """
    Binary SET mask (MATLAB ``IF``).

    Keep a TF bin when ``|STFT| > amp_thresh * mean(|x|)`` and the IF
    correction ``|-Re(va i G'/ (2π G))|`` is strictly less than
    ``if_tol`` frequency bins (default half a bin).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (float(va) * 1j * tfr_d) / (2.0 * np.pi * tfr)
    delta = -np.real(ratio)
    finite = np.isfinite(delta)
    strong = np.abs(tfr) > float(amp_thresh) * float(energy)
    return strong & finite & (np.abs(delta) < float(if_tol))


def set_transform(
    x: np.ndarray,
    hlength: Optional[int] = None,
    fs: float = 1.0,
    amp_thresh: float = 0.8,
    if_tol: float = 0.5,
    sigma: float = _WINDOW_SIGMA,
) -> Dict[str, Union[np.ndarray, float, int]]:
    """
    Synchroextracting transform (MATLAB ``SET_Y``).

    :param x: real 1-D record.
    :param hlength: int,
        Gaussian support in samples (made odd).  Default ``round(N/8)``.
    :param fs: float,
        Sampling frequency.
    :param amp_thresh: float,
        Amplitude gate as a multiple of ``mean(|x|)``.  MATLAB uses
        ``0.8``; set ``0`` to keep weak coefficients.
    :param if_tol: float,
        Half-width of the extracting operator in bins (MATLAB ``0.5``).
    :return: dict with ``te`` (SET), ``seo`` (binary operator), ``tfr``
        (amplitude-rectified STFT), ``freq``, ``fs``, ``hlength``.
    """
    samples = _as_real_1d(x)
    n_samples = int(samples.size)
    if hlength is None:
        hlength = matlab_round(n_samples / 8.0)
    n_taps = odd_window_length(hlength)
    tfr_raw, tfr_d, window, va = stft_gaussian_pair(samples, n_taps, sigma=sigma)
    energy = float(np.mean(np.abs(samples)))
    seo = synchroextracting_operator(
        tfr_raw, tfr_d, va, energy, amp_thresh=amp_thresh, if_tol=if_tol
    )
    tfr = tfr_raw / (np.sum(window) / 2.0)
    te = tfr * seo.astype(tfr.dtype)
    freq = frequency_axis_set(n_samples, fs=fs)
    return {
        "te": te,
        "seo": seo.astype(float),
        "tfr": tfr,
        "tfr_raw": tfr_raw,
        "freq": freq,
        "fs": float(fs),
        "hlength": n_taps,
        "va": float(va),
        "x": samples,
    }


def brevridge(
    tx: np.ndarray,
    freq: np.ndarray,
    ridge_lambda: float = 1.0,
    max_jump: int = 10,
    n_starts: int = 40,
) -> Tuple[np.ndarray, float]:
    """
    Greedy ridge of a TF map (MATLAB ``brevridge.m``).

    Several random (uniform) initial times, then a forward/backward
    search that penalises second differences of the frequency index.

    :return: ``(ridge, energy)`` with ``ridge`` a length-``T`` array of
        **0-based** frequency indices.
    """
    energy_map = np.log(
        np.abs(np.asarray(tx, dtype=float)) + np.finfo(float).eps ** 0.25
    )
    n_freq, n_time = energy_map.shape
    if n_time < 3 or n_freq < 1:
        raise ValueError("brevridge needs a TF map with at least 3 time samples")
    freq = np.asarray(freq, dtype=float).ravel()
    if freq.size < 2:
        domega = 1.0
    else:
        domega = float(freq[1] - freq[0])
    aux = float(ridge_lambda) * (domega**2) * 0.5
    max_jump = int(max_jump)
    n_starts = int(n_starts)

    starts = np.floor(
        np.linspace(
            n_time / (n_starts + 1.0), n_time - n_time / (n_starts + 1.0), n_starts
        )
    ).astype(int)
    starts = np.clip(starts, 1, n_time - 2)

    best_ridge = np.zeros(n_time, dtype=int)
    best_energy = -np.inf

    for start in starts:
        curve = np.zeros(n_time, dtype=int)
        idx = int(np.argmax(energy_map[:, start]))
        curve[start] = idx
        energy = float(energy_map[idx, start])
        idx = int(np.argmax(energy_map[:, start - 1]))
        curve[start - 1] = idx
        energy += float(energy_map[idx, start - 1])

        for time_idx in range(start + 1, n_time):
            lo = max(0, idx - max_jump)
            hi = min(n_freq, idx + max_jump + 1)
            penalty = (
                aux
                * (np.arange(lo, hi) - 2 * curve[time_idx - 1] + curve[time_idx - 2])
                ** 2
            )
            score = energy_map[lo:hi, time_idx] - penalty
            offset = int(np.argmax(score))
            idx = lo + offset
            curve[time_idx] = idx
            energy += float(score[offset])

        idx = int(curve[start])
        for time_idx in range(start - 1, -1, -1):
            lo = max(0, idx - max_jump)
            hi = min(n_freq, idx + max_jump + 1)
            if time_idx + 2 < n_time:
                curv = np.arange(lo, hi) - 2 * curve[time_idx + 1] + curve[time_idx + 2]
            else:
                curv = np.arange(lo, hi) - curve[time_idx + 1]
            penalty = aux * curv**2
            score = energy_map[lo:hi, time_idx] - penalty
            offset = int(np.argmax(score))
            idx = lo + offset
            curve[time_idx] = idx
            energy += float(score[offset])

        if energy > best_energy:
            best_energy = energy
            best_ridge = curve.copy()
    return best_ridge, float(best_energy)


def brevridge_mult(
    tx: np.ndarray,
    freq: np.ndarray,
    n_ridges: int,
    ridge_lambda: float = 1.0,
    clear_win: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ``n_ridges`` ridges (MATLAB ``brevridge_mult.m``).

    After each ridge, a frequency window of width ``clear_win`` is
    zeroed so the next search cannot lock onto the same component.

    :return: ``(ridges, energies)`` with ``ridges`` of shape
        ``(n_ridges, T)`` (0-based frequency indices).
    """
    n_ridges = int(n_ridges)
    if n_ridges < 1:
        raise ValueError("n_ridges must be >= 1")
    work = np.array(tx, dtype=float, copy=True)
    n_freq, n_time = work.shape
    ridges = np.zeros((n_ridges, n_time), dtype=int)
    energies = np.zeros(n_ridges, dtype=float)
    half = int(clear_win)
    for ridge_idx in range(n_ridges):
        curve, energy = brevridge(work, freq, ridge_lambda=ridge_lambda)
        ridges[ridge_idx] = curve
        energies[ridge_idx] = energy
        for time_idx in range(n_time):
            lo = max(0, int(curve[time_idx]) - half)
            hi = min(n_freq, int(curve[time_idx]) + half + 1)
            work[lo:hi, time_idx] = 0.0
    return ridges, energies


def reconstruct_from_ridges(te: np.ndarray, ridges: np.ndarray) -> np.ndarray:
    """
    Real SET coefficients along ridges (MATLAB ``real(Te(Cs(k,t), t))``).

    :param te: complex SET map ``(F, T)``.
    :param ridges: int array ``(K, T)`` of 0-based frequency indices.
    :return: real modes ``(K, T)``.
    """
    te = np.asarray(te)
    ridges = np.asarray(ridges, dtype=int)
    if ridges.ndim == 1:
        ridges = ridges.reshape(1, -1)
    n_modes, n_time = ridges.shape
    if te.shape[1] != n_time:
        raise ValueError("ridge length must match the time axis of Te")
    clipped = np.clip(ridges, 0, te.shape[0] - 1)
    time_idx = np.arange(n_time)
    modes = np.empty((n_modes, n_time), dtype=float)
    for mode_idx in range(n_modes):
        modes[mode_idx] = np.real(te[clipped[mode_idx], time_idx])
    return modes


class SET(object):
    """
    Synchroextracting Transform for time-frequency analysis and modes.

    * :meth:`transform` — SET map ``Te(t, ω)`` (and the STFT / SEO).
    * :meth:`fit_transform` — extract ``n_imfs`` ridges of ``|Te|`` and
      reconstruct each mode as the real SET coefficient on that ridge.
      The last row is the residual ``x - sum(modes)``.

    :param hlength: int or None,
        Gaussian window length (odd).  ``None`` uses ``round(N/8)``.
    :param fs: float,
        Sampling frequency (default 1).
    :param amp_thresh: float,
        STFT amplitude gate (MATLAB ``0.8``).
    :param if_tol: float,
        Extracting half-width in bins (MATLAB ``0.5``).
    :param n_imfs: int,
        Number of oscillatory modes for :meth:`fit_transform`.
    :param ridge_lambda: float,
        Smoothness of ``brevridge`` (examples use ``1``).
    :param clear_win: int,
        Frequency clearing window between successive ridges (examples
        use ``5``).
    """

    def __init__(
        self,
        hlength: Optional[int] = None,
        fs: float = 1.0,
        amp_thresh: float = 0.8,
        if_tol: float = 0.5,
        n_imfs: int = 1,
        ridge_lambda: float = 1.0,
        clear_win: int = 5,
    ) -> None:
        if hlength is not None and int(hlength) < 3:
            raise ValueError("hlength must be >= 3")
        if float(fs) <= 0.0:
            raise ValueError("fs must be positive")
        if int(n_imfs) < 1:
            raise ValueError("n_imfs must be >= 1")
        self.hlength = None if hlength is None else int(hlength)
        self.fs = float(fs)
        self.amp_thresh = float(amp_thresh)
        self.if_tol = float(if_tol)
        self.n_imfs = int(n_imfs)
        self.ridge_lambda = float(ridge_lambda)
        self.clear_win = int(clear_win)

        self.signal: Optional[np.ndarray] = None
        self.te_: Optional[np.ndarray] = None
        self.tfr_: Optional[np.ndarray] = None
        self.seo_: Optional[np.ndarray] = None
        self.freq_: Optional[np.ndarray] = None
        self.ridges_: Optional[np.ndarray] = None
        self.imfs: Optional[np.ndarray] = None
        self.residue: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return "Synchroextracting Transform (SET)"

    def __call__(self, signal: np.ndarray, n_imfs: Optional[int] = None) -> np.ndarray:
        return self.fit_transform(signal, n_imfs=n_imfs)

    def _run(self, signal: np.ndarray) -> Dict[str, Union[np.ndarray, float, int]]:
        result = set_transform(
            signal,
            hlength=self.hlength,
            fs=self.fs,
            amp_thresh=self.amp_thresh,
            if_tol=self.if_tol,
        )
        self.signal = np.asarray(result["x"])
        self.te_ = np.asarray(result["te"])
        self.tfr_ = np.asarray(result["tfr"])
        self.seo_ = np.asarray(result["seo"])
        self.freq_ = np.asarray(result["freq"])
        return result

    def transform(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        SET time-frequency map.

        :param signal: real 1-D record.
        :return: ``(te, freq)`` with ``te`` of shape ``(n_freq, T)``.
        """
        result = self._run(signal)
        return np.asarray(result["te"]), np.asarray(result["freq"])

    def fit_transform(
        self,
        signal: np.ndarray,
        n_imfs: Optional[int] = None,
    ) -> np.ndarray:
        """
        Ridge-based modal decomposition of the SET map.

        Each mode is ``real(Te[ridge_k, t])``.  The last row is the
        residual ``x - sum(modes)``.

        :param signal: real 1-D record.
        :param n_imfs: int,
            Number of ridges (defaults to the value given at init).
        :return: IMF array of shape ``(n_imfs + 1, T)``.
        """
        result = self._run(signal)
        n_modes = self.n_imfs if n_imfs is None else int(n_imfs)
        if n_modes < 1:
            raise ValueError("n_imfs must be >= 1")
        te = np.asarray(result["te"])
        freq = np.asarray(result["freq"])
        ridges, _energies = brevridge_mult(
            np.abs(te),
            freq,
            n_modes,
            ridge_lambda=self.ridge_lambda,
            clear_win=self.clear_win,
        )
        modes = reconstruct_from_ridges(te, ridges)
        residual = np.asarray(result["x"]) - np.sum(modes, axis=0)
        imfs = np.vstack([modes, residual[None, :]])
        self.ridges_ = ridges
        self.imfs = imfs
        self.residue = residual
        return imfs


def set(signal: np.ndarray, **kwargs) -> np.ndarray:
    """Functional modal-decomposition interface (``SET(...)(signal)``)."""
    n_imfs = kwargs.pop("n_imfs", 1)
    return SET(n_imfs=n_imfs, **kwargs).fit_transform(signal, n_imfs=n_imfs)
