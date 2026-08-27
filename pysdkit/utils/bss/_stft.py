# -*- coding: utf-8 -*-
"""
TFTB short-time Fourier transform used by BSS.

Faithful ports of ``tfrstft.m`` / ``tfristft.m`` from the Time-Frequency
Toolbox, with MATLAB ``hamming(L)`` (symmetric) rather than TFTB
``tftb_window`` Hamming.  Inner loops are vectorised over time; the
circular FFT-bin placement matches ``rem(N+tau,N)+1``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def matlab_round(value: float) -> int:
    """MATLAB ``round`` (ties away from zero)."""
    value = float(value)
    if value >= 0.0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))


def odd_window_length(length: int) -> int:
    """
    Force an odd window length.

    MATLAB BSS default: ``L = floor(T/4); L = L+1-rem(L,2)``.
    """
    length = int(length)
    if length < 3:
        raise ValueError("window length must be >= 3")
    return length + 1 - (length % 2)


def hamming_window(length: int) -> np.ndarray:
    """
    MATLAB symmetric ``hamming(L)``.

    ``0.54 - 0.46 * cos(2 π n / (L-1))``, ``n = 0 … L-1``.
    ``numpy.hamming`` is the same formula.  Length must be odd for TFTB.
    """
    length = int(length)
    if length < 3:
        raise ValueError("hamming window length must be >= 3")
    if length % 2 == 0:
        raise ValueError("H must be a smoothing window with odd length")
    return np.hamming(length).astype(np.float64)


def default_window_length(n_samples: int) -> int:
    """Odd ``floor(T/4)`` used when BSS is called without ``L``."""
    return odd_window_length(int(np.floor(int(n_samples) / 4.0)))


def frequency_axis_stft(n_freq: int, fs: float = 1.0) -> np.ndarray:
    """
    Two-sided STFT frequency axis (cycles in ``fs`` units).

    MATLAB even ``N``: ``[0:N/2-1, -N/2:-1]/N``; odd ``N``:
    ``[0:(N-1)/2, -(N-1)/2:-1]/N``.  Matches ``numpy.fft.fftfreq``.
    """
    return np.fft.fftfreq(int(n_freq), d=1.0 / float(fs))


def _as_column(signal: np.ndarray) -> np.ndarray:
    samples = np.asarray(signal)
    if samples.ndim == 2 and 1 in samples.shape:
        samples = samples.ravel()
    if samples.ndim != 1:
        raise ValueError("X must have one column")
    return np.ascontiguousarray(samples)


def tfrstft(
    signal: np.ndarray,
    times: Optional[np.ndarray] = None,
    n_freq: Optional[int] = None,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Short-time Fourier transform (TFTB ``tfrstft``).

    :param signal: 1-D record (column in MATLAB).
    :param times: time instants.  ``None``, ``0 … T-1`` or MATLAB
        ``1:T`` all run the uniform grid used by BSS.
    :param n_freq: number of FFT bins (default ``T``).
    :param window: odd-length smoothing window (default odd Hamming
        of length ``floor(n_freq/4)``).  Normalised to unit energy.
    :return: complex TFR of shape ``(n_freq, n_times)``.
    """
    samples = _as_column(signal)
    n_samples = int(samples.size)
    if n_freq is None:
        n_freq = n_samples
    n_freq = int(n_freq)
    if n_freq <= 0:
        raise ValueError("N must be greater than zero")
    if window is None:
        window = hamming_window(odd_window_length(int(np.floor(n_freq / 4.0))))
    if times is not None:
        times = np.asarray(times).ravel()
        if times.ndim != 1:
            raise ValueError("T must only have one row")
        if times.size != n_samples:
            raise ValueError("uniform tfrstft expects one column per sample")
    return tfrstft_uniform(samples, window, n_freq=n_freq)


def tfrstft_uniform(
    signal: np.ndarray,
    window: np.ndarray,
    n_freq: Optional[int] = None,
) -> np.ndarray:
    """
    ``tfrstft`` on the full grid ``0 … T-1`` (BSS path).

    Same numbers as MATLAB ``tfrstft(x, 1:T, T, h, 1)``.
    """
    samples = _as_column(signal)
    n_samples = int(samples.size)
    if n_freq is None:
        n_freq = n_samples
    n_freq = int(n_freq)
    window = np.asarray(window, dtype=np.float64).ravel()
    if window.size % 2 == 0:
        raise ValueError("H must be a smoothing window with odd length")
    window = window / np.linalg.norm(window)
    half_len = (int(window.size) - 1) // 2
    tau_cap = matlab_round(n_freq / 2.0) - 1

    tfr = np.zeros((n_freq, n_samples), dtype=np.complex128)
    for tau in range(-half_len, half_len + 1):
        if abs(tau) > tau_cap:
            continue
        freq_idx = tau % n_freq
        tap = np.conj(window[half_len + tau])
        if tau >= 0:
            tfr[freq_idx, : n_samples - tau] = samples[tau:] * tap
        else:
            tfr[freq_idx, -tau:] = samples[: n_samples + tau] * tap
    return np.fft.fft(tfr, axis=0)


def tfristft(
    tfr: np.ndarray,
    times: Optional[np.ndarray] = None,
    window: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Inverse short-time Fourier transform (TFTB ``tfristft``).

    :param tfr: complex TFR ``(n_freq, n_times)``.
    :param times: time instants; default a unit grid of length ``n_times``.
        Must have consecutive unit steps.
    :param window: odd-length window used in the forward transform.
    :return: reconstructed 1-D record of length ``n_times``.
    """
    tfr = np.asarray(tfr)
    if tfr.ndim != 2:
        raise ValueError("tfr must be a 2-D array")
    n_freq, n_times = tfr.shape
    if times is None:
        times = np.arange(n_times, dtype=float)
    times = np.asarray(times, dtype=float).ravel()
    if times.ndim != 1:
        raise ValueError("T must only have one row")
    if times.size != n_times:
        raise ValueError("tfr should have as many columns as t has rows.")
    if n_times >= 2:
        delta = np.diff(times)
        if float(np.min(delta)) != 1.0 or float(np.max(delta)) != 1.0:
            raise ValueError("The tfr must be computed at each time sample.")

    if window is None:
        raise ValueError("At least 3 parameters required")
    window = np.asarray(window, dtype=np.float64).ravel()
    if window.size % 2 == 0:
        raise ValueError("H must be a smoothing window with odd length")
    window = window / np.linalg.norm(window)
    half_len = (int(window.size) - 1) // 2

    tfr_time = np.fft.ifft(tfr, axis=0)
    half_n = n_freq / 2.0
    reconstructed = np.zeros(n_times, dtype=np.complex128)
    weight = np.zeros(n_times, dtype=float)

    for tau in range(-half_len, half_len + 1):
        if abs(tau) > half_n:
            continue
        freq_idx = tau % n_freq
        tap = window[half_len + tau]
        if tau >= 0:
            i_slice = slice(tau, n_times)
            j_slice = slice(0, n_times - tau)
        else:
            i_slice = slice(0, n_times + tau)
            j_slice = slice(-tau, n_times)
        reconstructed[i_slice] += tfr_time[freq_idx, j_slice] * tap
        weight[i_slice] += float(np.abs(tap) ** 2)

    reconstructed = reconstructed / np.maximum(weight, np.finfo(float).tiny)
    if np.max(np.abs(np.imag(reconstructed))) < 1e-8 * max(
        1.0, float(np.max(np.abs(reconstructed)))
    ):
        return np.real(reconstructed)
    return reconstructed


def padding_line(n_samples: int, window: np.ndarray) -> np.ndarray:
    """
    Edge-amplitude correction used after BSS inversion.

    MATLAB::

        tau = -min([round(T/2)-1, Lh, i-1]) : min([round(T/2)-1, Lh, T-i])
        pad(i) = (sum(h) / sum(h(Lh+1+tau))) ^ (1/1.6)

    ``h`` is unit-energy Hamming (same normalisation as the ISTFT).
    Interior samples sit near 1; the ends are boosted.
    """
    n_samples = int(n_samples)
    window = np.asarray(window, dtype=np.float64).ravel()
    if window.size % 2 == 0:
        raise ValueError("H must be a smoothing window with odd length")
    window = window / np.linalg.norm(window)
    half_len = (int(window.size) - 1) // 2
    tau_cap = matlab_round(n_samples / 2.0) - 1
    mass = float(np.sum(window))
    pad = np.empty(n_samples, dtype=np.float64)
    for index in range(n_samples):
        left = min(tau_cap, half_len, index)
        right = min(tau_cap, half_len, n_samples - 1 - index)
        tau = np.arange(-left, right + 1)
        pad[index] = (mass / float(np.sum(window[half_len + tau]))) ** (1.0 / 1.6)
    return pad
