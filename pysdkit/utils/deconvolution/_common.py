# -*- coding: utf-8 -*-
"""
Shared helpers for blind-deconvolution period estimation.

The three MATLAB solvers (IMCKD, ACYCBD, SMHD) all estimate a cycle
from the Hilbert envelope of either the raw record or the current
inverse-filter output: an ACF lag after the first zero-crossing
(``TT``) or an envelope harmonic-product spectrum (``EHPS``).

This module is a faithful port of those nested sub-functions.  The
period formula keeps MATLAB's 1-based ``zeroposi + max_position``
quirk (true ACF lag of the peak is ``T - 2``).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import correlate, hilbert
from scipy.stats import kurtosis as scipy_kurtosis


def matlab_round(value: float) -> int:
    """MATLAB ``round`` (ties away from zero)."""
    value = float(value)
    if value >= 0.0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))


def as_real_1d(signal: np.ndarray) -> np.ndarray:
    """Flatten ``signal`` to a real 1-D ``float64`` vector."""
    raw = np.asarray(signal)
    if np.iscomplexobj(raw) and np.max(np.abs(np.imag(raw))) > 0.0:
        raise ValueError("signal must be real")
    samples = np.asarray(raw, dtype=np.float64)
    if samples.ndim == 2 and 1 in samples.shape:
        samples = samples.ravel()
    if samples.ndim != 1:
        raise ValueError("signal must be 1-D")
    return samples.ravel()


def demean(signal: np.ndarray) -> np.ndarray:
    """Subtract the sample mean (MATLAB ``x - mean(x)``)."""
    samples = as_real_1d(signal)
    return samples - np.mean(samples)


def analytic_envelope(signal: np.ndarray) -> np.ndarray:
    """
    Mean-centred Hilbert envelope.

    MATLAB: ``abs(hilbert(x)) - mean(abs(hilbert(x)))``.
    """
    samples = as_real_1d(signal)
    envelope = np.abs(hilbert(samples))
    return envelope - np.mean(envelope)


def matlab_kurtosis(signal: np.ndarray) -> float:
    """
    MATLAB ``kurtosis`` (Pearson, biased).

    Gaussian noise scores near 3, not SciPy's default excess 0.
    """
    samples = as_real_1d(signal)
    if samples.size < 2:
        return float("nan")
    return float(scipy_kurtosis(samples, fisher=False, bias=True, nan_policy="omit"))


def xcorr_coeff(signal: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Positive lags of MATLAB ``xcorr(y, y, M, 'coeff')``.

    Returns length ``max_lag + 1`` (lag 0 through ``M``), matching
    ``NA(ceil(length(NA)/2):end)`` on the full two-sided sequence.
    Extra lags beyond ``N - 1`` are zero.
    """
    samples = as_real_1d(signal)
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")
    energy = float(np.dot(samples, samples))
    n_out = max_lag + 1
    if samples.size == 0 or energy <= 0.0:
        out = np.zeros(n_out, dtype=float)
        if samples.size > 0:
            out[0] = 1.0
        return out
    full = correlate(samples, samples, mode="full", method="fft") / energy
    lag0 = samples.size - 1
    n_avail = min(n_out, full.size - lag0)
    out = np.zeros(n_out, dtype=float)
    out[:n_avail] = np.real(full[lag0 : lag0 + n_avail])
    return out


def first_zero_crossing(acf: np.ndarray) -> int:
    """
    0-based index of MATLAB ``TT``'s first zero-crossing.

    Walks from lag 1 (MATLAB ``lag = 2``) until a ``+`` to ``-``
    sign change or an exact zero.  MATLAB ``zeroposi`` is this
    index plus one.
    """
    values = np.asarray(acf, dtype=float).ravel()
    if values.size < 2:
        raise ValueError("ACF must have at least two lags")
    sample1 = float(values[0])
    for lag in range(1, values.size):
        sample2 = float(values[lag])
        if (sample1 > 0.0 and sample2 < 0.0) or (sample1 == 0.0 or sample2 == 0.0):
            return lag
        sample1 = sample2
    raise ValueError("ACF never crosses zero")


def estimate_period(signal: np.ndarray, fs: float) -> Tuple[int, float]:
    """
    MATLAB nested ``TT``: period (samples) and harmonics-to-noise ratio.

    1. Normalised ACF up to lag ``M = fs``.
    2. Skip the first zero-crossing.
    3. ``T = zeroposi + max_position`` (MATLAB 1-based; equivalent
       0-based formula ``z0 + m0 + 2``).
    4. ``HNR = r_max / (1 - r_max)`` from that peak.

    :return: ``(period, hnr)``
    """
    samples = as_real_1d(signal)
    max_lag = int(fs)
    acf = xcorr_coeff(samples, max_lag)
    zero_pos = first_zero_crossing(acf)
    truncated = acf[zero_pos:]
    if truncated.size == 0:
        raise ValueError("ACF is empty after the first zero-crossing")
    max_index = int(np.argmax(truncated))
    period = zero_pos + max_index + 2
    peak = float(truncated[max_index])
    if peak >= 1.0:
        hnr = float("inf")
    else:
        hnr = peak / (1.0 - peak)
    return period, hnr


def envelope_spectrum(
    signal: np.ndarray,
    fs: float,
    scale: str = "length",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Magnitude FFT of the mean-centred Hilbert envelope.

    ``scale='length'`` matches ACYCBD's ``* 2 / N``;
    ``scale='fs'`` matches IMCKD / SMHD demos ``* 2 / fs``.

    :return: ``(freq_hz, magnitude)`` of length ``N``
    """
    envelope = analytic_envelope(signal)
    n_samples = envelope.size
    magnitude = np.abs(np.fft.fft(envelope, n_samples))
    if scale == "length":
        magnitude = magnitude * 2.0 / float(n_samples)
    elif scale == "fs":
        magnitude = magnitude * 2.0 / float(fs)
    else:
        raise ValueError("scale must be 'length' or 'fs'")
    freq = np.arange(n_samples, dtype=float) * (float(fs) / float(n_samples))
    return freq, magnitude


def ehps(
    signal: np.ndarray,
    fs: float,
    n_harmonics: int = 10,
    flim: float = 300.0,
) -> float:
    """
    Envelope harmonic-product spectrum (MATLAB ``EHPS``).

    Builds ``P(f) = prod_{k=1}^{K} |E(k f)|`` on the envelope
    spectrum (DC dropped) and returns the fundamental
    ``index_max * fs / L`` in hertz.  The loop guard
    ``k * f < fs / 2`` is the MATLAB comparison of a bin index
    against Nyquist in hertz, kept as written.
    """
    samples = demean(signal)
    n_samples = samples.size
    if n_samples < 2:
        raise ValueError("signal is too short for EHPS")
    envelope = analytic_envelope(samples)
    spectrum = np.abs(np.fft.fft(envelope, n_samples)) * 2.0 / float(n_samples)
    n_half = matlab_round(n_samples / 2.0)
    spectrum = spectrum[: max(n_half, 1)]
    harmonics = spectrum[1:]
    n_bins = matlab_round(float(flim) * float(fs) / float(n_samples))
    if n_bins < 1:
        raise ValueError("EHPS frequency grid is empty; increase flim or record length")
    product = np.ones(n_bins, dtype=float)
    nyquist = float(fs) / 2.0
    n_harm = int(n_harmonics)
    for fund in range(1, n_bins + 1):
        value = 1.0
        for order in range(1, n_harm + 1):
            if order * fund < nyquist:
                index = order * fund - 1
                if 0 <= index < harmonics.size:
                    value *= float(harmonics[index])
        product[fund - 1] = value
    index_max = int(np.argmax(product)) + 1
    return float(index_max) * (float(fs) / float(n_samples))


def peak_frequency(
    freq: np.ndarray,
    magnitude: np.ndarray,
    f_max: Optional[float] = None,
) -> float:
    """Frequency of the largest envelope-spectrum bin in ``(0, f_max]``."""
    freq = np.asarray(freq, dtype=float).ravel()
    magnitude = np.asarray(magnitude, dtype=float).ravel()
    mask = freq > 0.0
    if f_max is not None:
        mask &= freq <= float(f_max)
    if not np.any(mask):
        raise ValueError("no bins in the requested frequency range")
    subset = np.where(mask)[0]
    return float(freq[subset[int(np.argmax(magnitude[subset]))]])
