# -*- coding: utf-8 -*-
"""
Sparse Maximum Harmonics-to-Noise-Ratio Deconvolution (SMHD).

Aimed at *weak* periodic transients.  An inverse FIR is updated to
raise the harmonics-to-noise ratio (HNR) of the envelope
autocorrelation at lag ``T``, after a Gaussian-like sparsity map

    y ← y · (1 − exp(−y² / (2 μ²)))

has suppressed the dense noise floor.  ``T`` is re-estimated from the
envelope of the sparsified output; ``μ`` is adapted from the change
in kurtosis.

Y. Miao, M. Zhao, J. Lin, Y. Lei, *Sparse maximum harmonics-to-noise-
ratio deconvolution for weak fault signature detection in bearings*,
Measurement Science and Technology 27 (2016) 105004.

Y. Miao, B. Zhang, J. Lin et al., *A review on the application of
blind deconvolution in machinery fault diagnosis*, Mechanical Systems
and Signal Processing 163 (2022) 108202.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import correlate, lfilter

from ._common import (
    analytic_envelope,
    as_real_1d,
    estimate_period,
    matlab_kurtosis,
    matlab_round,
)


def sparse_map(signal: np.ndarray, mu: float) -> np.ndarray:
    """
    MATLAB sparsity map ``y .* (1 - exp(-y.^2 / (2 * mu^2)))``.
    """
    samples = as_real_1d(signal)
    scale = float(mu)
    if scale == 0.0:
        return np.zeros_like(samples)
    return samples * (1.0 - np.exp(-(samples**2) / (2.0 * scale**2)))


def _autocorr_toeplitz(signal: np.ndarray, n_taps: int) -> np.ndarray:
    """Unnormalised autocorr lags ``0 … L-1`` as a Toeplitz matrix."""
    samples = as_real_1d(signal)
    n_samples = samples.size
    full = correlate(samples, samples, mode="full", method="fft")
    lag0 = n_samples - 1
    auto = np.zeros(n_taps, dtype=float)
    n_copy = min(n_taps, n_samples)
    auto[:n_copy] = np.real(full[lag0 : lag0 + n_copy])
    return toeplitz(auto)


def _weighted_cross_corr(
    signal: np.ndarray,
    sparse_output: np.ndarray,
    period: int,
    n_taps: int,
) -> np.ndarray:
    """MATLAB SMHD weighted cross-correlation of length ``L``."""
    samples = as_real_1d(signal)
    sparse = as_real_1d(sparse_output)
    n_samples = samples.size
    lag = int(period)
    delayed_y = np.zeros(n_samples, dtype=float)
    if 0 < lag < n_samples:
        delayed_y[: n_samples - lag] = sparse[lag:]
    elif lag == 0:
        delayed_y = sparse.copy()
    sum_sq = float(np.sum(sparse**2))
    sum_yy = float(np.dot(sparse, delayed_y))
    weights = np.zeros(n_taps, dtype=float)
    if abs(sum_yy) <= 0.0:
        return weights
    n_valid = n_samples - lag if lag < n_samples else 0
    # lag-k: sum y1[k:] * x[:n-k]  and  sum y[k:n-T] * x[T:n-k]
    corr_y1 = correlate(delayed_y, samples, mode="full", method="fft")
    lag0 = n_samples - 1
    n_copy = min(n_taps, n_samples)
    weights[:n_copy] = np.real(corr_y1[lag0 : lag0 + n_copy])
    if n_valid > 0:
        y_head = sparse[:n_valid]
        x_tail = (
            samples[lag : lag + n_valid]
            if lag + n_valid <= n_samples
            else samples[lag:]
        )
        n_pair = min(y_head.size, x_tail.size)
        corr_yt = correlate(y_head[:n_pair], x_tail[:n_pair], mode="full", method="fft")
        lag0_p = n_pair - 1
        n_copy_p = min(n_taps, n_pair)
        weights[:n_copy_p] = weights[:n_copy_p] + np.real(
            corr_yt[lag0_p : lag0_p + n_copy_p]
        )
    weights *= sum_sq / sum_yy
    return weights


def smhd(
    signal: np.ndarray,
    fs: float,
    filter_size: int = 100,
    term_iter: int = 30,
    mu: Optional[float] = None,
    period: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[int, float, np.ndarray]]]:
    """
    Sparse maximum HNR deconvolution.

    Faithful port of MATLAB ``smhd.m``.  The returned ``y`` is the
    *sparsified* trace at the iteration of maximum envelope HNR, not
    the raw FIR output.  ``A_inv = inv(2 A)`` is built once from ``x``
    and is not rebuilt when ``T`` changes.  Loop updates of ``T`` are
    not re-rounded (only the initial estimate is).

    :param signal: 1-D record ``x``.
    :param fs: sampling frequency in hertz.
    :param filter_size: inverse FIR length ``L`` (default 100).
    :param term_iter: number of iterations (default 30; always runs fully).
    :param mu: sparse threshold; ``mean(x)`` when omitted.  The MATLAB
        demo uses ``1.5 * rms(x)``.
    :param period: prior period in samples; estimated from the raw
        envelope when omitted.
    :return: ``(y, fir, info)`` with ``kurt_iter``, ``hnr``, ``period``.
    """
    samples = as_real_1d(signal)
    n_taps = int(filter_size)
    n_iter = int(term_iter)
    if n_taps < 2:
        raise ValueError("filter_size must be >= 2")
    if n_iter < 1:
        raise ValueError("term_iter must be >= 1")

    if period is None:
        lag, _hnr = estimate_period(analytic_envelope(samples), fs)
        period_samples = matlab_round(float(lag))
    else:
        period_samples = matlab_round(float(period))
    if period_samples < 1:
        period_samples = 1

    threshold = float(np.mean(samples)) if mu is None else float(mu)
    if threshold == 0.0:
        threshold = float(np.std(samples)) + 1e-12

    gram = _autocorr_toeplitz(samples, n_taps)
    a_inv = np.linalg.inv(2.0 * gram)

    fir = np.zeros(n_taps, dtype=float)
    centre = matlab_round(n_taps / 2.0)
    fir[centre - 1] = 1.0
    if centre < n_taps:
        fir[centre] = -1.0

    kurt_hist = np.zeros(n_iter, dtype=float)
    hnr_hist = np.zeros(n_iter, dtype=float)
    period_hist = np.zeros(n_iter + 1, dtype=float)
    period_hist[0] = period_samples

    filtered = lfilter(fir, [1.0], samples)
    y_final = sparse_map(filtered, threshold)
    f_final = fir.copy()
    hnr_max = -np.inf

    step = 1
    while step == 1 or step <= n_iter:
        filtered = lfilter(fir, [1.0], samples)
        kurt_hist[step - 1] = matlab_kurtosis(filtered)
        _lag, hnr_hist[step - 1] = estimate_period(analytic_envelope(filtered), fs)
        sparse = sparse_map(filtered, threshold)
        cross = _weighted_cross_corr(samples, sparse, period_samples, n_taps)
        fir = a_inv @ cross
        fir_norm = float(np.sqrt(np.sum(fir**2)))
        if fir_norm > 0.0:
            fir = fir / fir_norm

        step = step + 1
        _tmp_lag, _temp_hnr = estimate_period(sparse, fs)
        kurt_new = matlab_kurtosis(lfilter(fir, [1.0], samples))
        prev_kurt = kurt_hist[step - 2]
        if prev_kurt == 0.0:
            delta_k = 1.0
        else:
            delta_k = kurt_new / prev_kurt
        if delta_k > 1.0:
            delta_k = 1.0 + 0.02 * (delta_k + 1.0) / delta_k
        else:
            delta_k = 1.0 - 0.02 * (delta_k + 1.0) / delta_k
        threshold = threshold * float(delta_k)

        lag, _hnr = estimate_period(analytic_envelope(sparse), fs)
        period_samples = int(lag)
        if period_samples < 1:
            period_samples = 1
        if step - 1 < period_hist.size:
            period_hist[step - 1] = period_samples

        if step == 2:
            hnr_max = float(hnr_hist[0])
            y_final = sparse.copy()
            f_final = fir.copy()
        elif hnr_hist[step - 2] > hnr_max:
            hnr_max = float(hnr_hist[step - 2])
            y_final = sparse.copy()
            f_final = fir.copy()

    info: Dict[str, Union[int, float, np.ndarray]] = {
        "kurt_iter": kurt_hist,
        "hnr": hnr_hist,
        "period_hist": period_hist,
        "period": int(period_samples),
        "mu": float(threshold),
        "hnr_max": float(hnr_max),
        "kurtosis": float(matlab_kurtosis(y_final)),
    }
    return y_final, f_final, info
