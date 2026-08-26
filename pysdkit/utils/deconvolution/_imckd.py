# -*- coding: utf-8 -*-
"""
Improved Maximum Correlated Kurtosis Deconvolution (IMCKD).

Blind deconvolution designs an inverse FIR ``f`` so that ``y = f * x``
recovers a sparse, nearly periodic impact train buried in an unknown
transmission path.  Classical MCKD maximises the *correlated kurtosis*
of ``y`` at a lag ``T`` (the fault period in samples) and therefore
needs that period as a prior.

IMCKD estimates ``T`` from the Hilbert-envelope autocorrelation of the
current output (MATLAB nested ``TT``) at every iteration, rebuilds the
delay tensor, and updates ``f`` until both the filter and the period
settle.

Y. Miao, M. Zhao, J. Lin, Y. Lei, *Application of an improved maximum
correlated kurtosis deconvolution method for fault diagnosis of rolling
element bearings*, Mechanical Systems and Signal Processing 92 (2017)
173–195.

G. L. McDonald, Q. Zhao, M. J. Zuo, *Maximum correlated Kurtosis
deconvolution and application on gear tooth chip fault detection*,
MSSP 33 (2012) 237–255.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np

from ._common import (
    analytic_envelope,
    as_real_1d,
    estimate_period,
    matlab_kurtosis,
    matlab_round,
)


def delay_tensor(
    signal: np.ndarray,
    filter_size: int,
    period: int,
    shift_order: int,
) -> np.ndarray:
    """
    MATLAB ``XmT``: FIR delay lines of ``x`` at lags ``m * T``.

    ``XmT[l, n, m]`` is ``x[n - l - m T]`` when that index is valid
    (0-based), else 0.  Shape ``(L, N, M + 1)``.
    """
    samples = as_real_1d(signal)
    n_samples = samples.size
    n_taps = int(filter_size)
    period = int(period)
    n_shift = int(shift_order)
    if n_taps < 1:
        raise ValueError("filter_size must be >= 1")
    if period < 0:
        raise ValueError("period must be >= 0")
    if n_shift < 0:
        raise ValueError("shift_order must be >= 0")
    tensor = np.zeros((n_taps, n_samples, n_shift + 1), dtype=float)
    for shift in range(n_shift + 1):
        for tap in range(n_taps):
            delay = tap + shift * period
            if 0 <= delay < n_samples:
                tensor[tap, delay:, shift] = samples[: n_samples - delay]
    return tensor


def correlated_kurtosis(output: np.ndarray, period: int, shift_order: int) -> float:
    """
    Correlated kurtosis of shift ``M`` at lag ``T``.

    MATLAB: ``sum(prod(yt, 2).^2) / (sum(y.^2)^(M + 1))``.
    """
    samples = as_real_1d(output)
    delayed = _delay_stack(samples, int(period), int(shift_order))
    energy = float(np.sum(samples**2))
    if energy <= 0.0:
        return 0.0
    product = np.prod(delayed, axis=1)
    return float(np.sum(product**2) / (energy ** (int(shift_order) + 1)))


def _delay_stack(output: np.ndarray, period: int, shift_order: int) -> np.ndarray:
    """Columns of ``y`` delayed by ``0, T, 2T, …`` (MATLAB ``yt``)."""
    n_samples = output.size
    stack = np.zeros((n_samples, shift_order + 1), dtype=float)
    stack[:, 0] = output
    for shift in range(1, shift_order + 1):
        if period > 0:
            stack[period:, shift] = stack[: n_samples - period, shift - 1]
        else:
            stack[:, shift] = stack[:, shift - 1]
    return stack


def imckd(
    signal: np.ndarray,
    fs: float,
    filter_size: int = 100,
    term_iter: int = 30,
    period: Optional[int] = None,
    shift_order: int = 3,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[int, float, np.ndarray]]]:
    """
    Improved MCKD with on-the-fly period estimation.

    Faithful port of MATLAB ``imckd.m``.  The solver is non-interactive
    (no ``plotyy``); iteration traces live in ``info``.

    :param signal: 1-D record ``x``.
    :param fs: sampling frequency in hertz (also the ACF max lag).
    :param filter_size: inverse FIR length ``L`` (default 100).
    :param term_iter: number of iterations (default 30; always runs fully).
    :param period: prior period in samples; estimated from the raw
        envelope when omitted.
    :param shift_order: CK shift order ``M`` (default 3; MATLAB demo uses 1).
    :return: ``(y, fir, info)`` filtered signal, unit-norm FIR, and
        ``ck_iter``, ``period_hist``, ``kurtosis_hist``, ``best_iter``.
    """
    samples = as_real_1d(signal)
    n_taps = int(filter_size)
    n_iter = int(term_iter)
    n_shift = int(shift_order)
    if n_taps < 2:
        raise ValueError("filter_size must be >= 2")
    if n_iter < 1:
        raise ValueError("term_iter must be >= 1")
    if n_shift < 0:
        raise ValueError("shift_order must be >= 0")

    if period is None:
        lag, _hnr = estimate_period(analytic_envelope(samples), fs)
        period_samples = matlab_round(float(lag))
    else:
        period_samples = matlab_round(float(period))
    if period_samples < 1:
        period_samples = 1

    n_samples = samples.size
    tensor = delay_tensor(samples, n_taps, period_samples, n_shift)
    gram = tensor[:, :, 0] @ tensor[:, :, 0].T
    x_inv = np.linalg.inv(gram)

    fir = np.zeros(n_taps, dtype=float)
    fir[1] = 1.0
    output = tensor[:, :, 0].T @ fir
    y_final = output.copy()
    f_final = fir.copy()
    iter_best = 0
    ck_best = 0.0

    ck_hist = np.zeros(n_iter, dtype=float)
    period_hist = np.zeros(n_iter + 1, dtype=float)
    period_hist[0] = period_samples
    kurt_hist = np.zeros(n_iter + 1, dtype=float)
    kurt_hist[0] = matlab_kurtosis(samples)
    kurmax = kurt_hist[0]

    step = 1
    while step == 1 or step <= n_iter:
        output = tensor[:, :, 0].T @ fir
        kurt_hist[step] = matlab_kurtosis(output)
        delayed = _delay_stack(output, period_samples, n_shift)

        alpha = np.zeros((n_samples, n_shift + 1), dtype=float)
        for shift in range(n_shift + 1):
            others = [index for index in range(n_shift + 1) if index != shift]
            if others:
                prod_ex = np.prod(delayed[:, others], axis=1)
            else:
                prod_ex = np.ones(n_samples, dtype=float)
            alpha[:, shift] = (prod_ex**2) * delayed[:, shift]

        beta = np.prod(delayed, axis=1)
        x_alpha = np.zeros(n_taps, dtype=float)
        for shift in range(n_shift + 1):
            x_alpha = x_alpha + tensor[:, :, shift] @ alpha[:, shift]

        energy = float(np.sum(output**2))
        beta_energy = float(np.sum(beta**2))
        if beta_energy > 0.0:
            fir = (energy / (2.0 * beta_energy)) * (x_inv @ x_alpha)
        fir_norm = float(np.sqrt(np.sum(fir**2)))
        if fir_norm > 0.0:
            fir = fir / fir_norm

        ck_hist[step - 1] = correlated_kurtosis(output, period_samples, n_shift)
        if ck_hist[step - 1] > ck_best:
            ck_best = float(ck_hist[step - 1])

        lag, _hnr = estimate_period(analytic_envelope(output), fs)
        period_samples = matlab_round(float(lag))
        if period_samples < 1:
            period_samples = 1
        period_hist[step] = period_samples

        tensor = delay_tensor(samples, n_taps, period_samples, n_shift)
        gram = tensor[:, :, 0] @ tensor[:, :, 0].T
        x_inv = np.linalg.inv(gram)

        step = step + 1
        if step == 2:
            kurmax = kurt_hist[0]
        elif kurt_hist[step - 2] > kurmax:
            kurmax = float(kurt_hist[step - 2])
            y_final = output.copy()
            f_final = fir.copy()
            iter_best = step - 1

    info: Dict[str, Union[int, float, np.ndarray]] = {
        "ck_iter": ck_hist,
        "period_hist": period_hist,
        "kurtosis_hist": kurt_hist,
        "best_iter": int(iter_best),
        "period": int(period_samples),
        "ck_best": float(ck_best),
        "kurtosis": float(matlab_kurtosis(y_final)),
    }
    return y_final, f_final, info
