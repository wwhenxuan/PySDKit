# -*- coding: utf-8 -*-
"""
Adaptive maximum second-order cyclostationarity blind deconvolution
(ACYCBD).

CYCBD maximises the second-order cyclostationarity of ``y = h * x``
at a cyclic frequency ``α`` by solving the generalised eigenproblem
``X W X h = κ X X h``.  Classical CYCBD needs ``α`` from geometry
and RPM.  ACYCBD estimates ``α`` each iteration from the envelope
harmonic-product spectrum (EHPS) of the current output, rebuilds the
periodic weights, and updates ``h`` until the ICS2 criterion settles.

B. Zhang, Y. Miao, J. Lin, Y. Yi, *Adaptive maximum second-order
cyclostationarity blind deconvolution and its application for
locomotive bearing fault diagnosis*, Mechanical Systems and Signal
Processing 158 (2021) 107736.

M. Buzzoni, J. Antoni, G. D'Elia, *Blind deconvolution based on
cyclostationarity maximization and its application to fault
identification*, Journal of Sound and Vibration 432 (2018) 569–601.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.linalg import eigh
from scipy.signal import lfilter

from ._common import as_real_1d, demean, ehps


def corr_matrix(
    signal: np.ndarray,
    weights: Optional[np.ndarray],
    n_taps: int,
) -> np.ndarray:
    """
    MATLAB ``CorrMatrix``: ``N × N`` (weighted) correlation of ``x``.

    Window length is ``L - N + 1``, aligned with the valid FIR output.
    ``weights is None`` uses a vector of ones (unweighted ``R_xx``).
    """
    samples = as_real_1d(signal)
    n_samples = samples.size
    order = int(n_taps)
    if order < 1:
        raise ValueError("n_taps must be >= 1")
    if n_samples < order:
        raise ValueError("signal is shorter than the filter")
    n_win = n_samples - order + 1
    if weights is None:
        weight = np.ones(n_win, dtype=float)
    else:
        weight = as_real_1d(weights)
        if weight.size != n_win:
            raise ValueError(
                "weights must have length L - N + 1, got {} vs {}".format(
                    weight.size, n_win
                )
            )
    delayed = np.empty((order, n_win), dtype=float)
    for tap in range(order):
        delayed[tap] = samples[order - 1 - tap : n_samples - tap]
    gram = (delayed * weight) @ delayed.T / float(n_win)
    return 0.5 * (gram + gram.T)


def periodic(signal: np.ndarray, alpha: np.ndarray, fs: float) -> np.ndarray:
    """
    MATLAB ``Periodic``: Fourier projection onto cyclic frequencies.

    Reconstructs the real series at the harmonics ``alpha``, then
    zeros samples below ``mean + 2 std`` (MATLAB ``std``, ``ddof=1``).
    """
    samples = as_real_1d(signal)
    n_samples = samples.size
    time = np.arange(n_samples, dtype=float) / float(fs)
    freqs = np.asarray(alpha, dtype=float).ravel()
    freqs = freqs[freqs != 0.0]
    if freqs.size == 0:
        projected = np.full(n_samples, np.mean(samples), dtype=float)
    else:
        phase = np.exp(-2j * np.pi * freqs[:, None] * time[None, :])
        coeff = np.mean(samples[None, :] * phase, axis=1)
        projected = np.full(n_samples, np.mean(samples), dtype=float)
        projected = projected + 2.0 * np.real(
            coeff @ np.exp(2j * np.pi * freqs[:, None] * time[None, :])
        )
    if n_samples > 1:
        threshold = float(np.mean(projected) + 2.0 * np.std(projected, ddof=1))
    else:
        threshold = float(np.mean(projected))
    gated = projected.copy()
    gated[gated < threshold] = 0.0
    return gated


def acycbd(
    signal: np.ndarray,
    fs: float,
    filter_size: int = 40,
    relative_error: float = 1e-3,
    max_iter: int = 50,
    order: int = 2,
    n_harmonics: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[int, float, np.ndarray]]]:
    """
    Adaptive CYCBD with EHPS cyclic-frequency estimation.

    Faithful port of MATLAB ``ACYCBD.m``.  The first relative error is
    ``inf`` (``kappa_old = 0``), matching MATLAB's divide-by-zero.

    :param signal: 1-D record ``x``.
    :param fs: sampling frequency in hertz.
    :param filter_size: inverse FIR length ``N`` (default 40).
    :param relative_error: MATLAB ``param.RE`` (default 1e-3).
    :param max_iter: MATLAB ``param.iter`` (default 50).
    :param order: cyclostationarity order ``p`` (default 2).
    :param n_harmonics: EHPS product order ``K`` (default 10).
    :return: ``(fir, s, info)`` with ``kappa``, ``weights``, ``count``,
        ``err``, ``f_est``.  ``s`` has length ``L - N + 1``.
    """
    samples = demean(signal)
    n_taps = int(filter_size)
    n_max = int(max_iter)
    expo = float(order)
    if n_taps < 2:
        raise ValueError("filter_size must be >= 2")
    if n_max < 1:
        raise ValueError("max_iter must be >= 1")
    if expo <= 0.0:
        raise ValueError("order must be > 0")

    n_samples = samples.size
    fir = np.zeros(n_taps, dtype=float)
    fir[1] = 1.0
    xx_gram = corr_matrix(samples, None, n_taps)

    errors = np.zeros(n_max, dtype=float)
    freq_hist = []
    kappa_old = 0.0
    kappa = 0.0
    weights = np.ones(n_samples - n_taps + 1, dtype=float)
    count = 1
    stopped = False

    while not stopped:
        filtered = lfilter(fir, [1.0], samples)
        weights = np.abs(filtered[n_taps - 1 :]) ** expo
        alpha_hat = float(ehps(filtered, fs, n_harmonics=int(n_harmonics)))
        freq_hist.append(alpha_hat)
        harmonics = alpha_hat * np.arange(1, 101, dtype=float)
        weights = periodic(weights, harmonics, fs)
        mean_w = float(np.mean(weights))
        if mean_w > 0.0:
            weights = weights / (mean_w ** (expo / 2.0))
        xwx = corr_matrix(samples, weights, n_taps)
        evals, evecs = eigh(xwx, xx_gram)
        index = int(np.argmax(np.abs(evals)))
        fir = np.real(np.asarray(evecs[:, index], dtype=float)).ravel()
        kappa = float(np.real(evals[index]))
        if kappa_old == 0.0:
            errors[count - 1] = np.inf
        else:
            errors[count - 1] = abs(kappa - kappa_old) / abs(kappa_old)
        if errors[count - 1] < float(relative_error) or count >= n_max:
            stopped = True
        count = count + 1
        kappa_old = kappa

    count = count - 1
    filtered = lfilter(fir, [1.0], samples)
    recovered = filtered[n_taps - 1 :]
    info: Dict[str, Union[int, float, np.ndarray]] = {
        "kappa": float(kappa),
        "weights": weights,
        "count": int(count),
        "err": errors[:count],
        "f_est": np.asarray(freq_hist, dtype=float),
        "period": float(fs / freq_hist[-1]) if freq_hist[-1] != 0.0 else np.inf,
    }
    return fir, recovered, info
