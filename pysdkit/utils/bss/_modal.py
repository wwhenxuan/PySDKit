# -*- coding: utf-8 -*-
"""
Modal-shape and SDOF helpers used with BSS.

Ports of MATLAB ``MAC_plot.m``, ``sdof_local.m`` and ``mrsp2mpfd.m``
from Yu's BSS pack (MAC without the blocking ``bar3``).  Examples 2
and 3 sign-correct the absolute mixing matrix with ``corr`` then
identify damped frequency / damping from each separated source.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np


def modal_assurance_criterion(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Modal assurance criterion (MATLAB ``MAC_plot`` without plotting).

    ``MAC[i, j] = (x_i · y_j)^2 / (|x_i|^2 |y_j|^2)`` for columns of
    ``left`` and ``right`` (mode-shape matrices, sensors × modes).
    """
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("mode-shape matrices must be 2-D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("mode-shape matrices must share the sensor axis")
    if x.shape[1] != y.shape[1]:
        raise ValueError("mode-shape matrices must have the same number of modes")
    n_modes = x.shape[1]
    mac = np.empty((n_modes, n_modes), dtype=float)
    for i in range(n_modes):
        for j in range(n_modes):
            num = float(np.dot(x[:, i], y[:, j])) ** 2
            den = float(np.dot(x[:, i], x[:, i]) * np.dot(y[:, j], y[:, j]))
            mac[i, j] = num / den if den != 0.0 else 0.0
    return mac


def sign_from_correlation(sources: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """
    Column signs for the absolute mixing matrix (MATLAB ``corr``).

    ``a1 = corr(source.', X.')`` is ``(K, m)``; ``sign = (a1./abs(a1)).'``
    is ``(m, K)``.  Zero correlations are treated as ``+1``.
    """
    sources = np.asarray(sources, dtype=float)
    observations = np.asarray(observations, dtype=float)
    if sources.ndim != 2 or observations.ndim != 2:
        raise ValueError("sources and observations must be 2-D")
    if sources.shape[1] != observations.shape[1]:
        raise ValueError("sources and observations must share the time axis")
    centred_s = sources - np.mean(sources, axis=1, keepdims=True)
    centred_x = observations - np.mean(observations, axis=1, keepdims=True)
    norm_s = np.linalg.norm(centred_s, axis=1, keepdims=True)
    norm_x = np.linalg.norm(centred_x, axis=1, keepdims=True)
    norm_s = np.maximum(norm_s, np.finfo(float).tiny)
    norm_x = np.maximum(norm_x, np.finfo(float).tiny)
    corr = (centred_s / norm_s) @ (centred_x / norm_x).T
    signs = np.sign(corr.T)
    signs[signs == 0.0] = 1.0
    return signs


def sdof_local(
    spectrum: np.ndarray,
    freq: np.ndarray,
    n_points: int = 7,
) -> Tuple[complex, complex, float, float, int]:
    """
    SDOF local frequency-domain fit (MATLAB ``sdof_local``).

    Peak of ``|h|``, ``n_points`` bins around it (clipped to the grid),
    then the least-squares pole ``h ≈ r / (jω − λ)`` written as
    ``[h, 1] [λ; r] = jω h``.

    :return: ``(lam, residue, fd_hz, damping_percent, n_used)``.
    """
    values = np.asarray(spectrum).ravel()
    freq = np.asarray(freq, dtype=float).ravel()
    if values.size != freq.size:
        raise ValueError("length(f) must = length(h).")
    n_points = int(n_points)
    if n_points > values.size:
        raise ValueError("np must be <= length(h).")
    # Positive frequencies only: for a real free-decay the two-sided
    # FFT is conjugate-symmetric and floating-point can make |H[N-k]|
    # slightly larger than |H[k]|, which would report fs - fd.
    n_pos = values.size // 2 + 1
    peak = int(np.argmax(np.abs(values[:n_pos])))
    half = int(np.floor(n_points / 2.0))
    indices = np.arange(peak - half, peak + half + 1)
    indices = indices[(indices >= 0) & (indices < values.size)]
    n_used = int(indices.size)
    omega = 2.0 * np.pi * freq[indices]
    hp = values[indices]
    design = np.column_stack([hp, np.ones(n_used, dtype=hp.dtype)])
    rhs = 1j * omega * hp
    fitted, *_rest = np.linalg.lstsq(design, rhs, rcond=None)
    lam = complex(fitted[0])
    residue = complex(fitted[1])
    sigma = float(np.real(lam))
    damped_omega = float(np.imag(lam))
    natural = float(np.sqrt(sigma * sigma + damped_omega * damped_omega))
    fd = damped_omega / (2.0 * np.pi)
    damping = 0.0 if natural == 0.0 else -sigma / natural * 100.0
    return lam, residue, fd, damping, n_used


def mrsp2mpfd(
    sources: np.ndarray,
    fs: float,
    n_points: int = 7,
    n_fft: Optional[int] = None,
) -> Dict[str, Union[np.ndarray, int]]:
    """
    Modal parameters of free-decay sources (MATLAB ``mrsp2mpfd``).

    MATLAB takes ``s`` as ``(n_times, n_modes)``.  A ``(K, T)`` BSS
    array is accepted and transposed.

    :return: dict with ``fd``, ``fn``, ``z`` (damping %), ``spectrum``,
        ``freq``, ``sources`` (sorted by increasing ``fd``), ``order``.
    """
    array = np.asarray(sources)
    if array.ndim != 2:
        raise ValueError("s must be a 2-D modal-response matrix")
    if array.shape[0] < array.shape[1]:
        # (K, T) from BSS -> MATLAB (T, K)
        array = array.T
    n_times, n_modes = array.shape
    n_points = 7 if n_points is None else int(n_points)
    n_fft = n_times if n_fft is None else int(n_fft)
    fs = float(fs)
    freq = np.arange(n_fft, dtype=float) * fs / float(n_fft)
    spectrum = np.empty((n_fft, n_modes), dtype=np.complex128)
    fd = np.empty(n_modes, dtype=float)
    fn = np.empty(n_modes, dtype=float)
    damping = np.empty(n_modes, dtype=float)
    for mode in range(n_modes):
        spectrum[:, mode] = np.fft.fft(array[:, mode], n=n_fft) / float(n_times)
        _lam, _r, fd1, z1, _n = sdof_local(spectrum[:, mode], freq, n_points)
        fd[mode] = fd1
        damping[mode] = z1
        fn[mode] = fd1 / np.sqrt(1.0 + (z1 / 100.0) ** 2)
    order = np.argsort(fd)
    return {
        "fd": fd[order],
        "fn": fn[order],
        "z": damping[order],
        "spectrum": spectrum[:, order],
        "freq": freq,
        "sources": array[:, order],
        "order": order,
    }
