# -*- coding: utf-8 -*-
"""
Variational Time-Frequency Mode Tracking Decomposition (VTFMTD).

Dong, H., Shan, T., Yu, G., Shi, Y., Chen, Y.
Variational time-frequency mode tracking for micro-Doppler signature extraction.
Signal Processing, 246:110603, 2026.
https://doi.org/10.1016/j.sigpro.2026.110603
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Packaged demo data
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_complex_npy(name: str) -> np.ndarray:
    path = _DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing VTFMTD demo data: {path}. "
            "Reinstall PySDKit or restore pysdkit/_tfa/data/"
        )
    return np.asarray(np.load(path))


def load_dual_signal_noise() -> Dict[str, Union[np.ndarray, float]]:
    """
    Load the packaged dual-component noisy complex demo
    (MATLAB ``Dual_signal_noise.mat``).

    Sampling rate is ``fs = 3000`` Hz and length is 3000 samples (1 s),
    matching ``test1.m``.
    """
    signal = _load_complex_npy("dual_signal_noise.npy").astype(np.complex128).ravel()
    fs = 3000.0
    t = np.arange(1, signal.size + 1, dtype=float) / fs
    return {"signal": signal, "fs": fs, "t": t, "K": 2}


def load_single_nsignal() -> Dict[str, Union[np.ndarray, float]]:
    """
    Load the packaged single-component noisy micro-Doppler demo
    (MATLAB ``Single_nsignal.mat``).

    Sampling rate is ``fs = 8011`` Hz and length is 8011 samples (1 s),
    matching ``test2.m``.
    """
    signal = _load_complex_npy("single_nsignal.npy").astype(np.complex128).ravel()
    fs = 8011.0
    t = np.arange(signal.size, dtype=float) / fs
    return {"signal": signal, "fs": fs, "t": t, "K": 1}


def load_map2() -> np.ndarray:
    """Load the packaged MATLAB ``map2`` colormap (shape ``(64, 3)``)."""
    return np.asarray(np.load(_DATA_DIR / "map2.npy"), dtype=float)


# ---------------------------------------------------------------------------
# STFT (MATLAB STFT.m)
# ---------------------------------------------------------------------------


def stft(signal: np.ndarray, hlength: Optional[int] = None) -> np.ndarray:
    """
    Short-time Fourier transform used by VTFMTD (MATLAB ``STFT.m``).

    Uses a Gaussian analysis window and hop size 1 (one column per sample).
    Only the positive-frequency half of the FFT is retained.

    :param signal: 1D array (real or complex), length ``N``
    :param hlength: window length; defaults to ``round(N / 8)``. Odd length
        is enforced as in MATLAB (``hlength + 1 - rem(hlength, 2)``).
    :return: STFT matrix of shape ``(F, N)`` with ``F = round(N / 2)``
    """
    x = np.asarray(signal).ravel()
    n = int(x.size)
    if n < 2:
        raise ValueError("signal length must be >= 2")

    if hlength is None:
        hlength = int(round(n / 8.0))
    hlength = int(hlength)
    hlength = hlength + 1 - (hlength % 2)

    ht = np.linspace(-0.5, 0.5, hlength)
    h = np.exp(-np.pi / (0.32**2) * ht**2)
    lh = (h.size - 1) // 2

    f_bins = int(round(n / 2.0))
    tfr = np.zeros((n, n), dtype=np.complex128)

    # Vectorized over time for each lag tau (equivalent to MATLAB STFT.m)
    half_n = n // 2 - 1
    for tau_i, w in enumerate(h):
        tau = int(tau_i - lh)
        if abs(tau) > half_n:
            continue
        row = int(np.mod(n + tau, n))
        # columns ti with valid sample index ti + tau
        ti_lo = max(0, -tau)
        ti_hi = min(n - 1, n - 1 - tau)
        if ti_lo > ti_hi:
            continue
        cols = np.arange(ti_lo, ti_hi + 1)
        tfr[row, cols] = x[cols + tau] * np.conj(w)

    tfr = np.fft.fft(tfr, axis=0)
    tfr = tfr[:f_bins, :]
    return tfr / n * 2.0


def frequency_axis(n: int, fs: float) -> np.ndarray:
    """
    Frequency axis (Hz) matching MATLAB demos:
    ``f = (0 : round(N/2)-1) * fs / N``.
    """
    n = int(n)
    f_bins = int(round(n / 2.0))
    return np.arange(f_bins, dtype=float) * (float(fs) / n)


def bin_index_grid(f_bins: int, n_time: int) -> np.ndarray:
    """
    MATLAB ``omega = repmat((1:F)', 1, T)`` — 1-based STFT bin indices
    broadcast to shape ``(F, T)``.
    """
    return np.tile(np.arange(1, f_bins + 1, dtype=float)[:, None], (1, n_time))


def expand_omega_init(
    omega_init: np.ndarray,
    f_bins: int,
    n_time: int,
    k_modes: int,
) -> np.ndarray:
    """
    Broadcast initial IFs to shape ``(F, T, K)``.

    Accepted forms:
    - ``(F, T, K)`` — used as-is
    - ``(K,)`` or ``(K, 1)`` — constant IF (bin index) per mode
    - ``(T, K)`` or ``(K, T)`` — time-varying IF per mode (no frequency axis)
    """
    w = np.asarray(omega_init, dtype=float)
    if w.shape == (f_bins, n_time, k_modes):
        return w.copy()

    out = np.zeros((f_bins, n_time, k_modes), dtype=float)
    if w.ndim == 1 and w.size == k_modes:
        for k in range(k_modes):
            out[:, :, k] = w[k]
        return out
    if w.shape == (k_modes, 1):
        for k in range(k_modes):
            out[:, :, k] = float(w[k, 0])
        return out
    if w.shape == (n_time, k_modes):
        for k in range(k_modes):
            out[:, :, k] = w[:, k][None, :]
        return out
    if w.shape == (k_modes, n_time):
        for k in range(k_modes):
            out[:, :, k] = w[k, :][None, :]
        return out
    raise ValueError(
        "omega_init must have shape (F,T,K), (K,), (T,K) or (K,T); "
        f"got {w.shape} for F={f_bins}, T={n_time}, K={k_modes}"
    )


def first_difference_gram(n_time: int) -> sparse.csc_matrix:
    """
    Build ``D.T @ D`` for the first-order difference operator in MATLAB
    ``VTFMTD.m`` (``spdiags`` with ``D(1,1)=1``).
    """
    t = int(n_time)
    # D: diagonal 1, sub-diagonal -1
    d0 = np.ones(t)
    d_m1 = -np.ones(t - 1)
    d = sparse.diags([d_m1, d0], [-1, 0], shape=(t, t), format="csc")
    return (d.T @ d).tocsc()


def estimate_if_centroid(
    mode_stft: np.ndarray,
    bin_grid: np.ndarray,
    eps: float = np.finfo(float).eps,
) -> np.ndarray:
    """
    Centroid IF estimate (MATLAB ``omega_est`` update).

    :param mode_stft: complex STFT of one mode, shape ``(F, T)``
    :param bin_grid: frequency-bin indices, shape ``(F, T)``
    :return: IF field shape ``(F, T)`` (constant along frequency)
    """
    power = np.abs(mode_stft) ** 2
    numerator = np.sum(bin_grid * power, axis=0)
    denominator = np.sum(power, axis=0)
    if_traj = np.zeros(denominator.shape, dtype=float)
    ok = denominator >= eps
    if_traj[ok] = numerator[ok] / denominator[ok]
    return np.tile(if_traj[None, :], (mode_stft.shape[0], 1))


def smooth_if(
    omega_est: np.ndarray,
    gram: sparse.csc_matrix,
    beta: float,
) -> np.ndarray:
    """
    Smooth IF trajectories (MATLAB ``omega_smooth`` update).

    Solves ``(2/beta * D'D + I) x = omega_est`` once per mode row pattern.
    Because centroids are constant along frequency, only one solve per
    time-series is required, then broadcast.
    """
    f_bins, n_time = omega_est.shape
    beta = float(beta)
    eye = sparse.eye(n_time, format="csc")
    system = ((2.0 / beta) * gram + eye).tocsc()
    # all frequency rows share the same target trajectory
    target = omega_est[0, :].astype(float)
    smoothed = spsolve(system, target)
    return np.tile(np.asarray(smoothed, dtype=float)[None, :], (f_bins, 1))


def moving_average_if(if_traj: np.ndarray, win: int = 30) -> np.ndarray:
    """
    Post-smoothing used in MATLAB demos (``Lth`` moving average on IF).

    :param if_traj: 1D IF trajectory
    :param win: odd/even window length (``Lth`` in the demos)
    """
    x = np.asarray(if_traj, dtype=float).ravel()
    m = x.size
    half = int(round(win / 2.0))
    out = np.empty(m, dtype=float)
    for i in range(m):
        lo = max(0, i - half)
        hi = min(m, i + half + 1)
        out[i] = np.mean(x[lo:hi])
    return out


def omega_bins_to_hz(omega_bins: np.ndarray, fs: float, n: int) -> np.ndarray:
    """
    Convert MATLAB-style 1-based bin IF values to Hertz.

    STFT row ``j`` (1-based) corresponds to frequency ``(j - 1) * fs / n``.
    """
    return (np.asarray(omega_bins, dtype=float) - 1.0) * (float(fs) / float(n))


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def vtfmtd(
    signal: np.ndarray,
    hlength: int,
    K: int,
    omega_init: np.ndarray,
    alpha: float = 1e-5,
    sigma: float = 1e-2,
    beta: float = 1.0,
    max_iter: int = 100,
    epsilon: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Functional VTFMTD interface (MATLAB ``VTFMTD``).

    :param signal: 1D real/complex signal
    :param hlength: STFT window length
    :param K: number of modes
    :param omega_init: initial IF field / centers (see ``expand_omega_init``)
    :param alpha: bandwidth penalty (larger -> narrower TF support)
    :param sigma: dual-ascent step size
    :param beta: IF smoothness penalty (smaller -> smoother IF)
    :param max_iter: maximum ADMM iterations
    :param epsilon: relative convergence tolerance on mode STFTs
    :return: ``(Gk, omega_smooth)`` with shapes ``(F, T, K)`` and ``(F, T, K)``
    """
    x = np.asarray(signal).ravel()
    g = stft(x, hlength=hlength)
    f_bins, n_time = g.shape
    k_modes = int(K)

    omega_smooth = expand_omega_init(omega_init, f_bins, n_time, k_modes)
    gk = np.zeros((f_bins, n_time, k_modes), dtype=np.complex128)
    lam = np.zeros((f_bins, n_time), dtype=np.complex128)
    bin_grid = bin_index_grid(f_bins, n_time)
    gram = first_difference_gram(n_time)

    alpha = float(alpha)
    sigma = float(sigma)
    beta = float(beta)
    epsilon = float(epsilon)

    for it in range(int(max_iter)):
        gk_prev = gk.copy()

        # ----- update mode STFTs (TF Wiener filtering) -----
        for k in range(k_modes):
            sum_others = np.sum(gk, axis=2) - gk[:, :, k]
            denom = 1.0 + alpha * (bin_grid - omega_smooth[:, :, k]) ** 2
            gk[:, :, k] = (g - sum_others - lam / 2.0) / denom

        # ----- centroid IF + smoothing -----
        for k in range(k_modes):
            omega_est = estimate_if_centroid(gk[:, :, k], bin_grid)
            omega_smooth[:, :, k] = smooth_if(omega_est, gram, beta=beta)

        # ----- dual ascent -----
        lam = lam + sigma * (g - np.sum(gk, axis=2))

        if it > 0:
            diff = np.sum(np.abs(gk - gk_prev))
            norm = np.sum(np.abs(gk_prev)) + np.finfo(float).eps
            if diff < epsilon * norm:
                break

    return gk, omega_smooth


class VTFMTD(object):
    """
    Variational Time-Frequency Mode Tracking Decomposition.

    Decomposes the STFT of a (possibly complex) 1D signal into ``K`` mode
    STFTs while tracking instantaneous-frequency ridges via ADMM, following
    Dong et al., Signal Processing 246:110603, 2026.
    """

    def __init__(
        self,
        hlength: int = 30,
        K: int = 2,
        alpha: float = 1e-5,
        sigma: float = 1e-2,
        beta: float = 1.0,
        max_iter: int = 100,
        epsilon: float = 1e-3,
    ) -> None:
        """
        :param hlength: STFT window length
        :param K: number of TF modes
        :param alpha: TF bandwidth penalty
        :param sigma: Lagrange dual step
        :param beta: IF smoothness penalty
        :param max_iter: maximum iterations
        :param epsilon: convergence tolerance
        """
        self.hlength = int(hlength)
        self.K = int(K)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.max_iter = int(max_iter)
        self.epsilon = float(epsilon)

        self.signal: Optional[np.ndarray] = None
        self.stft_signal: Optional[np.ndarray] = None
        self.Gk: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return "Variational Time-Frequency Mode Tracking Decomposition (VTFMTD)"

    def __call__(
        self,
        signal: np.ndarray,
        omega_init: np.ndarray,
        return_all: bool = False,
    ):
        return self.fit_transform(
            signal=signal, omega_init=omega_init, return_all=return_all
        )

    def fit_transform(
        self,
        signal: np.ndarray,
        omega_init: np.ndarray,
        return_all: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Run VTFMTD.

        :param signal: 1D real/complex input
        :param omega_init: initial IFs (bin units), see ``expand_omega_init``
        :param return_all: if True, also return the composite STFT
        :return: ``(Gk, omega)`` or ``(Gk, omega, G)``
        """
        x = np.asarray(signal).ravel()
        gk, omega = vtfmtd(
            signal=x,
            hlength=self.hlength,
            K=self.K,
            omega_init=omega_init,
            alpha=self.alpha,
            sigma=self.sigma,
            beta=self.beta,
            max_iter=self.max_iter,
            epsilon=self.epsilon,
        )
        g = stft(x, hlength=self.hlength)

        self.signal = x
        self.stft_signal = g
        self.Gk = gk
        self.omega = omega

        if return_all:
            return gk, omega, g
        return gk, omega

    def if_trajectories(self) -> np.ndarray:
        """
        Return IF trajectories with shape ``(K, T)`` (bin units),
        taken from the first frequency row of ``omega`` (MATLAB convention).
        """
        if self.omega is None:
            raise ValueError("Call fit_transform before requesting IF trajectories.")
        return np.asarray(self.omega[0, :, :], dtype=float).T.copy()
