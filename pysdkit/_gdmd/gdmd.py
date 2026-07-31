# -*- coding: utf-8 -*-
"""
Created on 2025/07/31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Generalized Dispersion Mode Decomposition (GDMD).

GDMD is a variational algorithm for separating wideband, dispersive /
impulse-like modes.  Working in the *frequency* domain, it models each
mode through a slowly varying group-delay (GD) curve and recovers the
modes by iterative demodulation, analogous to Adaptive Chirp Mode
Decomposition (ACMD) in the time domain.

Chen S, Wang K, Peng Z, et al.
Generalized dispersive mode decomposition: Algorithm and applications.
Journal of Sound and Vibration, 2020.

MATLAB reference:
https://uk.mathworks.com/matlabcentral/fileexchange/81823-generalized-dispersive-mode-decomposition-gdmd?s_tid=FX_rc2_behav
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm
from scipy.integrate import cumulative_trapezoid
from scipy.signal import hilbert
from scipy.sparse import block_diag, diags, eye, lil_matrix
from scipy.sparse.linalg import spsolve


def differ(y: np.ndarray, delta: float) -> np.ndarray:
    """
    Discrete derivative of a 1-D series (central difference interior).

    Matches the MATLAB helper ``Differ.m``.
    """
    y = np.asarray(y, dtype=float).ravel()
    L = y.size
    if L < 2:
        return np.zeros_like(y)
    ybar = np.empty(L, dtype=float)
    ybar[0] = (y[1] - y[0]) / delta
    ybar[-1] = (y[-1] - y[-2]) / delta
    if L > 2:
        ybar[1:-1] = (y[2:] - y[:-2]) / (2.0 * delta)
    return ybar


def curve_smooth(curves: np.ndarray, beta: float) -> np.ndarray:
    """
    Smooth IF / GD curves with second-order difference regularisation.

    Matches MATLAB ``curvesmooth.m``.  Smaller ``beta`` → smoother output.
    """
    f = np.atleast_2d(np.asarray(curves, dtype=float))
    K, N = f.shape
    e = np.ones(N)
    oper = diags([e[:-2], -2.0 * e[1:-1], e[2:]], [0, 1, 2], shape=(N - 2, N))
    opedoub = (oper.T @ oper).tocsc()
    A = (2.0 / beta) * opedoub + eye(N, format="csc")
    out = np.empty_like(f)
    for i in range(K):
        out[i] = spsolve(A, f[i])
    return out


def second_order_difference(N: int):
    """Build the ``(N-2) x N`` second-order difference matrix."""
    e = np.ones(N)
    return diags([e[:-2], -2.0 * e[1:-1], e[2:]], [0, 1, 2], shape=(N - 2, N))


def unilateral_spectrum(signal: np.ndarray) -> np.ndarray:
    """Return the unilateral (non-negative) FFT of a real time signal."""
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    nf = n // 2 + 1
    return np.fft.fft(x)[:nf]


def spectrum_to_time(spectrum: np.ndarray, n_time: int) -> np.ndarray:
    """
    Reconstruct a real time-domain signal from a unilateral spectrum.

    Mirrors the MATLAB pattern::

        full = [S, conj(fliplr(S(2:ceil(Nt/2))))];  ifft(full)
    """
    S = np.asarray(spectrum, dtype=complex).ravel()
    nf = n_time // 2 + 1
    if S.size != nf:
        raise ValueError(
            f"unilateral spectrum length {S.size} incompatible with n_time={n_time} "
            f"(expected {nf})"
        )
    full = np.concatenate([S, np.conj(S[-2:0:-1])])
    return np.real(np.fft.ifft(full))


def make_dispersive_signal(
    samp_freq: float = 100.0,
    duration: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic three-mode dispersive signal from the GDMD paper (Example 1).

    :return: ``(t, signal, f, spectrum, true_gds, true_modes_time)``
        - ``true_gds`` shape ``(3, Nf)``
        - ``true_modes_time`` shape ``(3, Nt)``
        - ``spectrum`` is the sum of unilateral mode spectra
    """
    nt = int(round(samp_freq * duration))
    nf = nt // 2 + 1
    t = np.arange(nt) / samp_freq
    f = np.arange(nf) / duration

    gd1 = -1.0 / 125.0 * f**2 + 2.0 / 5.0 * f + 6.0
    gd2 = 1.0 / 125.0 * f**2 - 2.0 / 5.0 * f + 10.5
    gd3 = -1.0 / 250.0 * f**2 + 12.0
    true_gds = np.vstack([gd1, gd2, gd3])

    ds1 = 1.5 * np.exp(
        -1j * 2 * np.pi * (-1.0 / 375.0 * f**3 + 1.0 / 5.0 * f**2 + 6.0 * f + 0.3)
    )
    ds2 = (1.0 + 0.2 * np.cos(2 * np.pi * 2.0 / 50.0 * f)) * np.exp(
        -1j * 2 * np.pi * (1.0 / 375.0 * f**3 - 1.0 / 5.0 * f**2 + 10.5 * f + 0.5)
    )
    ds3 = (1.0 + 0.2 * np.sin(2 * np.pi * 2.0 / 50.0 * f)) * np.exp(
        -1j * 2 * np.pi * (-1.0 / 750.0 * f**3 + 12.0 * f + 0.8)
    )

    modes_time = np.vstack(
        [
            spectrum_to_time(ds1, nt),
            spectrum_to_time(ds2, nt),
            spectrum_to_time(ds3, nt),
        ]
    )
    signal = np.real(modes_time.sum(axis=0))
    spectrum = ds1 + ds2 + ds3
    return t, signal, f, spectrum, true_gds, modes_time


def tf_spec_from_gd(
    gd_multi: np.ndarray,
    ia_multi: np.ndarray,
    time_range: Tuple[float, float],
    n_time_bins: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a simple TF image from estimated GDs and amplitudes (``TFspec.m``).

    :return: ``(ASpec, t_bins)`` with ``ASpec`` shaped ``(Nf, n_time_bins)``.
    """
    gd = np.atleast_2d(np.asarray(gd_multi, dtype=float))
    ia = np.atleast_2d(np.asarray(ia_multi, dtype=float))
    num, n_freq = gd.shape
    t_bins = np.linspace(time_range[0], time_range[1], n_time_bins)
    a_spec = np.zeros((n_time_bins, n_freq), dtype=float)
    delta = max(1, int(n_time_bins * 0.001))
    for kk in range(num):
        for ii in range(n_freq):
            idx = int(np.argmin(np.abs(t_bins - gd[kk, ii])))
            lo = max(idx - delta, 0)
            hi = min(idx + delta, n_time_bins - 1)
            a_spec[lo : hi + 1, ii] = ia[kk, ii]
    return a_spec.T, t_bins


def gdmd_core(
    spectrum: np.ndarray,
    duration: float,
    init_gd: np.ndarray,
    alpha: float = 1e-3,
    beta: float = 1e-7,
    tol: float = 1e-8,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Core GDMD solver (MATLAB ``GDMD.m``).

    :param spectrum: unilateral FFT spectrum, shape ``(N,)`` (complex)
    :param duration: time duration ``T`` of the original signal
    :param init_gd: initial group delays, shape ``(K, N)``
    :param alpha: bandwidth / roughness trade-off (smaller → narrower band)
    :param beta: GD-increment smoothness (smaller → smoother)
    :param tol: relative convergence tolerance on recovered spectra
    :param max_iter: maximum ADMM / demodulation iterations
    :return: ``(gd_final, modes_freq, gd_history, modes_history)``
        - ``gd_final`` ``(K, N)``
        - ``modes_freq`` ``(K, N)`` complex unilateral spectra
        - histories have shape ``(K, N, n_iters)``
    """
    s = np.asarray(spectrum, dtype=complex).ravel()
    e_gd = np.atleast_2d(np.asarray(init_gd, dtype=float)).copy()
    num, n = e_gd.shape
    if s.size != n:
        raise ValueError(f"spectrum length {s.size} must match init_gd columns {n}")
    if duration <= 0:
        raise ValueError("duration (T) must be positive")

    freq = np.arange(n, dtype=float) / duration
    df = 1.0 / duration

    oper = second_order_difference(n)
    opedoub = (oper.T @ oper).tocsc()
    phim = block_diag([oper] * num).tocsc()
    phidoubm = (phim.T @ phim).tocsc()

    gd_hist: List[np.ndarray] = []
    s_hist: List[np.ndarray] = []

    s_dif = tol + 1.0
    it = 0
    s_prev = None

    while s_dif > tol and it < max_iter:
        # Kernel matrix K: N x (N*num), block-diagonal complex phase kernels
        kerm = lil_matrix((n, n * num), dtype=complex)
        for kk in range(num):
            phase = cumulative_trapezoid(e_gd[kk], freq, initial=0.0)
            kern = np.exp(-1j * 2.0 * np.pi * phase)
            kerm[:, kk * n : (kk + 1) * n] = diags([kern], [0], shape=(n, n))
        kerm = kerm.tocsc()
        kerdoubm = (kerm.T.conj() @ kerm).tocsc()

        # Demodulated signals: (1/alpha Φ'Φ + K'K) y = K' s
        A = (1.0 / alpha) * phidoubm + kerdoubm
        ym_all = spsolve(A, kerm.T.conj() @ s)

        modes_iter = np.empty((num, n), dtype=complex)
        for kk in range(num):
            ym = ym_all[kk * n : (kk + 1) * n]
            delta_phase = np.unwrap(np.angle(ym))
            delta_gd = differ(delta_phase, df) / (2.0 * np.pi)
            delta_gd = spsolve((1.0 / beta) * opedoub + eye(n, format="csc"), delta_gd)
            e_gd[kk] = e_gd[kk] - delta_gd
            modes_iter[kk] = kerm[:, kk * n : (kk + 1) * n] @ ym

        gd_hist.append(e_gd.copy())
        s_hist.append(modes_iter)

        if s_prev is not None:
            s_dif = 0.0
            for kk in range(num):
                den = norm(s_prev[kk])
                if den < 1e-30:
                    continue
                s_dif += (norm(modes_iter[kk] - s_prev[kk]) / den) ** 2
        s_prev = modes_iter
        it += 1

    if it == 0:
        raise RuntimeError("GDMD failed to perform any iteration")

    gd_history = np.stack(gd_hist, axis=-1)
    modes_history = np.stack(s_hist, axis=-1)
    return e_gd, s_hist[-1], gd_history, modes_history


class GDMD(object):
    """
    Generalized Dispersion Mode Decomposition (GDMD).

    Two usage patterns (matching the MATLAB examples):

    1. **Joint multi-mode** — supply initial GD curves ``init_gd`` of shape
       ``(K, Nf)`` and decompose all modes together (Example 1).
    2. **Successive extraction** — set ``K`` and leave ``init_gd=None``; each
       mode is initialised from the envelope peak of the current residual
       (Example 2, impulse / bearing signals).
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        beta: float = 1e-7,
        tol: float = 1e-8,
        max_iter: int = 300,
        K: Optional[int] = None,
    ) -> None:
        """
        :param alpha: filtering bandwidth; larger helps rough GD initialisations
        :param beta: GD-increment smoothness; smaller → smoother / easier converge
        :param tol: convergence tolerance (e.g. 1e-7 … 1e-9)
        :param max_iter: maximum iterations of the demodulation loop
        :param K: number of modes for successive extraction (ignored if
                  ``init_gd`` is provided to ``fit_transform``)
        """
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.K = K

        self.modes_freq_: Optional[np.ndarray] = None
        self.modes_time_: Optional[np.ndarray] = None
        self.group_delays_: Optional[np.ndarray] = None
        self.residual_: Optional[np.ndarray] = None

    def __call__(
        self,
        signal: np.ndarray,
        fs: Optional[float] = None,
        init_gd: Optional[np.ndarray] = None,
        return_all: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        return self.fit_transform(signal, fs=fs, init_gd=init_gd, return_all=return_all)

    def __str__(self) -> str:
        return "Generalized Dispersion Mode Decomposition (GDMD)"

    def fit_transform(
        self,
        signal: np.ndarray,
        fs: Optional[float] = None,
        init_gd: Optional[np.ndarray] = None,
        return_all: bool = False,
        smooth_init_beta: Optional[float] = 1e-7,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Decompose a real time-domain signal with GDMD.

        :param signal: 1-D real signal
        :param fs: sampling frequency (Hz).  If ``None``, ``fs = len(signal)``
                   so that the duration is 1 second (consistent with ACMD).
        :param init_gd: optional initial GDs ``(K, Nf)`` in seconds.  When
                        given, all ``K`` modes are estimated jointly.
        :param return_all: if True, also return GDs and frequency-domain modes
        :param smooth_init_beta: if not ``None``, smooth ``init_gd`` with
                                 :func:`curve_smooth` before the solver
        :return: time-domain modes ``(K, N)``, or
                 ``(modes_time, group_delays, modes_freq)`` when ``return_all``
        """
        x = np.asarray(signal, dtype=float).ravel()
        if x.ndim != 1 or x.size < 8:
            raise ValueError("signal must be a 1-D array with length >= 8")

        n = x.size
        fs_use = float(n if fs is None else fs)
        if fs_use <= 0:
            raise ValueError("fs must be positive")
        duration = n / fs_use
        nf = n // 2 + 1
        spectrum = np.fft.fft(x)[:nf]

        if init_gd is not None:
            gd0 = np.atleast_2d(np.asarray(init_gd, dtype=float))
            if gd0.shape[1] != nf:
                raise ValueError(f"init_gd must have shape (K, {nf}), got {gd0.shape}")
            if smooth_init_beta is not None:
                gd0 = curve_smooth(gd0, smooth_init_beta)
            gd_final, modes_f, _, _ = gdmd_core(
                spectrum,
                duration,
                gd0,
                alpha=self.alpha,
                beta=self.beta,
                tol=self.tol,
                max_iter=self.max_iter,
            )
            modes_t = np.vstack([spectrum_to_time(m, n) for m in modes_f])
            residual = x - modes_t.sum(axis=0)
        else:
            k = self.K
            if k is None or k < 1:
                raise ValueError(
                    "Provide init_gd for joint decomposition, or set K>=1 "
                    "for successive envelope-based extraction"
                )
            modes_t, gd_final, modes_f, residual = self._successive(
                x, spectrum, duration, fs_use, k
            )

        self.modes_time_ = modes_t
        self.modes_freq_ = modes_f
        self.group_delays_ = gd_final
        self.residual_ = residual

        if return_all:
            return modes_t, gd_final, modes_f
        return modes_t

    def _successive(
        self,
        signal: np.ndarray,
        spectrum: np.ndarray,
        duration: float,
        fs: float,
        K: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Example-2 style successive GDMD with envelope-peak GD init."""
        n = signal.size
        nf = spectrum.size
        t = np.arange(n) / fs
        residual_t = signal.copy()
        residual_f = spectrum.copy()

        modes_t = np.zeros((K, n), dtype=float)
        modes_f = np.zeros((K, nf), dtype=complex)
        gds = np.zeros((K, nf), dtype=float)

        for k in range(K):
            envelope = np.abs(hilbert(residual_t))
            peak_t = float(t[int(np.argmax(envelope))])
            init_gd = peak_t * np.ones((1, nf))
            gd_k, mode_f, _, _ = gdmd_core(
                residual_f,
                duration,
                init_gd,
                alpha=self.alpha,
                beta=self.beta,
                tol=self.tol,
                max_iter=self.max_iter,
            )
            mode_t = spectrum_to_time(mode_f[0], n)
            modes_t[k] = mode_t
            modes_f[k] = mode_f[0]
            gds[k] = gd_k[0]
            residual_t = residual_t - mode_t
            residual_f = residual_f - mode_f[0]

        return modes_t, gds, modes_f, residual_t

    def decompose_spectrum(
        self,
        spectrum: np.ndarray,
        duration: float,
        init_gd: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Direct frequency-domain interface matching MATLAB ``GDMD(...)``.

        :return: ``(group_delays, modes_freq)`` final estimates
        """
        gd, modes, _, _ = gdmd_core(
            spectrum,
            duration,
            init_gd,
            alpha=self.alpha,
            beta=self.beta,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        self.group_delays_ = gd
        self.modes_freq_ = modes
        return gd, modes


def gdmd(
    signal: np.ndarray,
    fs: Optional[float] = None,
    init_gd: Optional[np.ndarray] = None,
    alpha: float = 1e-3,
    beta: float = 1e-7,
    tol: float = 1e-8,
    max_iter: int = 300,
    K: Optional[int] = None,
    return_all: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """Functional wrapper around :class:`GDMD`."""
    return GDMD(alpha=alpha, beta=beta, tol=tol, max_iter=max_iter, K=K).fit_transform(
        signal, fs=fs, init_gd=init_gd, return_all=return_all
    )
