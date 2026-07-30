# -*- coding: utf-8 -*-
"""
Created on 2026/07/30
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Short-Time Variational Mode Decomposition (STVMD).

Jia, H. et al. Short-time variational mode decomposition,
Signal Processing 238 (2026) 110203.
https://doi.org/10.1016/j.sigpro.2025.110203

Reference implementation:
https://github.com/plustar/Short-Time-Variational-Mode-Decomposition
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.fft import irfft, rfft
from scipy.signal import windows as sigwindows

from .base import Base


def _resolve_window(n_fft: int, window: Optional[Union[str, np.ndarray]]) -> np.ndarray:
    """Build a length-``n_fft`` analysis window."""
    if window is None:
        window = "hamming"
    if isinstance(window, str):
        name = window.lower()
        if name == "hamming":
            return sigwindows.hamming(n_fft, sym=False)
        if name == "hann":
            return sigwindows.hann(n_fft, sym=False)
        if name == "dpss":
            return sigwindows.dpss(n_fft, max(4, n_fft // 8), sym=False)
        if name == "boxcar":
            return sigwindows.boxcar(n_fft)
        raise ValueError(
            f"Unknown window '{window}'. "
            "Use 'hamming', 'hann', 'dpss', 'boxcar', or pass an array."
        )
    w = np.asarray(window, dtype=float).ravel()
    if w.size != n_fft:
        raise ValueError(f"window length must equal n_fft ({n_fft}), got {w.size}")
    return w


def _pad_width(n_fft: int) -> Tuple[int, int]:
    """Reflective padding used around the signal before STFT framing."""
    half = (n_fft - 1) // 2
    if (n_fft - 1) % 2 == 0:
        return half, half
    return half + 1, half


def _buffer(x: np.ndarray, n_fft: int, hop_len: int = 1) -> np.ndarray:
    """
    Frame a multi-channel signal into overlapping windows.

    :param x: array of shape ``(C, T)``
    :param n_fft: frame length
    :param hop_len: hop size between consecutive frames
    :return: framed array of shape ``(C, n_fft, n_frames)``
    """
    C, T = x.shape
    if T < n_fft:
        raise ValueError(f"signal length ({T}) must be >= n_fft ({n_fft})")
    n_frames = (T - n_fft) // hop_len + 1
    s0, s1 = x.strides
    framed = as_strided(
        x,
        shape=(C, n_fft, n_frames),
        strides=(s0, s1, s1 * hop_len),
        writeable=False,
    )
    return np.array(framed, copy=True)


def _unbuffer(
    xbuf: np.ndarray, window: np.ndarray, hop_len: int = 1, win_exp: int = 1
) -> np.ndarray:
    """
    Overlap-add framed windows back to a continuous multi-channel signal.

    :param xbuf: framed array ``(C, n_fft, n_frames)``
    :param window: synthesis window of length ``n_fft``
    :param hop_len: hop size used during framing
    :param win_exp: exponent applied to the window before overlap-add
    :return: reconstructed array ``(C, T_ola)``
    """
    if xbuf.ndim == 2:
        xbuf = xbuf[np.newaxis, ...]
    C, n_fft, n_frames = xbuf.shape
    if win_exp == 0:
        w = np.ones(n_fft, dtype=float)
    elif win_exp == 1:
        w = window
    else:
        w = window**win_exp
    T = n_fft + (n_frames - 1) * hop_len
    out = np.zeros((C, T), dtype=xbuf.dtype)
    for i in range(n_frames):
        start = i * hop_len
        out[:, start : start + n_fft] += xbuf[:, :, i] * w
    return out


def _window_norm(
    window: np.ndarray, hop_len: int, n_fft: int, n_frames: int, win_exp: int = 1
) -> np.ndarray:
    """Normalisation weights compensating overlap-add of the analysis window."""
    T = n_fft + (n_frames - 1) * hop_len
    wn = np.zeros(T, dtype=float)
    wpow = window ** (win_exp + 1)
    max_hops = (T - n_fft) // hop_len + 1
    for i in range(max_hops):
        start = i * hop_len
        wn[start : start + n_fft] += wpow
    return wn


class STVMD(Base):
    """
    Short-Time Variational Mode Decomposition.

    STVMD extends VMD / MVMD by replacing the global Fourier transform with a
    short-time Fourier transform (STFT).  Each overlapping time frame is
    decomposed into ``K`` band-limited modes via ADMM.  Two variants are
    available:

    * **non-dynamic** (``dynamic=False``): a single centre frequency per mode,
      shared across all frames (and channels);
    * **dynamic** (``dynamic=True``): centre frequencies may vary across frames,
      which better tracks non-stationary signals.

    Jia et al., Signal Processing, 238:110203, 2026.
    """

    def __init__(
        self,
        K: int = 3,
        alpha: float = 50.0,
        n_fft: int = 64,
        tau: float = 1e-5,
        tol: float = 1e-9,
        max_iter: int = 500,
        dynamic: bool = False,
        window: Optional[Union[str, np.ndarray]] = "hamming",
        win_exp: int = 1,
        init_omega: Optional[np.ndarray] = None,
    ) -> None:
        """
        :param K: number of modes to recover
        :param alpha: balancing / bandwidth parameter of the data-fidelity term
        :param n_fft: STFT frame length (and FFT size)
        :param tau: dual-ascent step size (set near 0 under strong noise)
        :param tol: convergence tolerance on the spectral update
        :param max_iter: maximum number of ADMM iterations
        :param dynamic: if True, centre frequencies vary across frames
        :param window: analysis window name (``'hamming'``, ``'hann'``,
            ``'dpss'``, ``'boxcar'``) or a length-``n_fft`` array
        :param win_exp: exponent applied to the window in overlap-add
        :param init_omega: optional initial centre frequencies of shape ``(K,)``.
            When provided, non-dynamic STVMD keeps them fixed.
        """
        super().__init__()
        if K < 1:
            raise ValueError("K must be >= 1")
        if n_fft < 4:
            raise ValueError("n_fft must be >= 4")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        self.K = int(K)
        self.alpha = float(alpha) * np.ones(self.K, dtype=float)
        self.n_fft = int(n_fft)
        self.hop_len = 1
        self.tau = float(tau)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.dynamic = bool(dynamic)
        self.win_exp = int(win_exp)
        self.window = _resolve_window(self.n_fft, window)
        self.padwidth = _pad_width(self.n_fft)
        self.freqs = np.arange(1, self.n_fft // 2 + 2) / (self.n_fft // 2 + 1)
        self.init_omega = (
            None if init_omega is None else np.asarray(init_omega, dtype=float).ravel()
        )
        if self.init_omega is not None and self.init_omega.size != self.K:
            raise ValueError("init_omega must have length K")

        self.signal: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.u_hat: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None
        self.n_iter: Optional[int] = None
        self._input_was_1d: bool = True

    def __call__(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Allow instances to be called like functions."""
        return self.fit_transform(signal=signal, return_all=return_all)

    def __str__(self) -> str:
        return "Short-Time Variational Mode Decomposition (STVMD)"

    def _prepare(self, signal: np.ndarray) -> np.ndarray:
        """Reflect-pad, window and take the real FFT of each frame."""
        x = np.asarray(signal, dtype=float)
        if x.ndim == 1:
            self._input_was_1d = True
            x = x.reshape(1, -1)
        elif x.ndim == 2:
            self._input_was_1d = False
        else:
            raise ValueError("signal must be 1-D or 2-D with shape (C, T)")

        xp = np.pad(x, ((0, 0), self.padwidth), mode="reflect")
        Sx = _buffer(xp, self.n_fft, self.hop_len)
        Sx *= self.window.reshape(1, -1, 1)
        return rfft(Sx, axis=1)

    def _postprocess(self, u_hat: np.ndarray) -> np.ndarray:
        """
        Invert the STFT of each mode and crop reflective padding.

        :param u_hat: mode spectra ``(C, F, K, N)``
        :return: time-domain modes ``(K, C, T)``
        """
        C, _, K, N = u_hat.shape
        wn = _window_norm(self.window, self.hop_len, self.n_fft, N, self.win_exp)
        # Guard against divide-by-zero at extreme edges.
        wn = np.maximum(wn, np.finfo(float).eps)

        u = np.zeros((K, C, N + np.sum(self.padwidth)), dtype=float)
        for k in range(K):
            xbuf = irfft(u_hat[:, :, k, :], n=self.n_fft, axis=1).real
            u[k] = _unbuffer(xbuf, self.window, self.hop_len, self.win_exp)
        u = u / wn.reshape(1, 1, -1)
        left, right = self.padwidth
        return u[:, :, left : u.shape[-1] - right]

    def _init_omega_vector(self) -> Tuple[np.ndarray, bool]:
        """Return ``(omega0, freeze)`` for non-dynamic STVMD."""
        if self.init_omega is not None:
            return self.init_omega.copy(), True
        omega0 = np.arange(self.K, dtype=float) / float(self.K)
        return omega0, False

    def _admm_nodynamic(self, f_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Non-dynamic STVMD: shared centre frequencies across frames."""
        C, F, N = f_hat.shape
        freqs = self.freqs
        if freqs.size != F:
            freqs = np.arange(1, F + 1, dtype=float) / float(F)

        omega0, freeze = self._init_omega_vector()
        omega = omega0.copy()

        u_hat = np.zeros((C, F, self.K, N), dtype=complex)
        u_hat_old = np.zeros_like(u_hat)
        sum_uk = np.zeros((C, F, N), dtype=complex)
        lambda_hat = np.zeros((C, F, N), dtype=complex)

        n_iter = 0
        for n_iter in range(1, self.max_iter):
            # Mode k = 0 (centre frequency kept at its initial value, usually DC)
            sum_uk = u_hat_old[:, :, self.K - 1, :] + sum_uk - u_hat_old[:, :, 0, :]
            denom = 1.0 + self.alpha[0] * (freqs.reshape(1, -1, 1) - omega[0]) ** 2
            u_hat[:, :, 0, :] = (f_hat - sum_uk - lambda_hat / 2.0) / denom

            for k in range(1, self.K):
                sum_uk = u_hat[:, :, k - 1, :] + sum_uk - u_hat_old[:, :, k, :]
                denom = 1.0 + self.alpha[k] * (freqs.reshape(1, -1, 1) - omega[k]) ** 2
                u_hat[:, :, k, :] = (f_hat - sum_uk - lambda_hat / 2.0) / denom

                if not freeze:
                    power = np.abs(u_hat[:, :, k, :]) ** 2
                    num = np.sum(freqs.reshape(1, -1, 1) * power)
                    den = np.sum(power)
                    omega[k] = num / den if den > 0 else omega[k]

            lambda_hat = lambda_hat + self.tau * (np.sum(u_hat, axis=2) - f_hat)

            # Convergence: max over channels/frames of mean spectral change
            delta = u_hat - u_hat_old
            # (C, N, K) mean over frequency bins
            udiff = np.mean(np.abs(delta) ** 2, axis=1)
            udiff = float(np.max(np.mean(udiff, axis=-1)))
            u_hat_old = u_hat.copy()

            if udiff < self.tol and n_iter > 2:
                break

        order = np.argsort(omega)
        return u_hat[:, :, order, :], omega[order], n_iter

    def _admm_dynamic(self, f_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """Dynamic STVMD: per-frame centre frequencies."""
        C, F, N = f_hat.shape
        freqs = np.arange(1, F + 1, dtype=float) / float(F)

        omega = np.zeros((self.K, N), dtype=float)
        for k in range(self.K):
            omega[k, :] = float(k) / float(self.K)

        u_hat = np.zeros((C, F, self.K, N), dtype=complex)
        u_hat_old = np.zeros_like(u_hat)
        sum_uk = np.zeros((C, F, N), dtype=complex)
        lambda_hat = np.zeros((C, F, N), dtype=complex)

        n_iter = 0
        for n_iter in range(1, self.max_iter):
            sum_uk = u_hat_old[:, :, self.K - 1, :] + sum_uk - u_hat_old[:, :, 0, :]
            denom = (
                1.0
                + self.alpha[0]
                * (freqs.reshape(1, -1, 1) - omega[0, :].reshape(1, 1, -1)) ** 2
            )
            u_hat[:, :, 0, :] = (f_hat - sum_uk - lambda_hat / 2.0) / denom

            for k in range(1, self.K):
                sum_uk = u_hat[:, :, k - 1, :] + sum_uk - u_hat_old[:, :, k, :]
                denom = (
                    1.0
                    + self.alpha[k]
                    * (freqs.reshape(1, -1, 1) - omega[k, :].reshape(1, 1, -1)) ** 2
                )
                u_hat[:, :, k, :] = (f_hat - sum_uk - lambda_hat / 2.0) / denom

                power = np.abs(u_hat[:, :, k, :]) ** 2  # (C, F, N)
                num = np.sum(freqs.reshape(1, -1, 1) * power, axis=(0, 1))
                den = np.sum(power, axis=(0, 1))
                valid = den > 0
                omega[k, valid] = num[valid] / den[valid]

            lambda_hat = lambda_hat + self.tau * (np.sum(u_hat, axis=2) - f_hat)

            delta = u_hat - u_hat_old
            udiff = np.mean(np.abs(delta) ** 2, axis=1)
            udiff = float(np.max(np.mean(udiff, axis=-1)))
            u_hat_old = u_hat.copy()

            if udiff < self.tol and n_iter > 2:
                break

        # Sort modes by centre frequency independently in each frame
        order = np.argsort(omega, axis=0)
        u_sorted = np.empty_like(u_hat)
        omega_sorted = np.empty_like(omega)
        for n in range(N):
            u_sorted[:, :, :, n] = u_hat[:, :, order[:, n], n]
            omega_sorted[:, n] = omega[order[:, n], n]
        return u_sorted, omega_sorted, n_iter

    def fit_transform(
        self,
        signal: np.ndarray,
        return_all: bool = False,
        dynamic: Optional[bool] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Decompose a signal with STVMD.

        :param signal: 1-D array of length ``T``, or 2-D array of shape ``(C, T)``
        :param return_all: if True, also return mode spectra and centre frequencies
        :param dynamic: override the instance-level ``dynamic`` flag
        :return:
            - ``u`` – modes of shape ``(K, T)`` (1-D input) or ``(K, C, T)``
            - optionally ``u_hat`` of shape ``(C, F, K, N_frames)``
            - optionally ``omega`` of shape ``(K,)`` or ``(K, N_frames)``
        """
        use_dynamic = self.dynamic if dynamic is None else bool(dynamic)
        f_hat = self._prepare(signal)

        if use_dynamic:
            u_hat, omega, n_iter = self._admm_dynamic(f_hat)
        else:
            u_hat, omega, n_iter = self._admm_nodynamic(f_hat)

        u = self._postprocess(u_hat)
        if self._input_was_1d:
            u = u[:, 0, :]

        self.signal = np.asarray(signal, dtype=float)
        self.u = u
        self.u_hat = u_hat
        self.omega = omega
        self.n_iter = n_iter

        if return_all:
            return u, u_hat, omega
        return u

    def plot_IMFs(
        self,
        max_imf: int = -1,
        colors: Optional[List] = None,
        save_figure: bool = False,
        return_figure: bool = False,
        dpi: int = 500,
        spine_width: float = 2,
        labelpad: float = 10,
        save_name: Optional[str] = None,
    ):
        """Visualise the last decomposition result (1-D signals only)."""
        from pysdkit.plot import plot_IMFs

        if self.u is None or self.signal is None:
            raise ValueError("Run fit_transform before calling plot_IMFs")
        if self.u.ndim != 2:
            raise ValueError("plot_IMFs supports 1-D (single-channel) results only")

        sig = np.asarray(self.signal, dtype=float).ravel()
        if sig.size != self.u.shape[1]:
            sig = sig[: self.u.shape[1]]

        return plot_IMFs(
            signal=sig,
            IMFs=self.u,
            max_imfs=max_imf,
            colors=colors,
            save_figure=save_figure,
            return_figure=return_figure,
            dpi=dpi,
            spine_width=spine_width,
            labelpad=labelpad,
            save_name=save_name,
        )


def stvmd(
    signal: np.ndarray,
    K: int = 3,
    alpha: float = 50.0,
    n_fft: int = 64,
    tau: float = 1e-5,
    tol: float = 1e-9,
    max_iter: int = 500,
    dynamic: bool = False,
    window: Optional[Union[str, np.ndarray]] = "hamming",
    return_all: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Functional interface to Short-Time Variational Mode Decomposition.

    See :class:`STVMD` for parameter descriptions.
    """
    return STVMD(
        K=K,
        alpha=alpha,
        n_fft=n_fft,
        tau=tau,
        tol=tol,
        max_iter=max_iter,
        dynamic=dynamic,
        window=window,
    ).fit_transform(signal=signal, return_all=return_all)
