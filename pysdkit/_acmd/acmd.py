# -*- coding: utf-8 -*-
"""
Adaptive Chirp Mode Decomposition (ACMD)

Faithful port of the MATLAB package by Chen & Peng
(File Exchange 69128), including TF helpers used by Test1 / Test2.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm
from scipy.integrate import cumulative_trapezoid
from scipy.signal import hilbert
from scipy.sparse import csr_matrix, diags, eye as speye, hstack, vstack
from scipy.sparse.linalg import spsolve

from pysdkit._vmd.base import Base


# ---------------------------------------------------------------------------
# Low-level helpers (MATLAB: Differ / SNR / addnoise / STFT / findridges / …)
# ---------------------------------------------------------------------------


def differ(y: np.ndarray, delta: float) -> np.ndarray:
    """Central difference of a 1-D series (MATLAB ``Differ``)."""
    y = np.asarray(y, dtype=float).ravel()
    L = y.size
    if L < 2:
        return np.zeros_like(y)
    mid = np.zeros(max(L - 2, 0), dtype=float)
    for i in range(1, L - 1):
        mid[i - 1] = (y[i + 1] - y[i - 1]) / (2.0 * delta)
    return np.concatenate(
        (
            np.array([(y[1] - y[0]) / delta], dtype=float),
            mid,
            np.array([(y[-1] - y[-2]) / delta], dtype=float),
        )
    )


def compute_snr(clean: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-noise ratio in dB (MATLAB ``SNR``)."""
    clean = np.asarray(clean, dtype=float).ravel()
    estimate = np.asarray(estimate, dtype=float).ravel()
    ps = np.sum((clean - np.mean(clean)) ** 2)
    pn = np.sum((clean - estimate) ** 2)
    if pn <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(ps / pn))


def add_noise(
    n: int,
    mean: float = 0.0,
    std: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Unit-normalized Gaussian noise scaled to ``mean`` / ``std`` (MATLAB ``addnoise``)."""
    rng = np.random.default_rng() if rng is None else rng
    y = rng.standard_normal(n)
    y = y / np.std(y)
    y = y - np.mean(y)
    return mean + std * y


def second_order_difference(n: int):
    """Sparse second-order difference operator ``(n-2) x n`` (MATLAB ``oper``)."""
    e = np.ones(n)
    data = np.vstack((e[:-2], -2.0 * e[1:-1], e[2:]))
    return diags(data, [0, 1, 2], shape=(n - 2, n), format="csc")


def curve_smooth(f: np.ndarray, beta: float) -> np.ndarray:
    """
    Smooth IF curve(s) with a second-order Tikhonov filter (MATLAB ``curvesmooth``).

    ``f`` may be 1-D (one curve) or 2-D with shape ``(K, N)``.
    """
    f = np.asarray(f, dtype=float)
    single = f.ndim == 1
    if single:
        f = f.reshape(1, -1)
    k, n = f.shape
    oper = second_order_difference(n)
    opedoub = (oper.T @ oper).tocsc()
    lhs = (2.0 / max(beta, 1e-30)) * opedoub + speye(n, format="csc")
    out = np.zeros((k, n), dtype=float)
    for i in range(k):
        sol = spsolve(lhs, f[i])
        if not np.all(np.isfinite(sol)):
            sol = np.linalg.lstsq(lhs.toarray(), f[i], rcond=None)[0]
        out[i] = sol
    return out.ravel() if single else out


def find_ridges(spec: np.ndarray, delta: int) -> np.ndarray:
    """
    Time-frequency ridge path (MATLAB ``findridges``).

    :param spec: TF magnitude / complex spectrogram, shape ``(n_freq, n_time)``
    :param delta: max frequency-bin jump between consecutive frames
    :return: frequency-bin indices (1-based MATLAB style → 0-based here), length ``n_time``
    """
    mag = np.abs(np.asarray(spec))
    m, n = mag.shape
    index = np.zeros(n, dtype=int)
    fmax, tmax = np.unravel_index(np.argmax(mag), mag.shape)
    index[tmax] = fmax

    f0 = fmax
    for j in range(min(tmax + 1, n), n):
        low = max(0, f0 - delta)
        up = min(m - 1, f0 + delta)
        f0 = low + int(np.argmax(mag[low : up + 1, j]))
        index[j] = f0

    f1 = fmax
    for j in range(max(0, tmax - 1), -1, -1):
        low = max(0, f1 - delta)
        up = min(m - 1, f1 + delta)
        f1 = low + int(np.argmax(mag[low : up + 1, j]))
        index[j] = f1
    return index


def stft(
    sig: np.ndarray,
    samp_freq: float,
    n_fft: int = 512,
    win_len: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Short-time Fourier transform matching MATLAB ``STFT.m`` in the ACMD package.

    :return: ``(Spec, f)`` with ``Spec`` shape ``(n_fft, n_samples)`` after ``fftshift``
    """
    sig = np.asarray(sig, dtype=float).ravel()
    if np.isrealobj(sig):
        sig = hilbert(sig)
    sig_len = sig.size
    win_len = int(np.ceil(win_len / 2.0) * 2)
    t_win = np.linspace(-1.0, 1.0, win_len)
    sigma = 0.28
    win_fun = (np.pi * sigma**2) ** (-0.25) * np.exp((-(t_win**2)) / (2.0 * sigma**2))
    lh = (win_len - 1) / 2.0

    spec = np.zeros((n_fft, sig_len), dtype=complex)
    half_n = int(np.round(n_fft / 2.0)) - 1
    for i_loop in range(sig_len):
        tau_lo = -min(half_n, int(lh), i_loop)
        tau_hi = min(half_n, int(lh), sig_len - i_loop - 1)
        tau = np.arange(tau_lo, tau_hi + 1)
        temp = i_loop + tau
        temp1 = int(lh) + tau
        r_sig = sig[temp] * np.conj(win_fun[temp1])
        spec[: r_sig.size, i_loop] = r_sig

    spec = np.fft.fftshift(np.fft.fft(spec, axis=0), axes=0)
    f = np.linspace(-samp_freq / 2.0, samp_freq / 2.0, spec.shape[0])
    return spec, f


def tf_spectrum(
    if_multi: np.ndarray,
    ia_multi: np.ndarray,
    band: Tuple[float, float],
    fr_num: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive time-frequency spectrum from IF / IA (MATLAB ``TFspec``).

    :param if_multi: shape ``(n_modes, n_samples)``
    :param ia_multi: shape ``(n_modes, n_samples)``
    :param band: ``(f_min, f_max)``
    """
    if_multi = np.atleast_2d(np.asarray(if_multi, dtype=float))
    ia_multi = np.atleast_2d(np.asarray(ia_multi, dtype=float))
    fbin = np.linspace(band[0], band[1], fr_num)
    num, n = if_multi.shape
    a_spec = np.zeros((fr_num, n), dtype=float)
    delta = int(np.floor(fr_num * 0.001))
    for kk in range(num):
        temp = np.zeros((fr_num, n), dtype=float)
        for ii in range(n):
            index = int(np.argmin(np.abs(fbin - if_multi[kk, ii])))
            lindex = max(index - delta, 0)
            rindex = min(index + delta, fr_num - 1)
            temp[lindex : rindex + 1, ii] = ia_multi[kk, ii]
        a_spec += temp
    return a_spec, fbin


def _spsolve_safe(a, b: np.ndarray) -> np.ndarray:
    """Sparse solve with dense lstsq fallback (MATLAB ``\\`` robustness)."""
    x = spsolve(a, b)
    if not np.all(np.isfinite(x)):
        x = np.linalg.lstsq(
            a.toarray(), np.asarray(b, dtype=float).ravel(), rcond=None
        )[0]
    return np.asarray(x, dtype=float).ravel()


# ---------------------------------------------------------------------------
# ACMD class
# ---------------------------------------------------------------------------


class ACMD(Base):
    """
    Adaptive Chirp Mode Decomposition

    Detection of Rub-Impact Fault for Rotor-Stator Systems: A Novel Method
    Based on Adaptive Chirp Mode Decomposition,
    Chen S, Yang Y, Peng Z, et al., Journal of Sound and Vibration, 2018.

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/69128
    """

    def __init__(
        self,
        K: int,
        fs: Optional[float] = None,
        alpha0: float = 1e-3,
        beta: float = 1e-4,
        tol: float = 1e-8,
        max_iter: int = 300,
    ) -> None:
        """
        :param K: number of modes to extract recursively
        :param fs: sampling frequency; if ``None``, uses ``len(signal)``
        :param alpha0: bandwidth penalty (smaller → narrower filter)
        :param beta: IF-increment smoothness penalty (smaller → smoother)
        :param tol: relative-mode change convergence tolerance
        :param max_iter: maximum iterations per mode
        """
        super().__init__()
        self.K = int(K)
        self.fs = fs
        self.alpha0 = float(alpha0)
        self.beta = float(beta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.imfs_: Optional[np.ndarray] = None
        self.ifs_: Optional[np.ndarray] = None
        self.ias_: Optional[np.ndarray] = None

    def __call__(
        self, signal: np.ndarray, return_all: Optional[bool] = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Allow instances to be called like functions."""
        return self.fit_transform(signal, return_all)

    def __str__(self) -> str:
        return "Adaptive Chirp Mode Decomposition (ACMD)"

    @staticmethod
    def init_IF1(signal: np.ndarray, SampFreq: float, N: int) -> np.ndarray:
        """
        Constant IF from the peak of the (one-sided) Fourier spectrum
        (MATLAB Test1 initialization).
        """
        signal = np.asarray(signal, dtype=float).ravel()
        spec = 2.0 * np.abs(np.fft.fft(signal)) / N
        # MATLAB: Spec = Spec(1:round(end/2)); round half-away-from-zero
        half = int(N / 2.0 + 0.5)
        spec = spec[:half]
        freq_bin = np.linspace(0.0, SampFreq / 2.0, len(spec))
        peak_fre = float(freq_bin[int(np.argmax(spec))])
        return peak_fre * np.ones(N, dtype=float)

    @staticmethod
    def differ(y: np.ndarray, delta: float, dtype: np.dtype = np.float64) -> np.ndarray:
        """Instance-accessible wrapper around module-level ``differ``."""
        out = differ(y, delta)
        return np.asarray(out, dtype=dtype)

    def iter(
        self, signal: np.ndarray, eIF: np.ndarray, N: int, fs: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract one chirp mode (MATLAB ``ACMD.m``).

        IF update: ``eIF ← eIF - ΔIF`` (full vector, not mean-centred BA-ACMD).

        :return: ``(sest, IFest, IAest)``
        """
        signal = np.asarray(signal, dtype=float).ravel()
        e_if = np.asarray(eIF, dtype=float).ravel().copy()
        if signal.size != N or e_if.size != N:
            raise ValueError("signal and eIF must both have length N")

        # MATLAB: t = (0:N-1)/fs
        t = np.arange(N, dtype=float) / float(fs)

        oper = second_order_difference(N)
        spzeros = csr_matrix((N - 2, N), dtype=float)
        opedoub = (oper.T @ oper).tocsc()
        phim = vstack((hstack((oper, spzeros)), hstack((spzeros, oper)))).tocsc()
        phidoubm = (phim.T @ phim).tocsc()

        if_hist = np.zeros((self.max_iter, N), dtype=float)
        s_hist = np.zeros((self.max_iter, N), dtype=float)
        y_hist = np.zeros((self.max_iter, 2 * N), dtype=float)

        n_it = 0
        s_dif = self.tol + 1.0
        alpha = self.alpha0
        ridge = 1e-10
        eye_n = speye(N, format="csc")
        eye_2n = speye(2 * N, format="csc")

        while s_dif > self.tol and n_it < self.max_iter:
            phase = cumulative_trapezoid(e_if, t, initial=0.0)
            cosm = np.cos(2.0 * np.pi * phase)
            sinm = np.sin(2.0 * np.pi * phase)
            cm = diags(cosm, 0, shape=(N, N), format="csc")
            sm = diags(sinm, 0, shape=(N, N), format="csc")
            kerm = hstack((cm, sm))
            kerdoubm = (kerm.T @ kerm).tocsc()

            a_mat = (1.0 / alpha) * phidoubm + kerdoubm + ridge * eye_2n
            ym = _spsolve_safe(a_mat, kerm.T @ signal)
            si = np.asarray(kerm @ ym).ravel()
            s_hist[n_it, :] = si
            y_hist[n_it, :] = ym

            ycm = ym[:N]
            ysm = ym[N:]
            ycm_bar = self.differ(ycm, 1.0 / fs)
            ysm_bar = self.differ(ysm, 1.0 / fs)
            denom = ycm**2 + ysm**2 + np.finfo(float).eps
            delta_if = (ycm * ysm_bar - ysm * ycm_bar) / denom / (2.0 * np.pi)
            smooth = (1.0 / max(self.beta, 1e-30)) * opedoub + eye_n
            delta_if = _spsolve_safe(smooth, delta_if)
            # MATLAB ACMD: eIF = eIF - deltaIF  (full trajectory)
            e_if = e_if - delta_if
            if_hist[n_it, :] = e_if

            if n_it > 0:
                s_dif = (
                    norm(s_hist[n_it] - s_hist[n_it - 1])
                    / (norm(s_hist[n_it - 1]) + np.finfo(float).eps)
                ) ** 2
            n_it += 1

        n_it = max(n_it - 1, 0)
        sest = s_hist[n_it]
        if_est = if_hist[n_it]
        ycm = y_hist[n_it, :N]
        ysm = y_hist[n_it, N:]
        ia_est = np.sqrt(ycm**2 + ysm**2)
        return sest, if_est, ia_est

    def extract_mode(
        self, signal: np.ndarray, init_if: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract a single mode given an initial IF (direct MATLAB ``ACMD`` API).

        :return: ``(sest, IFest, IAest)``
        """
        signal = np.asarray(signal, dtype=float).ravel()
        init_if = np.asarray(init_if, dtype=float).ravel()
        n = signal.size
        fs = float(n if self.fs is None else self.fs)
        return self.iter(signal, init_if, n, fs)

    def fit_transform(
        self, signal: np.ndarray, return_all: Optional[bool] = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Recursively extract ``K`` modes with Fourier-peak IF initialization
        (MATLAB Test1 workflow).
        """
        signal = np.asarray(signal, dtype=float).ravel().copy()
        n = signal.size
        fs = float(n if self.fs is None else self.fs)

        imfs = np.zeros((self.K, n), dtype=float)
        ifs = np.zeros((self.K, n), dtype=float)
        ias = np.zeros((self.K, n), dtype=float)
        residual = signal

        for ii in range(self.K):
            e_if = self.init_IF1(residual, fs, n)
            sest, if_est, ia_est = self.iter(residual, e_if, n, fs)
            imfs[ii] = sest
            ifs[ii] = if_est
            ias[ii] = ia_est
            residual = residual - sest

        self.imfs_ = imfs
        self.ifs_ = ifs
        self.ias_ = ias

        if return_all:
            return imfs, ifs, ias
        return imfs
