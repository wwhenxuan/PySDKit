# -*- coding: utf-8 -*-
"""
Short-Time Narrow-Banded Mode Decomposition (STNBMD).

McNeill, S.I. Decomposing a signal into short-time narrow-banded modes.
Journal of Sound and Vibration, 373:325–339, 2016.
https://doi.org/10.1016/j.jsv.2016.03.015

Faithful Python port of the MATLAB toolbox routines:
``ps90.m``, ``ps90f.m``, ``stnbm_decomp_ig.m``, ``fft_two2one.m``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.linalg import inv as dense_inv


ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------------
# Hilbert / 90-degree phase-shift helpers (MATLAB ps90.m / ps90f.m)
# ---------------------------------------------------------------------------


def ps90f(x: np.ndarray, n: Optional[int] = None) -> np.ndarray:
    """
    Shift phase of each column by 90 degrees in the frequency domain
    (MATLAB ``ps90f``).

    :param x: data matrix ``(nt, nch)`` (or 1-D vector)
    :param n: FFT length; default ``nt``. Truncate / zero-pad as in MATLAB.
    :return: phase-shifted real array with ``min(nt, n)`` rows
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    nt, nch = x.shape
    if n is None:
        n = nt
    n = int(n)

    if n % 2 == 1:  # odd
        nlast = (n + 1) // 2
        i2 = np.arange(nlast, 1, -1)  # nlast:-1:2  (1-based) -> 0-based nlast-1 .. 1
    else:
        nlast = n // 2 + 1
        i2 = np.arange(nlast - 1, 1, -1)  # nlast-1:-1:2

    tmp = np.fft.fft(x, n=n, axis=0)
    tmp = tmp[:nlast, :].copy()
    tmp2 = np.zeros((n, nch), dtype=np.complex128)
    tmp2[:nlast, :] = -1j * tmp
    # append conjugate spectrum for negative frequencies
    # MATLAB: tmp2(i1,:)=conj(tmp2(i2,:)) with i1 = nlast+1:n
    i1 = np.arange(nlast, n)  # 0-based rows nlast .. n-1
    # i2 is 1-based indices into tmp2 after assigning first nlast rows
    i2_0 = i2 - 1  # convert MATLAB 1-based to 0-based
    tmp2[i1, :] = np.conj(tmp2[i2_0, :])

    y = np.real(np.fft.ifft(tmp2, n=n, axis=0))
    y = y[: min(n, nt), :]
    if squeeze:
        return y[:, 0]
    return y


def ps90(x: np.ndarray) -> np.ndarray:
    """
    90-degree phase shift via mirroring + ``ps90f`` (MATLAB ``ps90``).

    Constructs ``xm = [x; -flipud(x)]``, applies ``ps90f``, then returns the
    first ``nt`` samples.
    """
    x = np.asarray(x, dtype=float)
    squeeze = False
    if x.ndim == 1:
        x = x.reshape(-1, 1)
        squeeze = True
    nt = x.shape[0]
    tmp = -np.flipud(x)
    xm = np.vstack([x, tmp])
    xm90 = ps90f(xm)
    y = xm90[:nt, :]
    if squeeze:
        return y[:, 0]
    return y


def analytic_signal(x: np.ndarray) -> np.ndarray:
    """Form the analytic signal ``x + 1j * ps90(x)`` as in ``stnbm_decomp_ig``."""
    x = np.asarray(x, dtype=float).ravel()
    return x + 1j * ps90(x)


# ---------------------------------------------------------------------------
# Difference operators and smoothing filters
# ---------------------------------------------------------------------------


def first_difference_matrix(nt: int, fs: float) -> sparse.csc_matrix:
    """
    MATLAB ``D1 = fs * spdiags([-tmp, tmp], [0, 1], nt-1, nt)``.

    Shape ``(nt-1, nt)``; ``(D1 x)_i = fs * (x_{i+1} - x_i)``.
    """
    nt = int(nt)
    fs = float(fs)
    # diagonals 0 and 1 of an (nt-1) x nt matrix
    d0 = -fs * np.ones(nt - 1)
    d1 = fs * np.ones(nt - 1)
    return sparse.diags([d0, d1], [0, 1], shape=(nt - 1, nt), format="csc")


def second_difference_matrix(nt: int, fs: float) -> sparse.csc_matrix:
    """
    MATLAB ``D2 = fs^2 * spdiags([tmp, -2*tmp, tmp], [0, 1, 2], nt-2, nt)``.

    Shape ``(nt-2, nt)``; second forward difference scaled by ``fs^2``.
    """
    nt = int(nt)
    fs2 = float(fs) ** 2
    d0 = fs2 * np.ones(nt - 2)
    d1 = -2.0 * fs2 * np.ones(nt - 2)
    d2 = fs2 * np.ones(nt - 2)
    return sparse.diags([d0, d1, d2], [0, 1, 2], shape=(nt - 2, nt), format="csc")


def build_smoothing_filters(
    nt: int,
    fs: float,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Precompute dense filters
    ``F1{ii} = inv(alpha(ii)*Q1 + I)``, ``F2{ii} = inv(beta(ii)*Q2 + I)``
    exactly as in MATLAB ``stnbm_decomp_ig``.
    """
    nt = int(nt)
    alpha = np.asarray(alpha, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    if alpha.size != beta.size:
        raise ValueError("alpha and beta must have the same length")

    d1 = first_difference_matrix(nt, fs)
    q1 = (d1.T @ d1).toarray()
    d2 = second_difference_matrix(nt, fs)
    q2 = (d2.T @ d2).toarray()
    eye = np.eye(nt)

    f1_list: List[np.ndarray] = []
    f2_list: List[np.ndarray] = []
    for a, b in zip(alpha, beta):
        f1_list.append(dense_inv(a * q1 + eye))
        f2_list.append(dense_inv(b * q2 + eye))
    return f1_list, f2_list


def schedule_index(iteration: int, abitr: np.ndarray) -> int:
    """
    MATLAB ``ip = min(find(itr - abitr <= 0))`` (1-based) → 0-based index.

    Selects which ``(alpha, beta)`` pair is active at the current iteration.
    """
    abitr = np.asarray(abitr, dtype=float).ravel()
    tmp = float(iteration) - abitr
    hits = np.where(tmp <= 0)[0]
    if hits.size == 0:
        return int(abitr.size - 1)
    return int(hits[0])


# ---------------------------------------------------------------------------
# Spectral helper (MATLAB fft_two2one.m)
# ---------------------------------------------------------------------------


def fft_two_to_one(
    xf2: np.ndarray,
    fs: float,
    nt: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert two-sided unscaled ``fft`` output to one-sided scaled spectrum
    (MATLAB ``fft_two2one``).

    :param xf2: ``(nf2, nch)`` two-sided FFT
    :param fs: sampling rate [Hz]
    :param nt: original time-series length (default ``nf2``)
    :return: ``(xf1, f1)`` one-sided spectrum and frequency axis
    """
    xf2 = np.asarray(xf2)
    squeeze = False
    if xf2.ndim == 1:
        xf2 = xf2.reshape(-1, 1)
        squeeze = True
    nf2, _nch = xf2.shape
    if nt is None:
        nt = nf2
    nt = int(nt)
    fs = float(fs)

    if nf2 % 2 == 1:
        nlast = (nf2 + 1) // 2
        xf1 = (2.0 / nf2) * np.sqrt(nf2 / nt) * xf2[:nlast, :].copy()
        xf1[0, :] = xf2[0, :] / nf2 * np.sqrt(nf2 / nt)
    else:
        nlast = nf2 // 2 + 1
        xf1 = (2.0 / nf2) * np.sqrt(nf2 / nt) * xf2[:nlast, :].copy()
        scale0 = 1.0 / nf2 * np.sqrt(nf2 / nt)
        xf1[0, :] = xf2[0, :] * scale0
        xf1[nlast - 1, :] = xf2[nlast - 1, :] * scale0

    f1 = np.arange(nlast, dtype=float) * (fs / nf2)
    if squeeze:
        return xf1[:, 0], f1
    return xf1, f1


def instantaneous_frequency(
    phz: np.ndarray, fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate IF from unwrapped phase as in the MATLAB demo:
    ``ifrq = fs/2/pi * diff(phz)``.

    :return: ``(tfrq_midpoints_unitless_diff_index, ifrq_hz)`` — caller usually
             builds time as ``(t[:-1] + t[1:]) / 2``.
    """
    phz = np.asarray(phz, dtype=float)
    if phz.ndim == 1:
        phz = phz.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    ifrq = (float(fs) / (2.0 * np.pi)) * np.diff(phz, axis=0)
    if squeeze:
        return ifrq[:, 0]
    return ifrq


# ---------------------------------------------------------------------------
# Demo signal factory (MATLAB example/test_ord_trk_stnbmd.m)
# ---------------------------------------------------------------------------


def make_order_tracking_demo(
    fs: float = 100.0,
    nt: int = 1000,
) -> Dict[str, np.ndarray]:
    """
    Reproduce the synthetic crossing-order demo from
    ``example/test_ord_trk_stnbmd.m`` (first three upsweep harmonics).
    """
    fs = float(fs)
    nt = int(nt)
    t = np.arange(nt, dtype=float) / fs

    f0, f1 = 1.5, 10.0 / 2.15
    opm = np.log2(f1 / f0) * 60.0 / t[-1]
    fup = f0 * 2.0 ** (opm / 60.0 * t)

    f = 0.75 / t[-1]
    am1 = 0.25 * np.sin(2 * np.pi * f * t - np.pi / 4) + 0.75
    am2 = 0.25 * np.sin(2 * np.pi * f * t - np.pi / 1.7) + 0.75
    am3 = 0.25 * np.sin(2 * np.pi * f * t - np.pi / 8) + 0.75

    x1 = am1 * np.sin(1 * 2 * np.pi * np.cumsum(fup / fs))
    x2 = am2 * np.sin(2 * 2 * np.pi * np.cumsum(fup / fs))
    x3 = am3 * np.sin(3 * 2 * np.pi * np.cumsum(fup / fs))
    x1 = x1 / np.std(x1)
    x2 = x2 / np.std(x2)
    x3 = x3 / np.std(x3)

    fp = np.column_stack([fup, 2 * fup, 3 * fup])
    xp = np.column_stack([x1, x2, x3])
    xe = np.sum(xp, axis=1)

    return {
        "t": t,
        "fs": np.asarray(fs),
        "signal": xe,
        "modes": xp,
        "true_if": fp,
    }


def constant_frequency_init(
    nt: int,
    fs: float,
    frequencies: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unit-amplitude constant-frequency initial guess used in the MATLAB demo:
    ``ampg = 1``, ``phzg(:,ii) = 2*pi*fg(ii)*t``.
    """
    nt = int(nt)
    fs = float(fs)
    freqs = np.asarray(frequencies, dtype=float).ravel()
    t = np.arange(nt, dtype=float) / fs
    nnb = freqs.size
    ampg = np.ones((nt, nnb), dtype=np.complex128)
    phzg = np.zeros((nt, nnb), dtype=float)
    for i, f0 in enumerate(freqs):
        phzg[:, i] = 2.0 * np.pi * f0 * t
    return ampg, phzg


# ---------------------------------------------------------------------------
# Core algorithm (MATLAB stnbm_decomp_ig.m)
# ---------------------------------------------------------------------------


def stnbm_decomp_ig(
    x: np.ndarray,
    fs: float,
    ampg: np.ndarray,
    phzg: np.ndarray,
    alpha: ArrayLike,
    beta: ArrayLike,
    abitr: ArrayLike,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a real signal into short-time narrow-banded analytic modes
    (MATLAB ``stnbm_decomp_ig``).

    :param x: real input vector ``(nt,)``
    :param fs: sampling rate [Hz]
    :param ampg: initial complex amplitudes ``(nt, nnb)``
    :param phzg: initial real phases [rad] ``(nt, nnb)``
    :param alpha: amplitude-smoothness weights (schedule)
    :param beta: frequency-smoothness weights (schedule)
    :param abitr: iteration breakpoints; ``abitr[-1]`` is ``maxitr``
    :param tol: relative change tolerance
    :return: ``(xnb, err, amp, phz)``
    """
    x = np.asarray(x, dtype=float).ravel()
    nt = x.size
    ampg = np.asarray(ampg, dtype=np.complex128)
    phzg = np.asarray(phzg, dtype=float)
    if ampg.ndim != 2 or ampg.shape[0] != nt:
        raise ValueError("ampg must have shape (len(x), nnb)")
    if phzg.shape != ampg.shape:
        raise ValueError("phzg must have the same shape as ampg")

    alpha = np.asarray(alpha, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    abitr = np.asarray(abitr, dtype=float).ravel()
    if not (alpha.shape == beta.shape == abitr.shape):
        raise ValueError("alpha, beta and abitr must have the same shape")
    if alpha.size < 1:
        raise ValueError("alpha/beta/abitr must be non-empty")

    # analytic signal
    x_an = analytic_signal(x)

    phz = phzg.astype(float, copy=True)
    amp = ampg.astype(np.complex128, copy=True)
    xnb = amp * np.exp(1j * phz)
    nnb = ampg.shape[1]

    f1_list, f2_list = build_smoothing_filters(nt, fs, alpha, beta)

    maxitr = int(abitr[-1])
    err = np.zeros(maxitr, dtype=float)
    tol = float(tol)

    for itr in range(1, maxitr + 1):
        ip = schedule_index(itr, abitr)
        f1 = f1_list[ip]
        f2 = f2_list[ip]

        for kk in range(nnb):
            xnbk_old = xnb[:, kk].copy()
            nonk = [i for i in range(nnb) if i != kk]

            # update amplitude
            if nonk:
                xnbk_hat = x_an - np.sum(xnb[:, nonk], axis=1)
            else:
                xnbk_hat = x_an.copy()
            amp[:, kk] = f1 @ (xnbk_hat * np.exp(-1j * phz[:, kk]))

            # update phase
            xnbk_hat = amp[:, kk] * np.exp(1j * phz[:, kk])
            phz[:, kk] = f2 @ np.unwrap(np.angle(xnbk_hat))
            xnb[:, kk] = amp[:, kk] * np.exp(1j * phz[:, kk])

            chngk = xnb[:, kk] - xnbk_old
            denom = np.vdot(xnbk_old, xnbk_old)
            if abs(denom) < np.finfo(float).eps:
                continue
            err[itr - 1] += float(np.real(np.vdot(chngk, chngk) / denom))

        if itr == maxitr or err[itr - 1] < tol:
            err = err[:itr]
            break

    return xnb, err, amp, phz


def stnbmd(
    signal: np.ndarray,
    fs: float,
    frequencies: Optional[ArrayLike] = None,
    ampg: Optional[np.ndarray] = None,
    phzg: Optional[np.ndarray] = None,
    alpha: ArrayLike = (1e-1, 1e-2, 1e-2),
    beta: ArrayLike = (1.0, 1e-1, 1e-3),
    abitr: ArrayLike = (20, 50, 200),
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Functional STNBMD interface.

    Provide either constant-frequency centres ``frequencies`` (demo-style
    initialisation) or explicit ``ampg`` / ``phzg`` guesses.
    """
    x = np.asarray(signal, dtype=float).ravel()
    if ampg is None or phzg is None:
        if frequencies is None:
            raise ValueError("Provide frequencies=... or both ampg and phzg")
        ampg, phzg = constant_frequency_init(x.size, fs, frequencies)
    return stnbm_decomp_ig(
        x, fs, ampg, phzg, alpha=alpha, beta=beta, abitr=abitr, tol=tol
    )


class STNBMD(object):
    """
    Short-Time Narrow-Banded Mode Decomposition.

    McNeill, J. Sound Vib. 373:325–339, 2016.
    """

    def __init__(
        self,
        fs: float = 100.0,
        alpha: ArrayLike = (1e-1, 1e-2, 1e-2),
        beta: ArrayLike = (1.0, 1e-1, 1e-3),
        abitr: ArrayLike = (20, 50, 200),
        tol: float = 1e-6,
    ) -> None:
        self.fs = float(fs)
        self.alpha = np.asarray(alpha, dtype=float).ravel()
        self.beta = np.asarray(beta, dtype=float).ravel()
        self.abitr = np.asarray(abitr, dtype=float).ravel()
        self.tol = float(tol)

        self.signal: Optional[np.ndarray] = None
        self.xnb: Optional[np.ndarray] = None
        self.err: Optional[np.ndarray] = None
        self.amp: Optional[np.ndarray] = None
        self.phz: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return "Short-Time Narrow-Banded Mode Decomposition (STNBMD)"

    def __call__(self, signal: np.ndarray, **kwargs):
        return self.fit_transform(signal, **kwargs)

    def fit_transform(
        self,
        signal: np.ndarray,
        frequencies: Optional[ArrayLike] = None,
        ampg: Optional[np.ndarray] = None,
        phzg: Optional[np.ndarray] = None,
        return_all: bool = False,
    ):
        """
        Run STNBMD.

        :param signal: real 1-D input
        :param frequencies: optional constant IF centres for initialisation
        :param ampg / phzg: optional explicit initial amplitude / phase
        :param return_all: if True, return ``(modes_real, xnb, err, amp, phz)``
        :return: real modes ``(K, nt)`` by default (rows), matching PySDKit style
        """
        xnb, err, amp, phz = stnbmd(
            signal=signal,
            fs=self.fs,
            frequencies=frequencies,
            ampg=ampg,
            phzg=phzg,
            alpha=self.alpha,
            beta=self.beta,
            abitr=self.abitr,
            tol=self.tol,
        )
        self.signal = np.asarray(signal, dtype=float).ravel()
        self.xnb = xnb
        self.err = err
        self.amp = amp
        self.phz = phz

        modes = np.real(xnb).T  # (K, nt)
        if return_all:
            return modes, xnb, err, amp, phz
        return modes

    def instantaneous_frequency_hz(self) -> np.ndarray:
        """IF in Hz with shape ``(nt-1, K)`` from stored phase."""
        if self.phz is None:
            raise ValueError("Call fit_transform before requesting IF.")
        return instantaneous_frequency(self.phz, self.fs)
