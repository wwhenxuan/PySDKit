# -*- coding: utf-8 -*-
"""
Variational Mode Extraction (VME).

Nazari, M. and Sakhaei, S. M.
Variational Mode Extraction: A New Efficient Method to Derive
Respiratory Signals from ECG.
IEEE Journal of Biomedical and Health Informatics, 22(4):1059-1067, 2018.
https://doi.org/10.1109/JBHI.2017.2734074

Faithful Python port of the MATLAB File Exchange toolbox ``vme.m``
(https://www.mathworks.com/matlabcentral/fileexchange/76003).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from pysdkit.utils import fft, fftshift, ifft, ifftshift

_DATA_DIR = Path(__file__).resolve().parent / "data"
_EPS = float(np.finfo(np.float64).eps)


def load_vme_ecg_055m() -> Dict[str, Union[np.ndarray, float]]:
    """
    Load the packaged MIMIC record ``055m`` shipped with MATLAB VME.

    The File Exchange archive stores ``val`` as ``int16`` of shape
    ``(7, 7500)``.  Following ``VME_test_script.m``, channel 0 is the ECG
    and the last channel is the simultaneous reference respiration.
    Sampling rate is 125 Hz (paper: 4000 samples = 32 s).

    :return: dict with ``val``, ``ecg``, ``respiration``, ``fs``, ``t``
    """
    path = _DATA_DIR / "ecg_055m.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing VME demo data: {path}. "
            "Reinstall PySDKit or restore pysdkit/_vmd/data/"
        )
    val = np.asarray(np.load(path), dtype=np.float64)
    if val.ndim != 2 or val.shape[0] < 1:
        raise ValueError("ecg_055m.npy must have shape (n_channels, n_samples)")
    fs = 125.0
    t = np.arange(val.shape[1], dtype=float) / fs
    return {
        "val": val,
        "ecg": val[0].copy(),
        "respiration": val[-1].copy(),
        "fs": fs,
        "t": t,
    }


def generate_vme_example1(
    n_samples: int = 1000, fs: float = 1000.0
) -> Dict[str, Union[np.ndarray, float, str]]:
    """MATLAB ``VME_test_script.m`` Example 1 (paper Eq. (23))."""
    t = np.arange(1, n_samples + 1, dtype=float) / float(n_samples)
    c1 = 1.0 / (1.2 + np.cos(2.0 * np.pi * t))
    c2 = 1.0 / (1.5 + np.sin(2.0 * np.pi * t))
    c3 = np.cos(32.0 * np.pi * t + 0.2 * np.cos(64.0 * np.pi * t))
    signal = c1 + c2 * c3
    return {
        "t": t,
        "signal": signal,
        "reference": c1,
        "omega_init": 0.0,
        "fs": float(fs),
        "name": "Example 1",
    }


def generate_vme_example2(
    n_samples: int = 1000, fs: float = 1000.0
) -> Dict[str, Union[np.ndarray, float, str]]:
    """MATLAB ``VME_test_script.m`` Example 2 (paper Eq. (24))."""
    t = np.arange(1, n_samples + 1, dtype=float) / float(n_samples)
    c1 = 2.0 * np.cos(4.0 * np.pi * t)
    c2 = np.cos(30.0 * np.pi * t) * (1.0 + np.cos(2.0 * np.pi * t)) / 2.0
    c3 = np.cos(80.0 * np.pi * t) * (1.0 + np.sin(2.0 * np.pi * t)) / 2.0
    signal = c1 + c2 + c3
    return {
        "t": t,
        "signal": signal,
        "reference": c2,
        "omega_init": 10.0,
        "fs": float(fs),
        "name": "Example 2",
    }


def generate_vme_example3a(
    n_samples: int = 1000, fs: float = 1000.0
) -> Dict[str, Union[np.ndarray, float, str]]:
    """MATLAB ``VME_test_script.m`` Example 3a (paper Eq. (25), chirp)."""
    t = np.arange(1, n_samples + 1, dtype=float) / float(n_samples)
    chirp = 2.0 * np.cos(10.0 * np.pi * t + 10.0 * np.pi * t**2)
    high = np.cos(60.0 * np.pi * t)
    high[n_samples // 2 :] = 0.0
    jump = np.cos(100.0 * np.pi * t - 10.0 * np.pi)
    jump[: n_samples // 2] = 0.0
    signal = chirp + high + jump
    return {
        "t": t,
        "signal": signal,
        "reference": chirp,
        "omega_init": 6.0,
        "fs": float(fs),
        "name": "Example 3a",
    }


def generate_vme_example3b(
    n_samples: int = 1000, fs: float = 1000.0
) -> Dict[str, Union[np.ndarray, float, str]]:
    """MATLAB ``VME_test_script.m`` Example 3b (second piecewise tone)."""
    t = np.arange(1, n_samples + 1, dtype=float) / float(n_samples)
    chirp = 2.0 * np.cos(10.0 * np.pi * t + 10.0 * np.pi * t**2)
    high = np.cos(60.0 * np.pi * t)
    high[n_samples // 2 :] = 0.0
    jump = np.cos(100.0 * np.pi * t - 10.0 * np.pi)
    jump[: n_samples // 2] = 0.0
    signal = chirp + high + jump
    return {
        "t": t,
        "signal": signal,
        "reference": high,
        "omega_init": 26.0,
        "fs": float(fs),
        "name": "Example 3b",
    }


def ensure_even_length(signal: np.ndarray) -> np.ndarray:
    """Drop the last sample of an odd-length vector (MATLAB ``T/2`` indexing)."""
    x = np.asarray(signal, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("signal must contain at least 2 samples")
    if x.size % 2 == 1:
        x = x[:-1]
    return x


def mirror_extend(signal: np.ndarray) -> np.ndarray:
    """
    Mirror-extend a 1-D signal as in MATLAB ``vme.m``.

    With even length ``T``::

        f_mir = [signal(T/2:-1:1), signal, signal(T:-1:T/2+1)]

    The result has length ``2 T``.
    """
    x = ensure_even_length(signal)
    half = x.size // 2
    return np.concatenate([x[:half][::-1], x, x[half:][::-1]])


def crop_mirror(extended: np.ndarray) -> np.ndarray:
    """Undo :func:`mirror_extend` (MATLAB ``u_d(:, T/4+1:3*T/4)``)."""
    y = np.asarray(extended).ravel()
    t = y.size
    return y[t // 4 : 3 * t // 4]


def spectral_axis(n_fft: int) -> np.ndarray:
    """Normalised frequency axis after ``fftshift`` (MATLAB ``omega_axis``)."""
    n_fft = int(n_fft)
    t = np.arange(1, n_fft + 1, dtype=float) / n_fft
    return t - 0.5 - 1.0 / n_fft


def onesided_fft(extended: np.ndarray) -> np.ndarray:
    """``fftshift(fft(f))`` with negative frequencies zeroed (Hilbert / VMD)."""
    f_hat = fftshift(fft(np.asarray(extended, dtype=float).ravel()))
    onesided = np.asarray(f_hat, dtype=np.complex128).copy()
    onesided[: onesided.size // 2] = 0.0
    return onesided


def compactness_kernel(
    omega_axis: np.ndarray, omega_d: float, alpha: float
) -> np.ndarray:
    """``alpha^2 * (omega - omega_d)^4`` used in the MATLAB mode / residual updates."""
    dw = np.asarray(omega_axis, dtype=float) - float(omega_d)
    return (float(alpha) ** 2) * dw**4


def update_mode_spectrum(
    f_hat_onesided: np.ndarray,
    u_hat: np.ndarray,
    dual: np.ndarray,
    omega_axis: np.ndarray,
    omega_d: float,
    alpha: float,
) -> np.ndarray:
    """One ADMM step for ``u_hat_d`` (MATLAB ``vme.m`` main loop)."""
    kernel = compactness_kernel(omega_axis, omega_d, alpha)
    numerator = f_hat_onesided + np.asarray(u_hat) * kernel + np.asarray(dual) / 2.0
    denominator = (1.0 + kernel) * (1.0 + 2.0 * kernel)
    return numerator / denominator


def residual_spectrum(
    f_hat_onesided: np.ndarray,
    u_hat: np.ndarray,
    omega_axis: np.ndarray,
    omega_d: float,
    alpha: float,
) -> np.ndarray:
    """Filtered residual ``F_r`` used inside the MATLAB dual-ascent update."""
    kernel = compactness_kernel(omega_axis, omega_d, alpha)
    return (kernel * (f_hat_onesided - np.asarray(u_hat))) / (1.0 + 2.0 * kernel)


def update_dual(
    dual: np.ndarray,
    f_hat_onesided: np.ndarray,
    u_hat: np.ndarray,
    omega_axis: np.ndarray,
    omega_d: float,
    alpha: float,
    tau: float,
) -> np.ndarray:
    """Dual ascent ``lambda <- lambda + tau * (f - (u_d + F_r))``."""
    residual = residual_spectrum(f_hat_onesided, u_hat, omega_axis, omega_d, alpha)
    return np.asarray(dual) + float(tau) * (
        f_hat_onesided - (np.asarray(u_hat) + residual)
    )


def update_center_frequency(
    u_hat: np.ndarray, omega_axis: np.ndarray, previous: float
) -> float:
    """Positive-frequency centroid of ``|u_hat|^2`` (paper Eq. (18))."""
    u_hat = np.asarray(u_hat)
    omega_axis = np.asarray(omega_axis, dtype=float)
    half = omega_axis.size // 2
    power = np.abs(u_hat[half:]) ** 2
    total = float(np.sum(power))
    if total <= _EPS:
        return float(previous)
    return float(np.dot(omega_axis[half:], power) / total)


def relative_spectrum_change(current: np.ndarray, previous: np.ndarray) -> float:
    """MATLAB loop criterion ``(1/T) * (u_n - u_{n-1}) * conj(...).'``."""
    delta = np.asarray(current) - np.asarray(previous)
    t = float(delta.size)
    return float(np.abs(_EPS + np.vdot(delta, delta) / t))


def reconstruct_hermitian(u_hat_onesided: np.ndarray) -> np.ndarray:
    """
    Build a two-sided Hermitian spectrum from the one-sided iterate.

    Port of MATLAB::

        u_hatd(T/2+1:T) = u_hat_d(N, T/2+1:T)
        u_hatd(T/2+1:-1:2) = conj(u_hat_d(N, T/2+1:T))
        u_hatd(1) = conj(u_hatd(end))
    """
    pos = np.asarray(u_hat_onesided, dtype=np.complex128).ravel()
    t = pos.size
    half = t // 2
    u_hat = np.zeros(t, dtype=np.complex128)

    # MATLAB u_hatd(T/2+1:T) = u_hat_d(N, T/2+1:T)
    u_hat[half:] = pos[half:]

    # MATLAB u_hatd(T/2+1:-1:2) = conj(u_hat_d(N, T/2+1:T))
    # 0-based destinations: half, half-1, ..., 1 (overwrites the Nyquist bin)
    u_hat[np.arange(half, 0, -1)] = np.conj(pos[half:])
    u_hat[0] = np.conj(u_hat[-1])
    return u_hat


def spectrum_to_time(u_hat: np.ndarray) -> np.ndarray:
    """IFFT of an ``fftshift``-ed spectrum; return the real part."""
    return np.real(ifft(ifftshift(np.asarray(u_hat, dtype=np.complex128).ravel())))


class VME(object):
    """
    Variational Mode Extraction.

    VME extracts **one** compact-spectrum mode around a prescribed centre
    frequency, rather than decomposing the whole signal as VMD does.  The
    residual is forced to have little energy at that centre frequency, which
    is the extra criterion relative to classical VMD.

    Nazari & Sakhaei, IEEE JBHI, 22(4):1059-1067, 2018.
    """

    def __init__(
        self,
        alpha: float = 20000.0,
        omega_init: float = 0.0,
        fs: float = 1.0,
        tau: float = 0.0,
        tol: float = 1e-7,
        max_iter: int = 300,
    ) -> None:
        """
        :param alpha: compactness / bandwidth penalty (paper / MATLAB default 2e4)
        :param omega_init: initial centre-frequency guess in Hz
        :param fs: sampling frequency in Hz; ``omega_init / fs`` is the
            normalised frequency used internally (MATLAB ``omega_int/fs``)
        :param tau: dual-ascent step.  Set to 0 under high-level noise
        :param tol: relative spectral-change tolerance (MATLAB default 1e-7)
        :param max_iter: maximum ADMM iterations (MATLAB hard-codes 300)
        """
        if float(alpha) < 0.0:
            raise ValueError("alpha must be non-negative")
        if float(fs) <= 0.0:
            raise ValueError("fs must be positive")
        if int(max_iter) < 2:
            raise ValueError("max_iter must be >= 2")
        if float(tol) <= 0.0:
            raise ValueError("tol must be positive")

        self.alpha = float(alpha)
        self.omega_init = float(omega_init)
        self.fs = float(fs)
        self.tau = float(tau)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.signal: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.u_hat: Optional[np.ndarray] = None
        self.omega: Optional[float] = None
        self.omega_hist: Optional[np.ndarray] = None
        self.n_iter: Optional[int] = None

    def __call__(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Allow instances to be called like functions."""
        return self.fit_transform(signal=signal, return_all=return_all)

    def __str__(self) -> str:
        return "Variational Mode Extraction (VME)"

    def fit_transform(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extract the mode of interest from ``signal``.

        :param signal: 1-D real array.  Odd lengths are truncated by one
            sample so that MATLAB ``T/2`` indexing is well-defined
        :param return_all: if True, also return the spectrum of the cropped
            mode and the centre-frequency iterates (normalised, in
            cycles/sample) as in MATLAB ``[u_d, u_hatd, omega]``
        :return: extracted mode, or ``(u_d, u_hat, omega_hist)``
        """
        x = ensure_even_length(signal)
        extended = mirror_extend(x)
        n_fft = extended.size
        omega_axis = spectral_axis(n_fft)
        f_hat_onesided = onesided_fft(extended)

        n_iter_max = self.max_iter
        u_hat_d = np.zeros((n_iter_max, n_fft), dtype=np.complex128)
        dual = np.zeros((n_iter_max, n_fft), dtype=np.complex128)
        omega_d = np.zeros(n_iter_max, dtype=float)
        omega_d[0] = self.omega_init / self.fs

        n = 0
        u_diff = self.tol + _EPS
        while u_diff > self.tol and n < n_iter_max - 1:
            u_hat_d[n + 1, :] = update_mode_spectrum(
                f_hat_onesided,
                u_hat_d[n, :],
                dual[n, :],
                omega_axis,
                omega_d[n],
                self.alpha,
            )
            omega_d[n + 1] = update_center_frequency(
                u_hat_d[n + 1, :], omega_axis, previous=omega_d[n]
            )
            dual[n + 1, :] = update_dual(
                dual[n, :],
                f_hat_onesided,
                u_hat_d[n + 1, :],
                omega_axis,
                omega_d[n + 1],
                self.alpha,
                self.tau,
            )
            n += 1
            u_diff = relative_spectrum_change(u_hat_d[n, :], u_hat_d[n - 1, :])

        last = min(n_iter_max, n)
        u_hat_full = reconstruct_hermitian(u_hat_d[last, :])
        u_time = crop_mirror(spectrum_to_time(u_hat_full))
        u_hat_crop = fftshift(fft(u_time))

        self.signal = x
        self.u = u_time
        self.u_hat = u_hat_crop
        self.omega_hist = omega_d[: last + 1].copy()
        self.omega = float(self.omega_hist[-1])
        self.n_iter = int(last)

        if return_all:
            return u_time, u_hat_crop, self.omega_hist
        return u_time


def vme(
    signal: np.ndarray,
    alpha: float = 20000.0,
    omega_init: float = 0.0,
    fs: float = 1.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 300,
    return_all: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Functional interface to :class:`VME` (MATLAB ``vme.m``)."""
    return VME(
        alpha=alpha,
        omega_init=omega_init,
        fs=fs,
        tau=tau,
        tol=tol,
        max_iter=max_iter,
    ).fit_transform(signal=signal, return_all=return_all)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    demo = generate_vme_example2()
    extractor = VME(
        alpha=20000.0,
        omega_init=float(demo["omega_init"]),
        fs=float(demo["fs"]),
        tau=0.0,
        tol=1e-7,
        max_iter=300,
    )
    mode = extractor.fit_transform(demo["signal"])
    t = demo["t"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t, demo["signal"])
    axes[0].set_title("Mixture")
    axes[1].plot(t, demo["reference"] / np.max(np.abs(demo["reference"])), "r-.")
    axes[1].plot(t, mode / np.max(np.abs(mode)))
    axes[1].set_title("Extracted mode vs. reference")
    plt.show()
