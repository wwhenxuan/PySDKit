# -*- coding: utf-8 -*-
"""
Empirical Fourier Decomposition (EFD).

Zhou, Feng, Xu, Wang, Lv.
Empirical Fourier decomposition: An accurate signal decomposition method
for nonlinear and non-stationary time series analysis.
Mechanical Systems and Signal Processing, 163:108155, 2022.

Faithful Python port of the MATLAB toolbox (File Exchange 97747):
``EFD.m``, ``Segm_tec.m``, ``plotbounds.m``, ``IFIA.m``.

EFD is an adaptive Fourier-spectrum method in the same family as EWT:
it partitions the magnitude spectrum and applies a filter bank.  Unlike
EMD it does not sift in time, and unlike EWT it uses ideal (brick-wall,
zero-phase) filters rather than Meyer wavelets with transition bands.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from scipy.signal import hilbert

from pysdkit.utils import fft, ifft


def matlab_half_length(n: int) -> int:
    """MATLAB ``round(n/2)`` for a positive integer length.

    Half-integers round away from zero, which for ``n > 0`` is ``(n + 1) // 2``.
    NumPy's ``np.round`` uses banker's rounding and disagrees for some odd ``n``.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    return (int(n) + 1) // 2


def copy_matlab_range(dst: np.ndarray, src: np.ndarray, a: int, b: int) -> np.ndarray:
    """Copy MATLAB 1-based inclusive range ``a:b`` from ``src`` into ``dst``.

    Empty when ``a > b`` (MATLAB's behaviour for decreasing ranges).
    """
    a = int(a)
    b = int(b)
    if a > b:
        return dst
    dst[a - 1 : b] = src[a - 1 : b]
    return dst


def mirror_extend(signal: np.ndarray) -> Tuple[np.ndarray, int]:
    """Mirror-extend a 1-D signal as in MATLAB ``EFD.m``.

    With ``l = round(length(x)/2)``::

        x_ext = [x(l-1:-1:1); x; x(end:-1:end-l+1)]

    Cropping the filtered result with ``x_ext[l-1 : len(x_ext)-l]`` recovers
    a vector of the original length (odd lengths are kept).

    :param signal: 1-D real signal
    :return: ``(x_ext, l)``
    """
    x = np.asarray(signal, dtype=float).ravel()
    l = matlab_half_length(x.size)
    left = np.flip(x[: l - 1])
    right = np.flip(x[-l:])
    return np.concatenate([left, x, right]), l


def apply_ideal_bandpass(
    spectrum: np.ndarray, bound_left: int, bound_right: int
) -> np.ndarray:
    """One ideal (brick-wall) band of MATLAB ``EFD.m``, including Hermitian bins.

    ``bound_left`` / ``bound_right`` are 1-based FFT indices as produced by
    ``ceil(bounds * round(len(ff)/2) / pi)``.  Adjacent bands share the
    boundary bin, matching the original implementation.
    """
    ff = np.asarray(spectrum)
    ft = np.zeros_like(ff)
    length = ff.size
    if bound_left == 0:
        copy_matlab_range(ft, ff, 1, bound_right)
        copy_matlab_range(ft, ff, length + 2 - bound_right, length)
    else:
        copy_matlab_range(ft, ff, bound_left, bound_right)
        copy_matlab_range(ft, ff, length + 2 - bound_right, length + 2 - bound_left)
    return ft


def segm_tec(f: np.ndarray, n_segments: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Improved Fourier-spectrum segmentation (MATLAB ``Segm_tec.m``).

    Detect local maxima of the one-sided magnitude spectrum, keep the ``N``
    largest, and place each boundary at the lowest sample between consecutive
    maxima (plus the spectrum ends).

    :param f: one-sided magnitude spectrum (MATLAB ``abs(ff(1:round(L/2)))``)
    :param n_segments: maximum number of modes / retained maxima (``N``)
    :return: ``(bounds, cerf)``

        * ``bounds`` — bin indices of the ``N+1`` boundaries (0-based in the
          same sense as MATLAB ``omega(i)+ind-2``, so the first bound can be 0)
        * ``cerf`` — central frequencies of the retained maxima, in ``[0, π]``
    """
    spectrum = np.asarray(f, dtype=float).ravel()
    n_spec = spectrum.size
    if n_spec < 3:
        raise ValueError("spectrum must contain at least 3 samples")
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")

    locmax = np.zeros(n_spec, dtype=float)
    for i in range(1, n_spec - 1):
        if spectrum[i - 1] < spectrum[i] and spectrum[i] > spectrum[i + 1]:
            locmax[i] = spectrum[i]
    locmax[0] = spectrum[0]
    locmax[-1] = spectrum[-1]

    # MATLAB: [lmax, Imax] = sort(locmax, 1, 'descend') — 1-based Imax
    order = np.argsort(-locmax, kind="mergesort")
    if locmax.size > n_segments:
        imax_1based = np.sort(order[:n_segments] + 1)
    else:
        imax_1based = np.sort(order + 1)
        n_segments = locmax.size

    n_bounds = n_segments + 1
    omega = np.concatenate([[1], imax_1based, [n_spec]])
    bounds = np.zeros(n_bounds, dtype=float)
    for i in range(n_bounds):
        left = int(omega[i])
        right = int(omega[i + 1])
        if (i == 0 or i == n_bounds - 1) and left == right:
            bounds[i] = left - 1
        else:
            slc = spectrum[left - 1 : right]
            ind = int(np.argmin(slc)) + 1
            bounds[i] = left + ind - 2

    # MATLAB: cerf = Imax * pi / round(length(f)), and f is already the
    # one-sided spectrum, so the denominator is n_spec (not n_spec/2).
    cerf = imax_1based.astype(float) * np.pi / float(n_spec)
    return bounds, cerf


def ifia(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Instantaneous frequency and amplitude via Hilbert transform (``IFIA.m``).

    Central differences of the unwrapped analytic phase, padded at the ends
    exactly as in MATLAB ``comp_inst_fre_amp``.

    :param x: 1-D component or 2-D array of modes with shape ``(n_modes, n_samples)``
    :param fs: sampling frequency
    :return: ``(inst_freq, inst_amp)`` with the same layout as ``x``
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return _comp_inst_freq_amp(arr, fs)
    if arr.ndim != 2:
        raise ValueError("x must be 1-D or 2-D")
    freqs = np.empty_like(arr)
    amps = np.empty_like(arr)
    for k in range(arr.shape[0]):
        freqs[k], amps[k] = _comp_inst_freq_amp(arr[k], fs)
    return freqs, amps


def _comp_inst_freq_amp(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """MATLAB nested ``comp_inst_fre_amp``."""
    analytic = hilbert(np.asarray(x, dtype=float).ravel())
    inst_amp = np.abs(analytic)
    phi = np.unwrap(np.angle(analytic))
    if phi.size < 3:
        inst_freq = np.zeros_like(phi)
        return inst_freq, inst_amp
    diff_phi = (phi[2:] - phi[:-2]) / 2.0
    diff_phi = np.concatenate([[diff_phi[0]], diff_phi, [diff_phi[-1]]])
    inst_freq = diff_phi * (float(fs) / (2.0 * np.pi))
    return inst_freq, inst_amp


def plot_bounds(
    signal: np.ndarray,
    boundaries: np.ndarray,
    ax=None,
):
    """Plot detected boundaries on the Fourier spectrum (``plotbounds.m``).

    :param signal: time-domain signal (FFT is taken internally)
    :param boundaries: boundary frequencies in radians on ``[0, π]``
    :param ax: optional Matplotlib axes; a new figure is created otherwise
    :return: the axes used for drawing
    """
    import matplotlib.pyplot as plt

    x = np.asarray(signal, dtype=float).ravel()
    magf = np.abs(fft(x))
    freq = 2.0 * np.pi * np.arange(magf.size) / magf.size
    half = matlab_half_length(magf.size)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(freq[:half], magf[:half])
    ax.set_xlim(0.0, np.pi)
    ymax = float(np.max(magf)) if magf.size else 1.0
    ax.set_ylim(0.0, ymax)
    for bound in np.atleast_1d(np.asarray(boundaries, dtype=float)).ravel():
        ax.plot(
            [bound, bound],
            [0.0, ymax],
            linestyle="--",
            color="magenta",
            marker="o",
            markersize=3,
            linewidth=1,
        )
    ax.set_xlabel(r"$\omega$ (rad)")
    ax.set_ylabel(r"$|\hat x(\omega)|$")
    return ax


class EFD(object):
    """
    Empirical Fourier Decomposition.

    The method combines an improved Fourier-spectrum segmentation technique
    with an ideal (zero-phase, no transition band) filter bank.  The number
    of modes is prescribed, which avoids the inconsistent segmentation of
    classical EWT, and the brick-wall filters avoid Meyer-wavelet leakage
    between neighbouring bands.

    Wei Zhou, Zhongren Feng, Y.F. Xu, Xiongjiang Wang, Hao Lv,
    Empirical Fourier decomposition: An accurate signal decomposition method
    for nonlinear and non-stationary time series analysis,
    Mechanical Systems and Signal Processing, 163:108155, 2022.
    https://doi.org/10.1016/j.ymssp.2021.108155
    """

    def __init__(self, max_imfs: Optional[int] = 3) -> None:
        if max_imfs is None or int(max_imfs) < 1:
            raise ValueError("max_imfs must be >= 1")
        self.max_imfs = int(max_imfs)

    def __call__(
        self, signal: np.ndarray, return_all: Optional[bool] = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Allow instances to be called like functions."""
        return self.fit_transform(signal=signal, return_all=return_all)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm."""
        return "Empirical Fourier Decomposition (EFD)"

    def fit_transform(
        self, signal: np.ndarray, return_all: Optional[bool] = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Decompose a 1-D signal with EFD (MATLAB ``EFD.m``).

        :param signal: time-domain signal (1-D array; row or column)
        :param return_all: if True, also return central frequencies and
            spectrum boundaries in ``[0, π]``
        :return: IMFs of shape ``(n_modes, n_samples)``, or
            ``(imfs, cerf, bounds)`` when ``return_all`` is True
        """
        x = np.asarray(signal, dtype=float).ravel()
        if x.size < 3:
            raise ValueError("signal must contain at least 3 samples")

        spectrum = fft(x)
        half = matlab_half_length(spectrum.size)
        bounds, cerf = segm_tec(np.abs(spectrum[:half]), n_segments=self.max_imfs)
        bounds = bounds * np.pi / half

        x_ext, half_len = mirror_extend(x)
        ff_ext = fft(x_ext)
        bound2 = np.ceil(bounds * matlab_half_length(ff_ext.size) / np.pi).astype(int)

        n_modes = bound2.size - 1
        imfs = np.zeros((n_modes, x.size), dtype=float)
        ext_len = ff_ext.size
        for k in range(n_modes):
            band = apply_ideal_bandpass(ff_ext, int(bound2[k]), int(bound2[k + 1]))
            recovered = np.real(ifft(band))
            imfs[k, :] = recovered[half_len - 1 : ext_len - half_len]

        if return_all:
            return imfs, cerf, bounds
        return imfs


def efd(
    signal: np.ndarray,
    max_imfs: Optional[int] = 3,
    return_all: Optional[bool] = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Functional interface to :class:`EFD`."""
    return EFD(max_imfs=max_imfs).fit_transform(signal=signal, return_all=return_all)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from pysdkit.plot import plot_IMFs

    fs = 1000.0
    t = np.arange(0.0, 1.0 + 1.0 / fs, 1.0 / fs)
    f11 = 6.0 * t
    f12 = 2.0 * np.cos(8.0 * np.pi * t)
    f13 = np.cos(40.0 * np.pi * t)
    s = f11 + f12 + f13

    decomp = EFD(max_imfs=3)
    modes = decomp.fit_transform(s)
    plot_IMFs(s, modes)
    plt.show()
