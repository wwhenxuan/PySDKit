# -*- coding: utf-8 -*-
"""
Underdetermined blind source separation (BSS).

Faithful Python port of Yu's MATLAB ``YGBSS.m`` (Shock and Vibration
2019; the ``YG`` prefix is the author's initials).  SCA clusters
*real STFT coefficients* and fails on delay mixtures (Lissajous
ellipses).  This BSS clusters **frequency energy**

    E_i(ω) = sum_t |G_i(t, ω)|^2

so a delay (phase) drops out and each modal becomes a straight line
in the energy scatter plot.  Frequency bins whose energy vector is
cosine-close to a peak are kept as a binary mask; the sources are
the inverse STFT of one observation through those masks, then
corrected by the empirical padding line (exponent ``1/1.6``).

The MATLAB code uses ``sum(abs(STFT).^2)`` rather than the paper's
``∫|G| dt``, and MATLAB ``hamming(L)`` rather than TFTB Hamming.

G. Yu, *An Underdetermined Blind Source Separation Method with
Application to Modal Identification*, Shock and Vibration 2019,
Article ID 1637163.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ._stft import (
    default_window_length,
    hamming_window,
    padding_line,
    tfristft,
    tfrstft_uniform,
)


def peakdet(
    values: np.ndarray,
    delta: float,
    x: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local maxima and minima (Eli Billauer, public domain ``peakdet``).

    A sample is a maximum if it is the running max and is then
    preceded (to the left) by a value lower by ``delta``.

    :param values: 1-D record.
    :param delta: positive scalar threshold.
    :param x: optional abscissa; default is ``0 … N-1`` (MATLAB uses
        ``1:N``).  When given, the first column of the tables stores
        these coordinates instead of integer indices.
    :return: ``(maxtab, mintab)`` each ``(n, 2)`` with columns
        ``(location, value)``.  Empty peaks are ``(0, 2)``.
    """
    samples = np.asarray(values, dtype=float).ravel()
    if x is None:
        axis = np.arange(samples.size, dtype=float)
    else:
        axis = np.asarray(x, dtype=float).ravel()
        if axis.size != samples.size:
            raise ValueError("Input vectors v and x must have same length")
    delta_arr = np.asarray(delta, dtype=float).ravel()
    if delta_arr.size != 1:
        raise ValueError("Input argument DELTA must be a scalar")
    delta = float(delta_arr[0])
    if delta <= 0.0:
        raise ValueError("Input argument DELTA must be positive")

    maxtab = []
    mintab = []
    mn = np.inf
    mx = -np.inf
    mnpos = np.nan
    mxpos = np.nan
    look_for_max = True
    for index, this in enumerate(samples):
        if this > mx:
            mx = float(this)
            mxpos = float(axis[index])
        if this < mn:
            mn = float(this)
            mnpos = float(axis[index])
        if look_for_max:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = float(this)
                mnpos = float(axis[index])
                look_for_max = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = float(this)
                mxpos = float(axis[index])
                look_for_max = True
    max_arr = (
        np.asarray(maxtab, dtype=float).reshape(-1, 2)
        if maxtab
        else np.zeros((0, 2), dtype=float)
    )
    min_arr = (
        np.asarray(mintab, dtype=float).reshape(-1, 2)
        if mintab
        else np.zeros((0, 2), dtype=float)
    )
    return max_arr, min_arr


def frequency_energy(coefs: np.ndarray) -> np.ndarray:
    """
    Frequency energy used by BSS.

    MATLAB ``sum(abs(squeeze(coefs).').^2)``: sum over **time** of
    ``|STFT|^2`` for each frequency bin.

    :param coefs: STFT ``(n_freq, T)`` or a stack ``(m, n_freq, T)``.
    :return: ``(n_freq,)`` or ``(m, n_freq)``.
    """
    maps = np.asarray(coefs)
    if maps.ndim == 2:
        return np.sum(np.abs(maps) ** 2, axis=1)
    if maps.ndim == 3:
        return np.sum(np.abs(maps) ** 2, axis=2)
    raise ValueError("coefs must be (n_freq, T) or (m, n_freq, T)")


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    """
    MATLAB ``pdist([a; b], 'cosine')`` = ``1 - (a·b) / (|a| |b|)``.

    Zero vectors yield ``nan`` (same as MATLAB); ``nan < e2`` is false.
    """
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return 1.0 - float(np.dot(a, b) / denom)


def cosine_masks(
    energy: np.ndarray,
    peak_indices: np.ndarray,
    e2: float = 0.004,
    e3: float = 0.1,
) -> np.ndarray:
    """
    Binary frequency masks from cosine clustering of energy vectors.

    For each peak ``p`` and bin ``ω``::

        sum_i E_i(ω) > e3 * sum_i E_i(ω_p)
        and  cosine(E(:,ω), E(:,ω_p)) < e2

    :param energy: ``(m, n_freq)``.
    :param peak_indices: 0-based frequency indices of the *positive*
        half of the detected peaks (length ``K``).
    :return: masks of shape ``(K, n_freq)``.
    """
    energy = np.asarray(energy, dtype=float)
    if energy.ndim != 2:
        raise ValueError("energy must be (n_channels, n_freq)")
    n_freq = energy.shape[1]
    peaks = np.asarray(peak_indices, dtype=int).ravel()
    n_sources = int(peaks.size)
    if n_sources == 0:
        return np.zeros((0, n_freq), dtype=float)

    energy_sum = np.sum(energy, axis=0)
    peak_sum = energy_sum[peaks]
    gate = energy_sum[:, None] > (float(e3) * peak_sum[None, :])

    norms = np.linalg.norm(energy, axis=0)
    dots = energy.T @ energy[:, peaks]
    denom = norms[:, None] * norms[peaks][None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_dist = 1.0 - dots / denom
    similar = np.isfinite(cos_dist) & (cos_dist < float(e2))
    return np.where(gate & similar, 1.0, 0.0).T


def _as_mixtures(signal: np.ndarray) -> np.ndarray:
    mixtures = np.asarray(signal, dtype=float)
    if mixtures.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_channels, n_samples)")
    n_channels, n_samples = mixtures.shape
    if n_channels > n_samples:
        raise ValueError("X must be row vectors")
    if n_channels < 2:
        raise ValueError("X must have two row at least")
    return np.ascontiguousarray(mixtures, dtype=float)


class BSS(object):
    """
    Underdetermined BSS by frequency-energy clustering (Yu, 2019).

    * :meth:`fit_transform` / :meth:`__call__` — separate ``X`` of
      shape ``(m, T)`` into sources ``(K, T)``.
    * :attr:`mixing_` — column-normalised absolute mixing / mode-shape
      matrix ``(m, K)``.

    :param window_length: odd Hamming length ``L``.  ``None`` uses
        odd ``floor(T/4)``.
    :param e1: peak-detection fraction of ``max(sum E)`` (default 0.1).
        Lower values yield more sources.
    :param e2: cosine-distance threshold (default 0.004).
    :param e3: low-energy gate relative to the peak (default 0.1).
    :param channel: 0-based observation used for inverse STFT
        (MATLAB ``n``, default 1).
    """

    def __init__(
        self,
        window_length: Optional[int] = None,
        e1: float = 0.1,
        e2: float = 0.004,
        e3: float = 0.1,
        channel: int = 0,
    ) -> None:
        if window_length is not None and int(window_length) < 3:
            raise ValueError("window length must be >= 3")
        if window_length is not None and int(window_length) % 2 == 0:
            raise ValueError("H must be a smoothing window with odd length")
        if int(channel) < 0:
            raise ValueError("channel must be >= 0")
        self.window_length = None if window_length is None else int(window_length)
        self.e1 = float(e1)
        self.e2 = float(e2)
        self.e3 = float(e3)
        self.channel = int(channel)

        self.signal: Optional[np.ndarray] = None
        self.sources_: Optional[np.ndarray] = None
        self.mixing_: Optional[np.ndarray] = None
        self.energy_: Optional[np.ndarray] = None
        self.peaks_: Optional[np.ndarray] = None
        self.masks_: Optional[np.ndarray] = None
        self.pad_: Optional[np.ndarray] = None
        self.window_: Optional[np.ndarray] = None
        self.coefs_: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return "Underdetermined BSS"

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return self.fit_transform(signal)

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Separate underdetermined mixtures.

        :param signal: observations ``(m, T)``, ``m >= 2``, ``m <= T``.
        :return: sources ``(K, T)``.  ``K = n_peaks // 2`` (positive-
            frequency half of the two-sided energy peaks).
        """
        mixtures = _as_mixtures(signal)
        n_channels, n_samples = mixtures.shape
        if self.channel >= n_channels:
            raise ValueError("n must be less than row vectors")

        length = (
            default_window_length(n_samples)
            if self.window_length is None
            else int(self.window_length)
        )
        window = hamming_window(length)
        coefs = np.empty((n_channels, n_samples, n_samples), dtype=np.complex128)
        for channel in range(n_channels):
            coefs[channel] = tfrstft_uniform(
                mixtures[channel], window, n_freq=n_samples
            )

        energy = frequency_energy(coefs)
        energy_sum = np.sum(energy, axis=0)
        delta = float(np.max(np.abs(energy_sum))) * self.e1
        maxtab, _mintab = peakdet(energy_sum, delta)
        n_max = int(maxtab.shape[0])
        n_sources = n_max // 2
        if n_sources == 0:
            empty = np.zeros((0, n_samples), dtype=float)
            self.signal = mixtures
            self.sources_ = empty
            self.mixing_ = np.zeros((n_channels, 0), dtype=float)
            self.energy_ = energy
            self.peaks_ = np.zeros((0,), dtype=int)
            self.masks_ = np.zeros((0, n_samples), dtype=float)
            self.pad_ = padding_line(n_samples, window)
            self.window_ = window
            self.coefs_ = coefs
            return empty

        peak_indices = np.asarray(np.round(maxtab[:n_sources, 0]), dtype=int)
        masks = cosine_masks(energy, peak_indices, e2=self.e2, e3=self.e3)

        pad = padding_line(n_samples, window)
        sources = np.empty((n_sources, n_samples), dtype=float)
        observation = coefs[self.channel]
        for source_idx in range(n_sources):
            masked = observation * masks[source_idx, :, None]
            recovered = tfristft(masked, window=window)
            sources[source_idx] = np.real(np.asarray(recovered).ravel()) * pad

        mixing = np.sqrt(np.maximum(energy[:, peak_indices], 0.0))
        col_norm = np.sqrt(np.sum(mixing**2, axis=0, keepdims=True))
        col_norm = np.maximum(col_norm, np.finfo(float).tiny)
        mixing = mixing / col_norm

        self.signal = mixtures
        self.sources_ = sources
        self.mixing_ = mixing
        self.energy_ = energy
        self.peaks_ = peak_indices
        self.masks_ = masks
        self.pad_ = pad
        self.window_ = window
        self.coefs_ = coefs
        return sources


def bss(
    signal: np.ndarray,
    window_length: Optional[int] = None,
    e1: float = 0.1,
    e2: float = 0.004,
    e3: float = 0.1,
    channel: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper around :class:`BSS`.

    MATLAB ``[source, A] = YGBSS(X, ...)``.  Not re-exported from
    ``pysdkit.utils`` so the ``bss`` subpackage is not shadowed.

    :return: ``(sources, mixing)``.
    """
    engine = BSS(
        window_length=window_length,
        e1=e1,
        e2=e2,
        e3=e3,
        channel=channel,
    )
    sources = engine.fit_transform(signal)
    mixing = np.asarray(engine.mixing_)
    return sources, mixing
