# -*- coding: utf-8 -*-
"""
Hilbert Vibration Decomposition (HVD).

Feldman, M. (2006). Time-varying vibration decomposition and analysis based
on the Hilbert transform. Journal of Sound and Vibration, 295(3-5):518–530.

The algorithm recursively extracts the largest-amplitude quasi-harmonic
component by (i) low-pass filtering the instantaneous frequency of the
analytic signal and (ii) synchronous demodulation of that carrier.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from scipy.signal import hilbert

from pysdkit.utils import differ, fft, ifft, fmirror


class HVD(object):
    """
    Hilbert Vibration Decomposition.

    Separates a multi-component non-stationary vibration into simple
    quasi-harmonic modes ordered by decreasing amplitude.  Each iteration
    estimates the carrier of the currently strongest component from the
    low-pass-filtered instantaneous frequency, recovers its envelope by
    coherent demodulation, subtracts the mode, and continues on the residual.

    Decomposition quality depends on ``fpar`` (low-pass cut-off in FFT bins).
    Endpoint effects are reduced by optional signal mirroring.

    References
    ----------
    Feldman, M. Journal of Sound and Vibration, 295:518–530, 2006.

    Ramos et al., IEEE PES T&D-LA, 2014.

    Python reference: https://github.com/MVRonkin/dsatools
    MATLAB: https://www.mathworks.com/matlabcentral/fileexchange/178804
    """

    def __init__(
        self,
        K: int = 3,
        fpar: Optional[int] = 20,
        mirror: Optional[bool] = True,
    ) -> None:
        """
        :param K: number of intrinsic components to extract
        :param fpar: low-pass cut-off in FFT bins for IF smoothing / demodulation
        :param mirror: mirror-pad the signal to mitigate end effects
        """
        if not isinstance(K, (int, np.integer)) or K < 1:
            raise ValueError("K must be a positive integer")
        if fpar is None:
            fpar = 20
        if not isinstance(fpar, (int, np.integer)) or fpar < 1:
            raise ValueError("fpar must be a positive integer")

        self.K = int(K)
        self.fpar = int(fpar)
        self.mirror = bool(mirror)

        self.imfs: Optional[np.ndarray] = None
        self.frequencies: Optional[np.ndarray] = None

    def __call__(
        self, signal: np.ndarray, return_all: Optional[bool] = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        return self.fit_transform(signal, return_all)

    def __str__(self) -> str:
        return "Hilbert Vibration Decomposition (HVD)"

    def square_window(
        self,
        seq_len: int,
        w_filt: Optional[Tuple[int, int]] = None,
        real_valued_filter: bool = True,
    ) -> np.ndarray:
        """
        Ideal band / low-pass window on ``0 … fs/2`` (FFT-bin units).

        :param seq_len: filter length (= signal length)
        :param w_filt: ``(low_bin, high_bin)``; default ``(0, fpar)``
        :param real_valued_filter: mirror the positive-frequency half
        """
        if w_filt is None:
            w_filt = (0, self.fpar)

        lp, fp = w_filt
        if lp > seq_len:
            lp = int(seq_len)
        if lp - fp < 0:
            lp, fp = fp, lp

        one = np.ones(lp - fp)
        z2 = np.zeros(seq_len - lp)
        if fp == 0:
            hp = np.hstack((one, z2))
        else:
            z1 = np.zeros(fp)
            hp = np.hstack((z1, one, z2))

        if real_valued_filter:
            hp = make_window_real_valued(hp, seq_len)
        return hp

    def fit_transform(
        self, signal: np.ndarray, return_all: Optional[bool] = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Run Hilbert Vibration Decomposition.

        :param signal: real 1-D vibration record
        :param return_all: if True, also return estimated carriers (cycles/sample)
        :return: IMFs of shape ``(K, N)`` [, frequency vector of length ``K``]
        """
        x = np.asarray(signal, dtype=float).ravel()
        if x.ndim != 1 or x.size < 8:
            raise ValueError("HVD expects a 1-D signal with length >= 8")

        sym = len(x) // 2
        work = fmirror(ts=x, sym=sym) if self.mirror else x.copy()
        seq_len = work.shape[0]
        time = np.arange(seq_len, dtype=float)

        hp = self.square_window(
            seq_len=seq_len, w_filt=(0, self.fpar), real_valued_filter=True
        )

        carriers = np.zeros(self.K, dtype=float)
        imfs = np.zeros((self.K, seq_len), dtype=float)
        residual = work.copy()

        for i in range(self.K):
            # Paper §4.1: IF of the analytic signal, then low-pass / average
            analytic = hilbert(residual)
            phase = np.unwrap(np.angle(analytic))
            inst_freq = differ(phase, delta=1) / (2.0 * np.pi)
            inst_freq_f = np.real(filter_by_window(inst_freq, hp))

            # Avoid filter transition near the ends
            lo, hi = seq_len // 50, max(seq_len - seq_len // 50, seq_len // 50 + 1)
            carriers[i] = float(np.abs(np.mean(inst_freq_f[lo:hi])))

            # Paper §4.2: synchronous demodulation (mixing + LPF)
            # For a real cosine A cos(ωt+φ), LPF(x e^{-jωt}) has magnitude A/2.
            carrier = np.exp(-2j * np.pi * carriers[i] * time)
            baseband = filter_by_window(residual * carrier, hp)
            env = np.abs(baseband)
            phase0 = np.angle(baseband)

            imfs[i, :] = 2.0 * env * np.cos(2.0 * np.pi * carriers[i] * time + phase0)
            residual = residual - imfs[i, :]

        if self.mirror:
            imfs = imfs[:, sym:-sym]

        self.imfs = imfs
        self.frequencies = carriers
        if return_all:
            return imfs, carriers
        return imfs


def make_window_real_valued(H: np.ndarray, N: int) -> np.ndarray:
    """Mirror the positive-frequency half so the window yields a real impulse response."""
    H = np.asarray(H, dtype=float).copy()
    H[N // 2 + 1 : N] = H[1 : N // 2][::-1]
    return H


def filter_by_window(signal: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply a frequency-domain window via FFT → multiply → IFFT.

    :param signal: time-domain signal (real or complex)
    :param H: frequency-domain window of length ``N``
    """
    signal = np.asarray(signal)
    n = int(signal.shape[0])
    spectrum = fft(signal) * np.conj(H[:n])
    return ifft(spectrum)
