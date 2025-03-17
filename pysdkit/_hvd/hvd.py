# -*- coding: utf-8 -*-
"""
Created on 2025/02/06 00:19:23
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Tuple

from pysdkit.utils import differ
from pysdkit.utils import fft, ifft
from pysdkit.utils import fmirror


class HVD(object):
    """
    Hilbert Vibration Decomposition

    The Hilbert Vibration Decomposition is an adaptive/data-driven separation of a multi-component non-stationary vibration signal into simple quasi-harmonic components.
    The method is characterized by high frequency and amplitude resolution, which provides  a comprehensive account of the case of amplitude and frequency modulated vibration analysis.
    The HVD is a simple and fast recursive vibration mode decomposition, that sifts out a 1D input signal into a set of k-band separated simplest components with their envelopes and instantaneous frequencies.
    The HVD decomposes composition as a vector and returns a matrix of inherent components as the Hilbert spectrum plus residual signal.

    Feldman, Michael. "Time-varying vibration decomposition and analysis based on the Hilbert transform."
    Journal of Sound and Vibration 295.3-5 (2006): 518-530.

    Ramos, J. J., J. I. Reyes, and E. Barocio. "An improved Hilbert Vibration Decomposition method for analysis of low frequency oscillations."
    2014 IEEE PES Transmission & Distribution Conference and Exposition-Latin America (PES T&D-LA). IEEE, 2014.

    Python code: https://github.com/MVRonkin/dsatools/blob/master/dsatools/_base/_imf_decomposition/_hvd.py#L6

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/178804-hilbert-vibration-decomposition?s_tid=FX_rc1_behav

    The original code has a serious endpoint problem.
    The algorithm provided cannot handle the signal decomposition near the endpoint well.
    Therefore, we have improved it by mirroring the input signal to alleviate this problem.
    It can work only with narrowband signals!
    """

    def __init__(
        self, K: int = 3, fpar: Optional[int] = 20, mirror: Optional[bool] = True
    ) -> None:
        """
        Create the Hilbert Vibration Decomposition instance

        Decomposition results are very depends on the `fpar` value
        We have improved the original code by mirroring the input signal to alleviate this problem.
        
        :param K: the number of intrinsic mode functions to be decomposed
        :param fpar: filter parameter, equal to point of cut frequency for low-pass filter (have to be regulized for optimal decomposition).
        :param mirror: whether to mirror the original input signal
        """
        self.K = K
        self.fpar = int(fpar)
        self.mirror = mirror

    def __call__(
        self, signal: np.ndarray, return_all: Optional[bool] = False
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(signal, return_all)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Hilbert Vibration Decomposition (HVD)"

    def square_window(
        self,
        seq_len: int,
        w_filt: Tuple[int, int] = None,
        real_valued_filter: bool = True,
    ) -> np.ndarray:
        """
        Square window in range 0:fs//2.
        It is mainly used to specify a frequency band range in the frequency domain,
        where the frequency components within this range are retained,
        while those outside the range are suppressed.

        :param seq_len: filter length
        :param w_filt: cut-off low and high points
        :param real_valued_filter: whether to make the window a real-valued filter
        :return: the square window
        """
        if w_filt is None:
            w_filt = (0, self.fpar)

        lp, fp = w_filt

        # Filter window parameter validation
        if lp > seq_len:
            lp = int(seq_len)
        if lp - fp < 0:
            lp, fp = fp, lp

        # Generate the window
        one = np.ones(lp - fp)
        z2 = np.zeros(seq_len - fp)

        if fp == 0:
            Hp = np.hstack((one, z2))
        else:
            z1 = np.zeros(fp)
            Hp = np.hstack((z1, one, z2))

        # Real-valued filter processing
        if real_valued_filter:
            # The function converts the window to a real-valued filter
            Hp = make_window_real_valued(Hp, seq_len)

        return Hp

    def fit_transform(
        self, signal: np.ndarray, return_all: Optional[bool] = False
    ) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Start executing the Hilbert Vibration Decomposition algorithm."""

        # Mirror the input signal to improve decomposition at the endpoints
        sym = len(signal) // 2
        if self.mirror is True:
            signal = fmirror(ts=signal, sym=sym)

        # Get the length of the input signal
        seq_len = signal.shape[0]

        # Create a time array
        time = np.arange(seq_len)

        # Alternatively, using band filtration of the second harmonic can be more stable
        hp = self.square_window(
            seq_len=seq_len, w_filt=(0, self.fpar), real_valued_filter=True
        )

        f = np.zeros(self.K)

        # Array to record the decomposition results
        imfs = np.zeros(shape=(self.K, seq_len), dtype=signal.dtype)

        # Make a copy of the input signal
        signal_rest = signal.copy()

        # Start the iterative loop for signal decomposition
        for i in range(self.K):
            phase = np.unwrap(np.angle(np.asarray(signal_rest)))
            intfreq = differ(phase, delta=1) / 2 / np.pi
            intfreqf = filter_by_window(intfreq, hp)

            # To avoid transition zone
            f[i] = np.abs(np.mean(intfreqf[seq_len // 50 : -seq_len // 50]))

            signal_ref = np.exp(-2j * np.pi * f[i] * time)

            signal_ = filter_by_window(signal_rest * signal_ref, hp)

            env = np.abs(signal_)

            phase0 = np.angle(signal_)

            # Record the intrinsic mode function from this decomposition
            imfs[i, :] = env * np.exp(1j * (2 * np.pi * f[i] * time + phase0))

            # Remove the separated mode from the original signal
            signal_rest = signal_rest - imfs[i, :]

        if self.mirror is True:
            # Remove the mirrored part after decomposition
            imfs = imfs[:, sym:-sym]

        if return_all:
            # Return both the decomposed intrinsic mode functions and frequency information
            return imfs, f
        else:
            return imfs


def make_window_real_valued(H, N):
    """
    Make window real valued

    :param H: spectrum window (in frequency domain) of 1D ndarray
    :param N: filter length (sample length)
    :return: spectrum window (in frequency domain) with two parts in the range 0:fs//2 and fs//2:fs
             (mirrored and shifted by 1 point to fs//2)
    """
    H[N // 2 + 1 : N] = H[1 : N // 2][::-1]
    return H


def filter_by_window(signal: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Filter by spectrum window

    Apply the specified filter `H` to the input signal `signal` to perform signal filtering.
    In the implementation, the input signal needs to be transformed from the time domain to the frequency domain to apply the filter.
    Finally, the filtered signal is reconstructed back to the time domain using the inverse Fourier transform.

    :param signal: the input 1D numpy ndarray signal
    :param H: filter window (in frequency domain) of 1D ndarray
    :return: filtered 1D numpy ndarray signal in the time domain
    """
    signal = np.asarray(signal)
    N = int(signal.shape[0])

    # Transform the input signal from the time domain to the frequency domain using the Fast Fourier Transform (FFT)
    Sp = fft(signal)

    # Apply the filter in the frequency domain
    Sp = Sp * np.conj(H[0:N])

    # Reconstruct the signal in the time domain using the inverse Fast Fourier Transform (IFFT)
    signal_time_domain = ifft(Sp)

    return signal_time_domain
