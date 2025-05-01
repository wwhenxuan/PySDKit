# -*- coding: utf-8 -*-
"""
Created on 2025/02/06 10:30:01
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
Hilbert-transform for demodulation
"""
import numpy as np
from pysdkit.utils import fft, ifft


def hilbert(signal: np.ndarray) -> np.ndarray:
    """
    Perform Hilbert transform along the last axis of input signal.

    :param signal: The Hilbert transform is performed along last dimension.

    :return: A complex tensor with the same shape of the input signal, representing its analytic signal.
    """
    signal = np.asarray(signal, dtype=np.float64)

    # Get the length of input signal
    N = signal.shape[-1]

    # Perform a Fast Fourier Transform
    Xf = fft(signal)

    # Get the spectrum of the analytical signal
    if N % 2 == 0:
        Xf[..., 1 : N // 2] *= 2
        Xf[..., N // 2 + 1 :] = 0
    else:
        Xf[..., 1 : (N + 1) // 2] *= 2
        Xf[..., (N + 1) // 2 :] = 0
    return ifft(Xf)


def get_envelope_frequency(signal: np.ndarray, fs: float, ret_analytic: bool = False):
    """
    Compute the envelope and instantaneous frequency function of the given signal using Hilbert transform.
    The transformation is done along the last axis.

    Parameters:
    -------------
    x : array_like
        Signal data. The last dimension of `x` is considered the temporal dimension.
    fs : float
        Sampling frequency in Hz.
    ret_analytic : bool, optional
        Whether to return the analytic signal. (Default: False)

    Returns:
    -------------
    (envelope, freq)             if `ret_analytic` is False
    (envelope, freq, analytic)   if `ret_analytic` is True

    envelope : ndarray
        The envelope function with the same shape as `x`.
    freq : ndarray
        The instantaneous frequency function in Hz with the same shape as `x`.
    analytic : ndarray
        The analytic (complex) signal with the same shape as `x`.
    :param signal: Signal data. The last dimension is considered the temporal dimension.
    :param fs: Sampling frequency in Hz.
    :param ret_analytic: Whether to return the analytic signal. (Default: False)

    :return: (envelope, freq) if `ret_analytic` is False
             (envelope, freq, analytic)   if `ret_analytic` is True
             - envelope : ndarray, the envelope function with the same shape as `x`.
    freq : ndarray
        The instantaneous frequency function in Hz with the same shape as `x`.
    analytic : ndarray
        The analytic (complex) signal with the same shape as `x`.
    """
    signal = np.asarray(signal, dtype=np.float64)
    analytic = hilbert(signal)  # Scipy's hilbert returns analytic signal

    envelope = np.abs(analytic)

    # Compute sub (discrete derivative with edge extension)
    sub = np.empty_like(analytic)
    diff = np.diff(analytic, axis=-1)
    sub[..., :-1] = diff
    # Handle last element (repeat last difference)
    sub[..., -1] = (
        analytic[..., -1] - analytic[..., -2] if analytic.shape[-1] >= 2 else 0.0
    )

    # Compute add (discrete sum with edge extension)
    add = np.empty_like(analytic)
    summed = analytic[..., 1:] + analytic[..., :-1]
    add[..., :-1] = summed
    # Handle last element (2 * last value)
    add[..., -1] = 2 * analytic[..., -1]

    # Calculate instantaneous frequency (handle division safely)
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = 2 * fs * np.imag(sub / add)
    freq[np.isinf(freq)] = 0  # Replace infs
    freq /= 2 * np.pi  # Convert to Hz

    if ret_analytic:
        return envelope, freq, analytic
    else:
        return envelope, freq
