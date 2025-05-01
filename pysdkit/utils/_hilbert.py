# -*- coding: utf-8 -*-
"""
Created on Sat Mar 7 12:09:42 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from typing import Optional


def hilbert_transform(signal: np.ndarray) -> np.ndarray:
    """
    Apply the Hilbert transform to a given numpy signal.

    :param signal: NumPy array containing the input signal.
    :return: A NumPy array containing the analytical signal obtained from the Hilbert transform.
    """
    analytical_signal = hilbert(signal)  # Compute the analytical signal
    return analytical_signal


def hilbert_real(signal: np.ndarray) -> np.ndarray:
    """Get the real part of the Hilbert transformed signal"""
    return np.real(signal)


def hilbert_imaginary(signal: np.ndarray) -> np.ndarray:
    """Get the imaginary part of the Hilbert transformed signal"""
    return np.imag(signal)


# def hilbert_spectrum(
#         imfs_env, imfs_freq, fs, freq_lim=None, freq_res=None, time_range=None, time_scale=1
# ):
#     """
#     Compute the Hilbert spectrum H(t, f) (which quantify the changes of frequencies of all IMFs over time).
#
#     Parameters:
#     ------------
#     imfs_env : Tensor, of shape (..., # IMFs, # sampling points )
#             The envelope functions of all IMFs.
#     imfs_freq : Tensor, of shape (..., # IMFs, # sampling points )
#             The instantaneous frequency functions of all IMFs.
#     fs : real.
#         Sampling frequencies in Hz.
#     freq_max : real, Optional.
#         Specifying the maximum instantaneous frequency. If not given, it will be
#         automatically selected.
#     freq_res : real. Optional.
#         Specifying the frequency resolution.
#         If not given, it will be 1 / (total_time_length) = fs / N.
#     time_range : (real, real)-tuple. Optional.
#         Specifying the range of time domain. If not given, it will be the time span
#         of the whole signal, i.e. (0, N*fs).
#     time_scale : int. Optional. ( Default : 1 )
#         Specifying the scale for the time axis.
#         Thus temporal resolution will be exactly `1/fs * time_scale`.
#
#     Returns:
#     ----------
#     (spectrum, time_axis, freq_axis)
#
#     spectrum : Tensor, of shape ( ..., # time_bins, # freq_bins ).
#         A pytorch tensor, representing the Hilbert spectrum H(t, f).
#         The tensor will be on the same device as `imfs_env` and `imfs_freq`.
#     time_axis : Tensor, 1D, of shape ( # time_bins )
#         The label for the time axis of the spectrum.
#     freq_axis : Tensor, 1D, of shape ( # freq_bins )
#         The label for the frequency axis (in `freq_unit`) of the spectrum.
#
#     """
#
#     N = imfs_freq.shape[-1]  # total number of sampling points
#     T = N / fs  # total time length
#
#     if freq_lim is None:
#         freq_min, freq_max = 0, fs / 2
#     else:
#         freq_min, freq_max = freq_lim
#
#     if freq_res is None:
#         freq_res = (freq_max - freq_min) / 200  # frequency resolution
#
#     dim_batch = imfs_env.shape[:-2]
#     num_imfs = imfs_env.shape[-2]
#     imfs_env = imfs_env.view(-1, num_imfs, N)
#     imfs_freq = imfs_freq.view(-1, num_imfs, N)
#     num_batches = imfs_env.shape[0]
#
#     if time_range:
#         L, R = time_range
#         L, R = min(int(L * fs), N - 1), min(int(R * fs) + 1, N)
#         imfs_env, imfs_freq = imfs_env[..., L:R], imfs_freq[..., L:R]
#         N = R - L
#
#     freq_bins = int((freq_max - freq_min) / freq_res) + 1
#     time_bins = N // time_scale + 1
#
#     spectrum = torch.zeros((num_batches, time_bins, freq_bins + 1), device=device)
#
#     batch_idx = (torch.arange(num_batches, dtype=torch.long, device=device)).view(
#         -1, 1, 1
#     )
#     time_idx = (torch.arange(N, dtype=torch.long, device=device) // time_scale).view(
#         1, 1, -1
#     )
#     freq_idx = ((imfs_freq - freq_min) / freq_res).long()
#
#     # out-of-range frequency values will be discarded later
#     freq_idx[freq_idx < 0] = freq_bins
#     freq_idx[freq_idx > freq_bins] = freq_bins
#
#     spectrum[batch_idx, time_idx, freq_idx] += imfs_env**2
#     # spectrum = spectrum / freq_res * fs / time_scale (density spectrum)
#     del batch_idx, time_idx, freq_idx
#
#     time_axis = torch.arange(
#         N // time_scale + 1, dtype=torch.double
#     ) * time_scale / fs + (L / fs if time_range is not None else 0)
#     freq_axis = torch.arange(freq_bins, dtype=torch.double) * freq_res + freq_min
#
#     return (
#         spectrum[:, :, :freq_bins].view(dim_batch + torch.Size([time_bins, freq_bins])),
#         time_axis,
#         freq_axis,
#     )


def hilbert_spectrum(
        imfs_env, imfs_freq, fs, freq_lim=None, freq_res=None, time_range=None, time_scale=1
):
    """
    Compute the Hilbert spectrum H(t, f) using numpy.

    Parameters:
    ------------
    imfs_env : array_like, shape (..., # IMFs, # sampling points)
        The envelope functions of all IMFs.
    imfs_freq : array_like, shape (..., # IMFs, # sampling points)
        The instantaneous frequency functions.
    fs : float
        Sampling frequency in Hz.
    freq_lim : tuple (float, float), optional
        Frequency range (min, max). Defaults to (0, fs/2).
    freq_res : float, optional
        Frequency resolution. Defaults to (freq_max - freq_min)/200.
    time_range : tuple (float, float), optional
        Time range (start, end) in seconds.
    time_scale : int, optional
        Temporal scaling factor (Default: 1).

    Returns:
    ----------
    (spectrum, time_axis, freq_axis)

    spectrum : ndarray, shape (..., time_bins, freq_bins)
        Hilbert spectrum matrix
    time_axis : ndarray, 1D
        Time axis labels
    freq_axis : ndarray, 1D
        Frequency axis labels
    """
    imfs_env = np.asarray(imfs_env, dtype=np.float64)
    imfs_freq = np.asarray(imfs_freq, dtype=np.float64)

    N = imfs_freq.shape[-1]  # Number of sampling points
    original_shape = imfs_env.shape[:-2]
    num_imfs = imfs_env.shape[-2]

    # Frequency parameters
    if freq_lim is None:
        freq_min, freq_max = 0.0, fs / 2
    else:
        freq_min, freq_max = freq_lim

    if freq_res is None:
        freq_res = (freq_max - freq_min) / 200

    # Time range handling
    L, R = 0, N
    if time_range is not None:
        L = int(np.clip(time_range[0] * fs, 0, N-1))
        R = int(np.clip(time_range[1] * fs + 1, L+1, N))
        imfs_env = imfs_env[..., L:R]
        imfs_freq = imfs_freq[..., L:R]
        N = R - L

    # Reshape for batch processing
    imfs_env = imfs_env.reshape(-1, num_imfs, N)
    imfs_freq = imfs_freq.reshape(-1, num_imfs, N)
    num_batches = imfs_env.shape[0]

    # Initialize spectrum
    freq_bins = int(np.round((freq_max - freq_min) / freq_res)) + 1
    time_bins = N // time_scale + 1
    spectrum = np.zeros((num_batches, time_bins, freq_bins + 1))

    # Generate indices
    batch_idx = np.arange(num_batches)[:, np.newaxis, np.newaxis]
    time_idx = (np.arange(N) // time_scale).reshape(1, 1, -1)

    # Frequency index calculation
    freq_idx = np.round((imfs_freq - freq_min) / freq_res).astype(int)
    freq_idx = np.clip(freq_idx, 0, freq_bins)

    # Accumulate energy (using np.add.at for index accumulation)
    imfs_energy = imfs_env ** 2
    for b in range(num_batches):
        for i in range(num_imfs):
            np.add.at(
                spectrum[b, :, :],
                (time_idx[0,0,:], freq_idx[b,i,:]),
                imfs_energy[b,i,:]
            )

    # Remove overflow bin and reshape
    spectrum = spectrum[..., :-1].reshape(original_shape + (time_bins, freq_bins))

    # Generate axis labels
    time_axis = (np.arange(time_bins) * time_scale) / fs + (L / fs if time_range else 0)
    freq_axis = np.arange(freq_bins) * freq_res + freq_min

    return spectrum, time_axis, freq_axis


def plot_hilbert(
    signal: np.ndarray,
    analytical_signal: Optional[np.ndarray] = None,
    return_figure: bool = False,
) -> Optional[plt.figure]:
    """
    Plot the Hilbert transform of a signal

    :param signal: Original NumPy signal.
    :param analytical_signal: A NumPy array containing the analytical signal obtained from the Hilbert transform.
    :param return_figure: Whether to return the figure.
    :return: The plot figure or None.
    """
    # Determine whether the Hilbert transform needs to be calculated
    if analytical_signal is None:
        analytical_signal = hilbert_transform(signal)

    # Extract real and imaginary parts
    real_part = hilbert_real(analytical_signal)
    imaginary_part = hilbert_imaginary(analytical_signal)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6))

    # plot original signal
    axes[0].plot(signal, color="k")
    axes[0].set_title("Original Signal")
    axes[0].grid(which="major", color="gray", linestyle="--", lw=0.5, alpha=0.8)

    # plot real and imaginary parts of hilbert transform
    axes[1].plot(real_part, label="Real Part")
    axes[1].plot(imaginary_part, label="Imaginary Part")
    axes[1].set_title("Hilbert Transform")
    axes[1].grid(which="major", color="gray", linestyle="--", lw=0.5, alpha=0.8)
    axes[1].legend()

    if return_figure is True:
        return fig


def plot_hilbert_complex_plane(
    analytical_signal: np.ndarray, return_figure: bool = False
) -> plt.figure:
    """
    Plot the Hilbert transform of a signal on the complex plane.

    :param analytical_signal: NumPy array containing the analytical signal (Hilbert transform of the original).
    :param return_figure: Whether to return the figure.
    :return: The plot figure or None.
    """
    # Extract real and imaginary parts from the analytical signal
    real_part = hilbert_real(analytical_signal)
    imaginary_part = hilbert_imaginary(analytical_signal)

    # Plotting on the complex plane
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.figure(figsize=(10, 6))
    ax.plot(
        real_part,
        imaginary_part,
        "royalblue",
        label="Analytical Signal (Hilbert Transform)",
    )
    # Mark some points to see the trend
    ax.scatter(real_part[::50], imaginary_part[::50], color="tomato")
    ax.set_title("Hilbert Transform on the Complex Plane")
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.grid(which="major", color="gray", linestyle="--", lw=0.5, alpha=0.8)
    # Ensure the scale of both axes is the same
    ax.axis("equal")
    ax.legend()

    if return_figure is True:
        return fig


if __name__ == "__main__":
    """Demo"""
    # 1 second duration, 500 samples
    t = np.linspace(0, 1, 500, endpoint=False)
    frequency = 5  # 5 Hz cosine wave
    cosine_signal = np.cos(2 * np.pi * frequency * t)
    plot_hilbert(signal=cosine_signal, analytical_signal=None)

    plt.show()
