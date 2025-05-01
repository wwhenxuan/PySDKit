# -*- coding: utf-8 -*-
"""
Created on Sat Mar 7 12:09:42 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from typing import Optional, Tuple


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


def hilbert_spectrum(
    imfs_env: np.ndarray,
    imfs_freq: np.ndarray,
    fs: int,
    freq_lim: Optional[tuple[float, float]] = None,
    freq_res: Optional[float] = None,
    time_range: Optional[tuple[float, float]] = None,
    time_scale: Optional[int] = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Hilbert spectrum H(t, f) using numpy.

    :param imfs_env: The envelope functions of all IMFs.
    :param imfs_freq: The instantaneous frequency functions.
    :param fs: Sampling frequency in Hz.
    :param freq_lim: Frequency range (min, max). Defaults to (0, fs/2).
    :param freq_res: Frequency resolution. Defaults to (freq_max - freq_min)/200.
    :param time_range: Time range (start, end) in seconds.
    :param time_scale: Temporal scaling factor (Default: 1).

    :return: (spectrum, time_axis, freq_axis)
            - spectrum : ndarray, shape (..., time_bins, freq_bins), Hilbert spectrum matrix
            - time_axis : ndarray, 1D, Time axis labels
            - freq_axis : ndarray, 1D, Frequency axis labels
    """
    imfs_env = np.asarray(imfs_env, dtype=np.float64)
    imfs_freq = np.asarray(imfs_freq, dtype=np.float64)

    # Number of sampling points
    N = imfs_freq.shape[-1]
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
        L = int(np.clip(time_range[0] * fs, 0, N - 1))
        R = int(np.clip(time_range[1] * fs + 1, L + 1, N))
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
    imfs_energy = imfs_env**2
    for b in range(num_batches):
        for i in range(num_imfs):
            np.add.at(
                spectrum[b, :, :],
                (time_idx[0, 0, :], freq_idx[b, i, :]),
                imfs_energy[b, i, :],
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
