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


def plot_hilbert(signal: np.ndarray, analytical_signal: Optional[np.ndarray] = None,
                 return_figure: bool = False) -> Optional[plt.figure]:
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
    axes[0].plot(signal, color='k')
    axes[0].set_title("Original Signal")
    axes[0].grid(which='major', color='gray', linestyle='--', lw=0.5, alpha=0.8)

    # plot real and imaginary parts of hilbert transform
    axes[1].plot(real_part, label='Real Part')
    axes[1].plot(imaginary_part, label='Imaginary Part')
    axes[1].set_title("Hilbert Transform")
    axes[1].grid(which='major', color='gray', linestyle='--', lw=0.5, alpha=0.8)
    axes[1].legend()

    if return_figure is True:
        return fig


def plot_hilbert_complex_plane(analytical_signal: np.ndarray, return_figure: bool = False) -> plt.figure:
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
    ax.plot(real_part, imaginary_part, 'royalblue', label='Analytical Signal (Hilbert Transform)')
    # Mark some points to see the trend
    ax.scatter(real_part[::50], imaginary_part[::50], color='tomato')
    ax.set_title('Hilbert Transform on the Complex Plane')
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.grid(which='major', color='gray', linestyle='--', lw=0.5, alpha=0.8)
    # Ensure the scale of both axes is the same
    ax.axis('equal')
    ax.legend()

    if return_figure is True:
        return fig


if __name__ == '__main__':
    """Demo"""
    # 1 second duration, 500 samples
    t = np.linspace(0, 1, 500, endpoint=False)
    frequency = 5  # 5 Hz cosine wave
    cosine_signal = np.cos(2 * np.pi * frequency * t)
    plot_hilbert(signal=cosine_signal, analytical_signal=None)
