# -*- coding: utf-8 -*-
"""
Created on 2025/02/11 21:42:45
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from matplotlib import pyplot as plt

from typing import Optional

from pysdkit.utils import fft, fftshift


def plot_signal(
    time: np.array,
    signal: np.array,
    spectrum: Optional[bool] = False,
    color: str = "royalblue",
    save: bool = False,
    dpi: int = 128,
    fontsize: int = 12,
) -> plt.figure:
    """
    Plot and optionally save an amplitude modulated (AM) signal with time on the x-axis and amplitude on the y-axis.

    :param time: Array of time points corresponding to the signal.
    :param signal: Array containing the signal data to be plotted.
    :param spectrum: bool, Whether to draw the spectrum signal of fast Fourier transform at the same time
    :param color: Color of the plot line.
    :param save: Boolean flag to indicate whether the plot should be saved to a file.
    :param dpi: Dots per inch (resolution) of the figure, if saved.
    :param fontsize: Font size of the axis labels.
    :return: The figure object containing the plot.
    """

    # Determine the number of input signals
    n_dim = len(signal.shape)

    if n_dim == 1:
        # Whether to perform fast Fourier transform to draw frequency domain features
        if spectrum is False:
            # Create a figure and a single subplot
            fig, ax = plt.subplots(figsize=(8, 3))

            # Plot the signal against time with specified line color
            ax.plot(time, signal, color=color)

            # Set the x-axis label with specified font size
            ax.set_xlabel("Time (seconds)", fontsize=fontsize)

            # Set the y-axis label with default font size
            ax.set_ylabel("Amplitude", fontsize=fontsize)

            # Enable grid for better readability
            ax.grid(True)
        else:
            fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
            # Plot the signal against time with specified line color
            ax[0].plot(time, signal, color=color)

            # Set the x-axis label with specified font size
            ax[0].set_xlabel("Time (seconds)", fontsize=fontsize)

            # Set the y-axis label with default font size
            ax[0].set_ylabel("Time Values", fontsize=fontsize)

            # Plot the frequency domain features after fast Fourier transform
            ax[1].plot(time, np.abs(fftshift(fft(signal))), color=color)

            # Set the x-axis label with specified font size
            ax[1].set_xlabel("Frequency", fontsize=fontsize)

            # Set the y-axis label with default font size
            ax[1].set_ylabel("Amplitude", fontsize=fontsize)

            # Enable grid for better readability
            ax[0].grid(True), ax[1].grid(True)

    elif n_dim == 2:
        # Determining the number of multivariate signals
        n_vars = signal.shape[0]

        # Whether to perform fast Fourier transform to draw frequency domain features
        if spectrum is False:
            # Create a figure and a single subplot
            fig, ax = plt.subplots(
                nrows=n_vars, figsize=(10, 1 + 2 * n_vars), sharex=True
            )

            for i in range(n_vars):
                # Plot the signal against time with specified line color
                ax[i].plot(time, signal[i, :], color=color)

                # Set the y-axis label with default font size
                ax[i].set_ylabel("Amplitude", fontsize=fontsize)

                # Enable grid for better readability
                ax[i].grid(True)

            # Set the x-axis label with specified font size
            ax[-1].set_xlabel("Time (seconds)", fontsize=fontsize)
        else:
            # Create a figure and a single subplot
            fig, ax = plt.subplots(
                nrows=n_vars, ncols=2, figsize=(15, 1 + 1.6 * n_vars), sharex=True
            )

            for i in range(n_vars):
                # Plot the signal against time with specified line color
                ax[i, 0].plot(time, signal[i, :], color=color)
                ax[i, 1].plot(time, np.abs(fftshift(fft(signal[i, :]))), color=color)

                # Set the y-axis label with default font size
                ax[i, 0].set_ylabel("Time Values", fontsize=fontsize)
                ax[i, 1].set_ylabel("Amplitude", fontsize=fontsize)

                # Enable grid for better readability
                ax[i, 0].grid(True)
                ax[i, 1].grid(True)

            # Set the x-axis label with specified font size
            ax[-1, 0].set_xlabel("Time (seconds)", fontsize=fontsize)
            ax[-1, 1].set_xlabel("Frequency", fontsize=fontsize)

    else:
        raise ValueError("Wrong input signal, please input the 1D signal.")

    # Check if the figure needs to be saved
    if save:
        # Save the figure with the specified dpi and bounding box tightly around the figure
        fig.savefig("generate_signal.jpg", dpi=dpi, bbox_inches="tight")

    # Return the figure object
    return fig
