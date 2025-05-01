# -*- coding: utf-8 -*-
"""
Created on 2024/6/2 21:12
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from numpy import fft as f
from matplotlib import pyplot as plt
from typing import Optional, List, Tuple
from ._functions import set_themes
from ._functions import generate_random_hex_color
from pysdkit.utils import simple_moving_average, weighted_moving_average
from pysdkit.utils import gaussian_smoothing, savgol_smoothing, exponential_smoothing

# set_themes(choice='classic')


def max_min_normalization(x: np.ndarray) -> np.ndarray:
    """
    Perform min-max normalization on the input signal
    :param x: Input 1D sequence
    :return: Normalized sequence
    """
    return (x - x.min()) / (x.max() - x.min())


def plot_IMFs_amplitude_spectra(
    IMFs: np.ndarray,
    norm: Optional[bool] = True,
    smooth: Optional[str] = None,
    colors: Optional[List] = None,
    figsize: Optional[Tuple] = None,
    save_figure: bool = False,
    return_figure: bool = False,
    dpi: int = 500,
    fontsize: float = 14,
    save_name: Optional[str] = None,
) -> plt.figure:
    """
    Plot amplitude spectra of Intrinsic Mode Functions (IMFs) obtained from signal decomposition

    :param IMFs: Input Intrinsic Mode Functions
    :param norm: Whether to normalize the Fourier transform results
    :param smooth: Whether to smooth the amplitude spectra, and the method to use
    :param colors: List of colors to use for plotting
    :param figsize: Size of the figure (default is (12, 5))
    :param save_figure: Whether to save the plotted figure
    :param return_figure: Whether to return the figure object
    :param dpi: Resolution of the created figure (default is 500)
    :param fontsize: Font size for the labels and title (default is 14)
    :param save_name: Name to save the figure
    :return: Figure object (if return_figure is True)
    """
    set_themes(choice="classic")

    def get_inputs(x):
        """Return the input as is without any changes"""
        return x

    # Determine whether to apply min-max normalization
    if norm is True:
        fun = max_min_normalization
    else:
        fun = get_inputs

    # Determine which smoothing function to apply
    if smooth is None:
        # Do not apply any smoothing function
        smooth_function = get_inputs
    elif smooth == "simple":
        # Simple moving average smoothing
        smooth_function = simple_moving_average
    elif smooth == "weight":
        # Weighted moving average smoothing
        smooth_function = weighted_moving_average
    elif smooth == "gaussian":
        # Gaussian filtering smoothing
        smooth_function = gaussian_smoothing
    elif smooth == "savgol":
        # Savitzky-Golay filtering smoothing
        smooth_function = savgol_smoothing
    elif smooth == "exp":
        # Exponential smoothing
        smooth_function = exponential_smoothing
    else:
        raise ValueError("Invalid smoothing method")

    # Get the number of IMFs and the length of the signal
    channels, length = IMFs.shape

    # Simplify the sampling rate handling
    sampling_rate = length

    # Create the time series
    t = np.linspace(0, 1, sampling_rate, endpoint=False)

    # Set the colors for plotting
    if colors is None:
        colors = ["#228B22", "#BA55D3", "#FF8C00", "#4169E1", "#FF6347", "#20B2AA"]
    # Add random colors if there are not enough colors in the list
    while len(colors) <= channels:
        colors.append(generate_random_hex_color())

    # Adjust the figure size and aspect ratio
    if figsize is None:
        figsize = (12, 5)
    # Create the figure object
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Iterate through each IMF and plot its amplitude spectrum
    for i in range(channels):
        imf = IMFs[i, :]
        # Perform Fourier transform and get the amplitude spectrum
        fft_imf = np.abs(f.fft(imf))
        # Apply normalization and smoothing if specified
        fft_imf = smooth_function(fun(fft_imf))
        frequencies = f.fftfreq(len(fft_imf), 1.0 / sampling_rate)
        ax.plot(
            frequencies[: len(frequencies) // 2],
            fft_imf[: len(frequencies) // 2],
            color=colors[i],
            lw=1.8,
        )

    # Set labels and title for the plot
    ax.set_title("Amplitude Spectrum", fontsize=fontsize + 2)
    ax.set_xlabel("Frequency (Hz)", fontsize=fontsize)
    ax.set_ylabel("Amplitude", fontsize=fontsize)

    # Save the figure if requested
    saved = False
    if save_figure is True:
        if save_name is not None:
            for formate in [".jpg", ".pdf", ".png", ".bmp"]:
                if formate in save_name:
                    fig.savefig(save_name, dpi=dpi, bbox_inches="tight")
                    saved = True
                    break
            if saved is False:
                fig.savefig(save_name + ".jpg", dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig("plot_imfs.jpg", dpi=dpi, bbox_inches="tight")

    # Return the figure if requested
    if return_figure is True:
        return fig


def plot_HilbertSpectrum(
        spectrum,
        time_axis,
        freq_axis,
        energy_scale="log",
        save_spectrum=None,
        save_marginal=None,
        title=None,
):
    """
    Visualize the Hilbert spectrum, by plotting all the IMFs on a time-frequency plane.

    Parameters:
    ------------
    spectrum : Tensor, of shape ( ..., # time_bins, # freq_bins ).
        A pytorch tensor, representing the Hilbert spectrum H(t, f).
        The tensor will be on the same device as `imfs_env` and `imfs_freq`.
    time_axis : Tensor, 1D, of shape ( # time_bins )
        The label for the time axis of the spectrum.
    freq_axis : Tensor, 1D, of shape ( # freq_bins )
        The label for the frequency axis (in `freq_unit`) of the spectrum.
    energy_scale : string ('linear' or 'log')
        Specifying whether to visualize the energy in linear/log scale.
    save_spectrum : string or list of string, Optional.
        If specified, the Hilbert spectrum will be saved as a file named `save_spectrum`.
    save_maeginal : string or list of string, Optional.
        If specified, the Hilbert marginal spectrum will be saved as a file named `save_marginal`.
    title : string, optional.
        Specifying the title of the figure.
    """
    spectrum = spectrum.view(-1, spectrum.shape[-2], spectrum.shape[-1])
    time_axis = time_axis
    freq_axis = freq_axis

    for batch_idx in range(spectrum.shape[0]):

        if energy_scale == "log":
            eps = spectrum[batch_idx, :, :].max() * (1e-5)
            coutour = plt.pcolormesh(
                time_axis,
                freq_axis,
                10 * np.log10(spectrum[batch_idx, :, :].T + eps),
                shading="auto",
                cmap=plt.cm.jet,
                )
            plt.colorbar(coutour, label="energy (dB)")
        else:
            coutour = plt.pcolormesh(
                time_axis,
                freq_axis,
                spectrum[batch_idx, :, :].T,
                shading="auto",
                cmap=plt.cm.jet,
            )
            plt.colorbar(coutour, label="energy")

        plt.xlabel("time")
        plt.ylabel("frequency (Hz)")
        plt.title("Hilbert spectrum" if title is None else title)
        if save_spectrum is not None:
            plt.savefig(
                (
                    save_spectrum
                    if type(save_spectrum) is str
                    else save_spectrum[batch_idx]
                ),
                dpi=600,
            )
        plt.show()

    if save_marginal is None:
        return

    marginal = spectrum.sum(dim=-2)
    for batch_idx in range(spectrum.shape[0]):
        if energy_scale == "log":
            eps = marginal[batch_idx, :].max() * (1e-5)
            plt.plot(
                freq_axis, 10 * np.log10(marginal[batch_idx, :] + eps)
            )
            plt.ylabel("energy (dB)")
        else:
            plt.plot(freq_axis, marginal[batch_idx, :])
            plt.ylabel("energy")
        plt.xlabel("frequency (Hz)")
        plt.title("Marginal spectrum" if title is None else title + " (marginal)")
        plt.savefig(
            save_marginal if type(save_marginal) is str else save_marginal[batch_idx],
            dpi=600,
        )
        plt.show()