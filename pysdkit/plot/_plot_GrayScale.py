# -*- coding: utf-8 -*-
"""
Created on 2025/02/02 16:47:10
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt

from typing import Optional, Tuple


def plot_grayscale_image(
    img: np.ndarray,
    figsize: Optional[Tuple] = (5, 5),
    dpi: Optional[int] = 100,
    cmap: Optional[str] = "gray",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize a 2D grayscale image.

    :param img: The input 2D ndarray matrix from numpy.
    :param figsize: The size of the figure.
    :param dpi: The resolution used, default is 100.
    :param cmap: The colormap used.
    :return: Figure and Axes from matplotlib.
    """
    # Create the figure object
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img, cmap=cmap)  # Visualize the image
    ax.set_aspect("equal")
    return fig, ax


def plot_grayscale_spectrum(
    img: np.ndarray,
    figsize: Optional[Tuple] = (5, 5),
    dpi: Optional[int] = 100,
    cmap: Optional[str] = "gray",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the spectrum distribution of a 2D grayscale image.

    :param img: The input 2D ndarray matrix from numpy.
    :param figsize: The size of the figure.
    :param dpi: The resolution used, default is 100.
    :param cmap: The colormap used.
    :return: Figure and Axes from matplotlib.
    """
    # Create the figure object
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Perform a 2D Fast Fourier Transform on the input image
    spectrum = np.abs(fft.fftshift(fft.fft2(img)))  # Obtain the power spectrum
    ax.imshow(spectrum, cmap=cmap)  # Visualize the image
    ax.set_aspect("equal")
    return fig, ax


def plot_grayscale():
    """同时绘制具有空域和频域的图像"""
