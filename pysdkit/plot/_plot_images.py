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


def plot_images(
    img: np.ndarray,
    spectrum: Optional[bool] = False,
    dpi: Optional[int] = 128,
    cmap: Optional[str] = "coolwarm",
    colorbar: Optional[bool] = False,
    save_figure: Optional[bool] = False,
    save_name: Optional[str] = None,
    return_figure: Optional[bool] = False,
) -> Optional[plt.Figure]:
    """
    Visualize univariate and multivariate 2D images.
    It is a packaged general interface.
    The input data `img` is a univariate image [height, width] or a multivariate image [n_vars, height, width]
    The `spectrum` variable controls whether to visualize the time domain
    The `colorbar` variable controls whether to add a color bar

    :param img: The input images,which shape are [height, width]或[n_vars, height, width]
    :param spectrum: bool, Whether to draw the spectrum image of fast Fourier transform at the same time
    :param dpi: The resolution at which the image is drawn
    :param cmap: The colormap to use, defaults is `colorwarm`
    :param colorbar: bool, whether to add a color bar to the drawn image
    :param save_figure: Whether to save the figure as an image
    :param save_name: The name of the saved image file
    :param return_figure: Whether to return the figure object
    :return: The plotting Figure from matplotlib
    """
    # Get the shape of the input image to determine whether it is a unary or multivariate image
    shape = img.shape

    if len(shape) == 2:
        # Two dimensions represent a univariate image

        # The width of the created image
        width = 5
        # Whether to plot frequency domain features via 2D Fast Fourier Transform
        if colorbar is True:
            width += 0.75

        if spectrum is True:
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(width, 10), dpi=dpi)
            # Plotting features in the spatial domain of an image
            cax_image = ax[0].imshow(img, cmap=cmap)
            ax[0].set_aspect("equal")
            # Plotting the features of the image in the frequency domain
            cax_spectrum = ax[1].imshow(np.abs(fft.fftshift(fft.fft2(img))), cmap=cmap)
            ax[1].set_aspect("equal")

            # Add color bar here
            if colorbar is True:
                fig.colorbar(cax_image, ax=ax[0], orientation="vertical", fraction=0.05)
                fig.colorbar(
                    cax_spectrum, ax=ax[1], orientation="vertical", fraction=0.05
                )

        else:
            # Do not plot frequency domain images
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=dpi)
            # Draw only the airspace image
            cax_image = ax.imshow(img, cmap=cmap)

            # Add color bar here
            if colorbar is True:
                fig.colorbar(cax_image, ax=ax, orientation="vertical", fraction=0.05)

    elif len(shape) == 3:
        # Three dimensions represent binary images
        # Get the number of channels of the image
        n_vars = shape[0]

        # The width of the created image
        width = 5
        # Whether to plot frequency domain features via 2D Fast Fourier Transform
        if colorbar is True:
            width += 0.2

        if spectrum is True:
            # Whether to plot frequency domain features via 2D Fast Fourier Transform
            fig, ax = plt.subplots(
                nrows=2, ncols=n_vars, figsize=(width * n_vars, 10), dpi=dpi
            )
            # Plotting features in the spatial domain of an image
            cax_image, cax_spectrum = None, None
            for n in range(n_vars):
                cax_image = ax[0, n].imshow(img[n], cmap=cmap)
                cax_spectrum = ax[1, n].imshow(
                    np.abs(fft.fftshift(fft.fft2(img[n]))), cmap=cmap
                )

            # Adding a colorbar
            if colorbar is True:
                fig.colorbar(
                    cax_image,
                    ax=[ax[0, i] for i in range(n_vars)],
                    orientation="vertical",
                    fraction=0.05,
                )
                fig.colorbar(
                    cax_spectrum,
                    ax=[ax[1, i] for i in range(n_vars)],
                    orientation="vertical",
                    fraction=0.05,
                )

        else:
            # Do not plot frequency domain images
            fig, ax = plt.subplots(
                nrows=1, ncols=n_vars, figsize=(5 * n_vars, 5), dpi=dpi
            )
            # Draw only the airspace image
            cax_image = None
            for n in range(n_vars):
                cax_image = ax[n].imshow(img[n], cmap=cmap)

            # Adding a colorbar
            if colorbar is True:
                fig.colorbar(
                    cax_image, ax=[ax[i] for i in range(n_vars)], fraction=0.05
                )

    else:
        raise ValueError(
            "The input shape is wrong, please input your univariate image with shape [height, width] and multivariate image with shape [n_vars, height, width]."
        )

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
            if len(shape) == 2:
                save_name = "plot_image.jpg"
            else:
                save_name = "plot_images.jpg"
            fig.savefig(save_name, dpi=dpi, bbox_inches="tight")

    # Return the figure if requested
    if return_figure is True:
        return fig


def plot_grayscale_image(
    img: np.ndarray,
    figsize: Optional[Tuple] = (5, 5),
    dpi: Optional[int] = 100,
    cmap: Optional[str] = "coolwarm",
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
    cmap: Optional[str] = "coolwarm",
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


if __name__ == "__main__":
    from pysdkit.data import test_univariate_image, test_multivariate_image
    from matplotlib import pyplot as plt

    plot_images(test_univariate_image(), spectrum=True, colorbar=True)
    plt.show()

    plot_images(test_univariate_image(), spectrum=False, colorbar=True)
    plt.show()

    plot_images(test_multivariate_image(case=(5, 6, 7)), spectrum=True, colorbar=True)
    plt.show()

    plot_images(test_multivariate_image(case=(5, 6, 7)), spectrum=False, colorbar=True)
    plt.show()
