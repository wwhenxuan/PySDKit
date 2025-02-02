# -*- coding: utf-8 -*-
"""
Created on Sat Mar 4 11:59:21 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List

from ._functions import generate_random_hex_color
from ._functions import set_themes

# set_themes(choice="plot_imfs")


def plot_IMFs(signal: np.ndarray,
              IMFs: np.ndarray,
              max_imf: int = -1,
              colors: Optional[List] = None,
              save_figure: bool = False,
              return_figure: bool = False,
              dpi: int = 128,
              fontsize: float = 14,
              spine_width: float = 2,
              labelpad: float = 10,
              save_name: Optional[str] = None) -> Optional[plt.figure]:
    """
    Visualizes the numpy array of intrinsic mode functions derived from the decomposition of a signal.
    Can be used as a generic interface for plotting.
    :param signal: The input original signal
    :param IMFs: The intrinsic mode functions obtained after signal decomposition
    :param max_imf: The number of decomposition modes to be plotted
    :param colors: List of color strings for plotting
    :param save_figure: Whether to save the figure as an image
    :param return_figure: Whether to return the figure object
    :param dpi: The resolution of the saved image
    :param fontsize: The font size of the axis labels
    :param spine_width: The width of the visible axes spines
    :param labelpad: Controls the filling distance of the y-axis coordinate
    :param save_name: The name of the saved image file
    :return: The figure object for the plot
    """
    # Set the matplotlib configs
    set_themes(choice="plot_imfs")

    # Determine the number of rows
    if max_imf == -1:
        nrows = IMFs.shape[0] + 1
    else:
        nrows = min(max_imf, IMFs.shape[0]) + 1

    # The length of the signal
    length = IMFs.shape[1]
    # Edge padding
    padding = int(length / 50)

    # Create the figure and axes
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 2 * nrows - 1), dpi=400)
    fig.tight_layout()

    # Set the colors for plotting
    if colors is None:
        colors = ['#000000', '#228B22', '#FF8C00', '#BA55D3', '#4169E1', '#FF6347', '#20B2AA']
    # Add random colors if there are not enough colors in the list
    while len(colors) <= nrows:
        colors.append(generate_random_hex_color())

    for i in range(0, nrows):
        # Plot a horizontal gray line as the x-axis
        ax[i].axhline(y=0, color='gray', linestyle='-', alpha=0.6, linewidth=spine_width)

        if i == 0:
            # Plot the original signal in black
            ax[i].plot(signal, color=colors[i])
            ax[i].set_ylabel("Signal", fontsize=fontsize, labelpad=labelpad)
        else:
            # Plot the intrinsic mode functions in other colors
            ax[i].plot(IMFs[i - 1], color=colors[i])
            ax[i].set_ylabel(f"IMF-{i - 1}", fontsize=fontsize, labelpad=labelpad)

        ax[i].set_xlim(-padding, length + padding)

        # Keep only the left spine visible
        for spine_name, spine in ax[i].spines.items():
            if spine_name != 'left':
                # Only keep the left spine visible
                spine.set_visible(False)
            else:
                # Set the width of the left spine
                spine.set_visible(True)
                spine.set_linewidth(spine_width)

        # Hide x-axis tick labels for all but the last plot
        if i != nrows - 1:
            ax[i].set_xticks([])

    # Open the bottom spine of the last axes and set its position
    ax[-1].spines['bottom'].set_position(('axes', -0.2))
    ax[-1].spines['bottom'].set_visible(True)
    ax[-1].spines['bottom'].set_linewidth(spine_width)

    # Save the figure if requested
    saved = False
    if save_figure is True:
        if save_name is not None:
            for formate in ['.jpg', '.pdf', '.png', '.bmp']:
                if formate in save_name:
                    fig.savefig(save_name, dpi=dpi, bbox_inches='tight')
                    saved = True
                    break
            if saved is False:
                fig.savefig(save_name + '.jpg', dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig("plot_imfs.jpg", dpi=dpi, bbox_inches='tight')

    # Return the figure if requested
    if return_figure is True:
        return fig

