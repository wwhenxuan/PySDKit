# -*- coding: utf-8 -*-
"""
Created on Sat Mar 4 11:59:21 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

可视化图像的函数需要再专门去写几个
就不同可视化信号的放在一起了

对分解后的频谱进行可视化
"""
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List

from pysdkit.plot._functions import generate_random_hex_color
from pysdkit.plot._functions import set_themes

COLORS = [
    "#000000",
    "#228B22",
    "#FF8C00",
    "#BA55D3",
    "#4169E1",
    "#FF6347",
    "#20B2AA",
]


def plot_IMFs(
    signal: np.ndarray,
    IMFs: np.ndarray,
    max_imfs: Optional[int] = -1,
    view: Optional[str] = "2d",
    colors: Optional[List] = None,
    save_figure: Optional[bool] = False,
    return_figure: Optional[bool] = False,
    dpi: Optional[int] = 64,
    spine_width: float = 2,
    labelpad: float = 10,
    save_name: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Visualizes the numpy array of intrinsic mode functions derived from the decomposition of a signal.
    Can be used as a generic interface for plotting.

    You can choose to visualize on a 2D plane or in 3D space.

    The 2D plane visualization is more intuitive,
    while the 3D visualization can better reflect the size relationship between the decomposed modes.

    :param signal: The input original signal
    :param IMFs: The intrinsic mode functions obtained after signal decomposition
    :param max_imfs: The number of decomposition modes to be plotted
    :param view: The view of the figure, choice ["2d", "3d"]
    :param colors: List of color strings for plotting
    :param save_figure: Whether to save the figure as an image
    :param return_figure: Whether to return the figure object
    :param dpi: The resolution of the saved image
    :param spine_width: The width of the visible axes spines
    :param labelpad: Controls the filling distance of the y-axis coordinate
    :param save_name: The name of the saved image file
    :return: The figure object for the plot
    """
    # Get the selected visualization
    view = view.lower()

    # Here you need to determine the dimension of the function input and then select the function to use
    # Judging whether the input signal is multivariate by its dimension
    shape = signal.shape

    if view == "2d":
        # 在2D平面上进行可视化
        if len(shape) == 1:
            # Plotting the results of the decomposition of a univariate signal on a 2D plane
            return plot_2D_IMFs(
                signal=signal,
                IMFs=IMFs,
                max_imfs=max_imfs,
                colors=colors,
                save_figure=save_figure,
                return_figure=return_figure,
                dpi=dpi,
                spine_width=spine_width,
                labelpad=labelpad,
                save_name=save_name,
            )
        elif len(shape) == 2:
            # Plotting the results of the decomposition of a multivariate signal on a 2D plane
            return plot_multi_IMFs(
                signal=signal,
                IMFs=IMFs,
                max_imfs=max_imfs,
                colors=colors,
                save_figure=save_figure,
                return_figure=return_figure,
                dpi=dpi,
                spine_width=spine_width,
                save_name=save_name,
            )
        else:
            # The output data is in the wrong format
            raise ValueError(
                "The shape of the input signal must be the univariate with [seq_len, ] or multivariate with [n_vars, seq_len]"
            )
    elif view == "3d":
        # Visualization in 3D space
        if len(shape) == 1:
            # Plot the results of the decomposition of a univariate signal in 3D space
            return plot_3D_IMFs(
                signal=signal,
                IMFs=IMFs,
                max_imfs=max_imfs,
                colors=colors,
                save_figure=save_figure,
                return_figure=return_figure,
                dpi=dpi,
                save_name=save_name,
            )
        elif len(shape) == 2:
            # Plotting the results of the decomposition of the multivariate signal in 3D space
            return plot_multi_3D_IMFs(
                signal=signal,
                IMFs=IMFs,
                max_imfs=max_imfs,
                colors=colors,
                save_figure=save_figure,
                return_figure=return_figure,
                dpi=dpi,
                save_name=save_name,
            )
        else:
            # The output data is in the wrong format
            raise ValueError(
                "The shape of the input signal must be the univariate with [seq_len, ] or multivariate with [n_vars, seq_len]"
            )
    else:
        raise ValueError("View must be either `2d` or `3d`")


def plot_2D_IMFs(
    signal: np.ndarray,
    IMFs: np.ndarray,
    max_imfs: Optional[int] = -1,
    colors: Optional[List] = None,
    save_figure: Optional[bool] = False,
    return_figure: Optional[bool] = False,
    dpi: Optional[int] = 64,
    fontsize: float = 14,
    spine_width: float = 2,
    labelpad: float = 10,
    save_name: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Visualizes the numpy array of intrinsic mode functions derived from the decomposition of a signal.
    Can be used as a generic interface for plotting.
    :param signal: The input original signal
    :param IMFs: The intrinsic mode functions obtained after signal decomposition
    :param max_imfs: The number of decomposition modes to be plotted
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
    if max_imfs == -1:
        n_rows = IMFs.shape[0] + 1
    else:
        n_rows = min(max_imfs, IMFs.shape[0]) + 1

    # The length of the signal
    length = IMFs.shape[1]
    # Edge padding
    padding = int(length / 50)

    # Create the figure and axes
    fig, ax = plt.subplots(nrows=n_rows, ncols=1, figsize=(10, 2 * n_rows - 1), dpi=256)
    fig.tight_layout()

    # Set the colors for plotting
    if colors is None:
        colors = COLORS

    # Add random colors if there are not enough colors in the list
    while len(colors) <= n_rows:
        colors.append(generate_random_hex_color())

    for i in range(0, n_rows):
        # Plot a horizontal gray line as the x-axis
        ax[i].axhline(
            y=0, color="gray", linestyle="-", alpha=0.6, linewidth=spine_width
        )

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
            if spine_name != "left":
                # Only keep the left spine visible
                spine.set_visible(False)
            else:
                # Set the width of the left spine
                spine.set_visible(True)
                spine.set_linewidth(spine_width)

        # Hide x-axis tick labels for all but the last plot
        if i != n_rows - 1:
            ax[i].set_xticks([])

    # Open the bottom spine of the last axes and set its position
    ax[-1].spines["bottom"].set_position(("axes", -0.2))
    ax[-1].spines["bottom"].set_visible(True)
    ax[-1].spines["bottom"].set_linewidth(spine_width)

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


def plot_3D_IMFs(
    signal: np.ndarray,
    IMFs: np.ndarray,
    max_imfs: Optional[int] = -1,
    colors: Optional[List] = None,
    save_figure: Optional[bool] = False,
    return_figure: Optional[bool] = False,
    dpi: Optional[int] = 64,
    fontsize: float = 5,
    save_name: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Visualizes the numpy array of intrinsic mode functions derived from the decomposition of a signal.
    Can be used as a generic interface for plotting.
    :param signal: The input original signal
    :param IMFs: The intrinsic mode functions obtained after signal decomposition
    :param max_imfs: The number of decomposition modes to be plotted
    :param colors: List of color strings for plotting
    :param save_figure: Whether to save the figure as an image
    :param return_figure: Whether to return the figure object
    :param dpi: The resolution of the saved image
    :param fontsize: The font size of the axis labels
    :param save_name: The name of the saved image file
    :return: The figure object for the plot
    """
    # Determine the number of rows
    if max_imfs == -1:
        n_rows = IMFs.shape[0] + 1
    else:
        n_rows = min(IMFs.shape[0], max_imfs) + 1

    # The length of the signal
    length = IMFs.shape[1]

    # Set the colors for plotting
    if colors is None:
        colors = COLORS

    # Add random colors if there are not enough colors in the list
    while len(colors) <= n_rows:
        colors.append(generate_random_hex_color())

    # Create the figure and axes
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Stacking all the signal including input and IMFs
    signals = np.vstack([signal, IMFs])

    # Create the x and y axes for 3D plotting
    x = np.flip(np.arange(n_rows))
    y = np.arange(length)

    # Set the x string label
    x_label = ["Signal"]
    for num in range(n_rows - 1):
        x_label.append(f"IMF-{num + 1}")

    # Traverse each signal segment to draw an image
    for i in range(0, n_rows):
        ax.plot(np.ones(length) * x[i], y, signals[i, :], color=colors[i], lw=0.75)

    # Set the x axes ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(x_label)

    # Set the tick params including fontsize and colors
    ax.tick_params(axis="both", which="major", labelsize=8, colors="black")

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
            fig.savefig("plot_IMFs_3D.jpg", dpi=dpi, bbox_inches="tight")

    # Return the figure if requested
    if return_figure is True:
        return fig


def plot_multi_IMFs(
    signal: np.ndarray,
    IMFs: np.ndarray,
    max_imfs: Optional[int] = -1,
    colors: Optional[List] = None,
    save_figure: Optional[bool] = False,
    return_figure: Optional[bool] = False,
    dpi: Optional[int] = 64,
    fontsize: float = 8,
    spine_width: float = 2,
    save_name: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plotting a multivariate signal and its decomposed intrinsic mode functions
    :param signal: The input original signal
    :param IMFs: The intrinsic mode functions obtained after signal decomposition
    :param max_imfs: The number of decomposition modes to be plotted
    :param colors: List of color strings for plotting
    :param save_figure: Whether to save the figure as an image
    :param return_figure: Whether to return the figure object
    :param dpi: The resolution of the saved image
    :param fontsize: The font size of the axis labels
    :param save_name: The name of the saved image file
    :param spine_width: The width of the visible axes spines
    :return: The figure object for the plot
    """

    # Set the matplotlib configs
    set_themes(choice="plot_imfs")

    # Get the number of elements and signal length of a multi-element signal
    n_vars, seq_len = signal.shape

    # Edge padding
    padding = int(seq_len / 50)

    # Determine the number of rows
    if max_imfs == -1:
        n_rows = IMFs.shape[0] + 1
    else:
        n_rows = min(max_imfs, IMFs.shape[0]) + 1

    # Reconstruct the input signal
    signals = np.zeros(shape=(n_rows, seq_len, n_vars))
    signals[0, :, :] = signal.transpose(1, 0)
    signals[1:, :, :] = IMFs

    # Set the colors for plotting
    if colors is None:
        colors = COLORS
    # Add random colors if there are not enough colors in the list
    while len(colors) <= n_rows:
        colors.append(generate_random_hex_color())

    # Create the figure and axes for multi-plotting
    fig, ax = plt.subplots(
        nrows=n_rows, ncols=n_vars, figsize=(3 * n_vars, 1.5 * n_rows - 1), dpi=256
    )
    fig.tight_layout()

    # Start drawing the signal
    for row in range(n_rows):
        # Iterate over each row and plot the original signal and the intrinsic mode function
        for col in range(n_vars):
            # Traverse each column to draw each element signal and its decomposition result
            ax[row, col].axhline(
                y=0, color="gray", linestyle="-", alpha=0.6, linewidth=spine_width
            )

            # Plotting the signal
            ax[row, col].plot(signals[row, :, col], color=colors[row], lw=0.75)

            # Set the range of the signal
            ax[row, col].set_xlim(-padding, seq_len + padding)

            # Adjust the size of the axis scale
            ax[row, col].tick_params(axis="both", which="major", labelsize=8)

            # Keep only the left spine visible
            for spine_name, spine in ax[row, col].spines.items():
                if spine_name != "left":
                    # Only keep the left spine visible
                    spine.set_visible(False)
                else:
                    # Set the width of the left spine
                    spine.set_visible(True)
                    spine.set_linewidth(spine_width)

            # Hide x-axis tick labels for all but the last plot
            if row != n_rows - 1:
                ax[row, col].set_xticks([])

    for col in range(n_vars):
        # Open the bottom spine of the last axes and set its position
        ax[-1, col].spines["bottom"].set_position(("axes", -0.1))
        ax[-1, col].spines["bottom"].set_visible(True)
        ax[-1, col].spines["bottom"].set_linewidth(spine_width)

    # Setting the y label
    for row in range(n_rows):
        if row == 0:
            ax[row, 0].set_ylabel("Signal", fontsize=fontsize)
        else:
            ax[row, 0].set_ylabel(f"IMF-{row - 1}", fontsize=fontsize)

    # Setting the number of variations
    for col in range(n_vars):
        ax[0, col].set_title(f"Var-{col}", fontsize=fontsize + 1)

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


def plot_multi_3D_IMFs(
    signal: np.ndarray,
    IMFs: np.ndarray,
    max_imfs: Optional[int] = -1,
    colors: Optional[List] = None,
    save_figure: Optional[bool] = False,
    return_figure: Optional[bool] = False,
    dpi: Optional[int] = 128,
    save_name: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot the results of multivariate signal decomposition in 3D
    :param signal: The input original signal
    :param IMFs: The intrinsic mode functions obtained after signal decomposition
    :param max_imfs: The number of decomposition modes to be plotted
    :param colors: List of color strings for plotting
    :param save_figure: Whether to save the figure as an image
    :param return_figure: Whether to return the figure object
    :param dpi: The resolution of the saved image
    :param save_name: The name of the saved image file
    :return: The figure object for the plot
    """

    # Determine the number of rows
    if max_imfs == -1:
        n_rows = IMFs.shape[0] + 1
    else:
        n_rows = min(IMFs.shape[0], max_imfs) + 1

    # Get the number of elements and signal length of a multi-element signal
    n_vars, seq_len = signal.shape

    # Reconstruct the input signal
    signals = np.zeros(shape=(n_rows, seq_len, n_vars))
    signals[0, :, :] = signal.transpose(1, 0)
    signals[1:, :, :] = IMFs

    # Set the colors for plotting
    if colors is None:
        colors = COLORS
    # Add random colors if there are not enough colors in the list
    while len(colors) <= n_rows:
        colors.append(generate_random_hex_color())

    # Create the figure and axes
    fig = plt.figure(figsize=(5 * n_vars, 6), dpi=200)

    # Adjust the ax object by using a list
    axes = [
        fig.add_subplot(100 + n_vars * 10 + i, projection="3d")
        for i in range(1, n_vars + 1)
    ]

    # Create the x and y axes for 3D plotting
    x = np.flip(np.arange(n_rows))
    y = np.arange(seq_len)

    # Set the x string label
    x_label = ["Signal"]
    for num in range(n_rows - 1):
        x_label.append(f"IMF-{num + 1}")

    # 遍历每一个维度绘制信号
    for col in range(n_vars):
        # 遍历原始信号和分解的本征模态函数
        for row in range(n_rows):
            axes[col].plot(
                np.ones(seq_len) * x[row],
                y,
                signals[row, :, col],
                color=colors[row],
                lw=0.75,
            )

        # Adjust the size of the axis scale
        axes[col].tick_params(axis="both", which="major", labelsize=8)

        # Set the x axes ticks and labels
        axes[col].set_xticks(x)
        axes[col].set_xticklabels(x_label)

    # Setting the number of variations
    for col in range(n_vars):
        axes[col].set_title(f"Var-{col}", fontsize=15)

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


if __name__ == "__main__":
    from pysdkit.data._generator import test_multivariate_1D_1
    from pysdkit import EWT, MVMD
    from matplotlib import pyplot as plt

    time, signal = test_multivariate_1D_1()

    print(signal.shape)

    mvmd = MVMD(alpha=100, K=2, tau=0.0)
    IMFs = mvmd(signal=signal)

    print(IMFs.shape)

    plot_multi_3D_IMFs(signal=signal, IMFs=IMFs)
    plt.show()
