# -*- coding: utf-8 -*-
"""
Created on 2025/02/01 22:30:40
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from numpy import floating
from scipy.stats import mode

from typing import Any, Optional, Tuple, Union

from pysdkit._faemd.extrema import extrema
from pysdkit.utils import simple_moving_average


class FAEMD(object):
    """
    Fast and Adaptive Empirical Mode Decomposition

    Thirumalaisamy, Mruthun R., and Phillip J. Ansell.
    “Fast and Adaptive Empirical Mode Decomposition for Multidimensional, Multivariate Signals.”
    IEEE Signal Processing Letters, vol. 25, no. 10, Institute of Electrical and Electronics Engineers (IEEE),
    Oct. 2018, pp. 1550–54, doi:10.1109/lsp.2018.2867335.

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd

    also see: `EMD`, `EEMD`, `REMD` and `CEEMDAN`.
    """

    def __init__(
        self,
        max_imfs: Optional[int],
        tol: Optional[float] = None,
        window_type: Optional[int] = 0,
    ) -> None:
        """
        Compared to the `EMD` algorithm, `FAEMD3D` requires simpler parameters to be specified and is faster
        :param max_imfs:The number of IMFs to be extracted
        :param tol: The threshold for loop stopping in an iterative decomposition
        :param window_type: Sliding window type using smoothing algorithm
        """
        self.max_imfs = max_imfs
        if max_imfs < 1:
            raise ValueError("`max_imfs` must be a positive integer")
        self.tol = tol

        # This parameter needs to be further tested.
        self.window_type = window_type
        if self.window_type not in [0, 1, 2, 3, 4, 5, 6]:
            raise ValueError("`window_type` must be 0, 1, 2, 3, 4, 5, 6")

        # Saving imfs and residue for external references
        self.imfs = None
        self.residue = None

    def __call__(
        self,
        signal: np.ndarray,
        return_all: bool = False,
        max_imfs: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(
            signal=signal, return_all=return_all, max_imfs=max_imfs
        )

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Fast and Adaptive Empirical Mode Decomposition (FAEMD3D)"

    def _get_tol(self, signal: np.ndarray) -> float:
        """Get the tolerance parameter for FAEMD3D"""

        # When the `tol` variable defaults to None
        if self.tol is None:
            tol = min(np.sqrt(np.mean(signal**2)), 1) * 0.001
            return tol

        # Returns the user-specified parameter
        return self.tol

    def filter_size1D(self, imax: np.ndarray, imin: np.ndarray):
        """
        To determine the window size for order statistics filtering of a signal.
        The determination of the window size is based on the work of Bhuiyan et al
        """

        # Example of calculating the difference between extreme points
        edge_len_max = np.diff(np.sort(imax))
        edge_len_min = np.diff(np.sort(imin))

        # Window size calculations
        d1 = np.min([np.min(edge_len_max), np.min(edge_len_min)])
        d2 = np.max([np.min(edge_len_max), np.min(edge_len_min)])
        d3 = np.min([np.max(edge_len_max), np.max(edge_len_min)])
        d4 = np.max([np.max(edge_len_max), np.max(edge_len_min)])
        d5 = (d1 + d2 + d3 + d4) / 4
        concat = np.concatenate((edge_len_min, edge_len_max))
        d6 = np.median(concat)
        d7 = mode(concat)[0]

        windows = np.array([d1, d2, d3, d4, d5, d6, d7])

        # making sure w_size is an odd integer
        windows = 2 * np.floor(windows / 2) + 1

        # Traverse the window array to normalize its minimum value
        for t in range(7):
            if windows[self.window_type] < 3:
                windows[self.window_type] = 3

        return windows

    def sift(self, H: np.ndarray, w_sz: Union[float, np.ndarray]) -> np.ndarray:
        """Perform an iteration of the EMD algorithm"""

        # Envelope Generation
        Env_max, Env_min = self.OSF(H=H, w_sz=w_sz)

        # padding
        Env_med = env_smoothing(env_max=Env_max, env_min=Env_min, w_sz=w_sz)

        # Subtracting from residue
        H1 = H - Env_med

        return H1

    def OSF(
        self, H: np.ndarray, w_sz: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Used to generate upper and lower envelope spectra of a signal"""

        # Max envelope
        Max = self.ord_filt1(H, order="max", window_size=w_sz)
        # Min envelope
        Min = self.ord_filt1(H, order="min", window_size=w_sz)

        return Max, Min

    @staticmethod
    def ord_filt1(signal, order, window_size) -> np.ndarray:
        """1-D Rank order filter function"""

        # Pre-processing
        # Original signal size
        shape = signal.shape

        # Removing the singleton dimensions
        signal = np.squeeze(signal)

        # Length of the signal
        L = len(signal)

        # Ensure that the processed signal is always a column vector
        signal = np.reshape(signal, newshape=[L, 1])

        r = int((window_size - 1) / 2)

        # Padding boundaries
        x = np.concatenate([np.flip(signal[:r]), signal, np.flip(signal[-r:])])

        M = x.shape[0]

        y = np.zeros(x.shape)

        # Switch the order
        if order == "max":
            for m in range(r, M - r + 1):
                # Extract a window of size (2r+1) around (m)
                temp = x[(m - r) : (m + r)]
                w = np.sort(temp)
                # Select the greatest element
                y[m] = w[-1]
        elif order == "min":
            for m in range(r, M - r + 1):
                # Extract a window of size (2r+1) around (m)
                temp = x[(m - r) : (m + r)]
                w = np.sort(temp)
                # Select the smallest element
                y[m] = w[-1]
        else:
            raise ValueError("No such filering operation defined")

        f_signal = y[r:-r]

        # Restoring Signal size
        f_signal = np.reshape(f_signal, newshape=shape)

        return f_signal

    def fit_transform(
        self,
        signal: np.ndarray,
        return_all: bool = False,
        max_imfs: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """
        Execute the signal decomposition algorithm
        :param signal: The input 1D NumPy signal
        :param return_all: whether to return all results or just the IMFs
        :param max_imfs: The number of IMFs to be extracted
        :return: The IMFs of input signal
        """
        # Get the variable number and length of a signal
        signal, inputs_shape = check_inputs(signal=signal)

        seq_len, num_vars = signal.shape

        # Adjust the maximum number of modes in the decomposition
        max_imfs = self.max_imfs if max_imfs is None else max_imfs

        # Initialisations
        imfs = np.zeros(shape=(seq_len, num_vars, max_imfs))
        H1 = np.zeros(shape=(seq_len, num_vars))
        mse = np.zeros(num_vars)

        # Array storing sliding average window
        windows = np.zeros(shape=(7, max_imfs))

        # The number of sliding averages
        sift_count = np.zeros(shape=max_imfs)

        # Intrinsic mode function pointer
        imf = 0

        # Initialization of residual components of decomposition
        Residue = signal.copy()

        # Get the tolerance parameter for algorithm stopping
        tol = self._get_tol(signal)

        # 开始进行信号的迭代分解
        while imf < max_imfs - 1:

            # Initialising intermediary IMFs
            H = Residue.copy()

            # flag to control sifting loop
            sift_stop = 0

            # Combining two signals with equal weights
            Combined = np.sum(H / np.sqrt(num_vars), axis=1)

            # Obtaining extrema of combined signal
            Maxima, MaxPos, Minima, MinPos = extrema(Combined)

            # Checking whether there are too few extrema in the IMF
            if np.count_nonzero(Maxima) < 3 or np.count_nonzero(Minima) < 3:
                # Fewer than three extrema found in extrema map. Stopping now...
                break

            # Window size determination by delaunay triangulation
            windows[:, imf] = self.filter_size1D(imax=MaxPos, imin=MinPos)

            # extracting window size chosen by input parameter
            w_sz = windows[self.window_type, imf]

            # Begin sifting iteration
            while not sift_stop:
                # Incrementing sift counter
                sift_count[imf] = sift_count[imf] + 1

                # Entering parallel sift calculations
                for i in range(num_vars):
                    H1[:, i] = self.sift(H[:, i], w_sz=w_sz)

                    # Calculate mean square error
                    mse[i] = immse(a=H1[:, i], b=H[:, i])

                # Stop condition checks
                if (mse < tol).all() and sift_count[imf] != 1:
                    sift_stop = True

                H = H1

            # Storing IMFs
            imfs[:, :, imf] = H

            # Subtracting from Residual Signals
            Residue = Residue - imfs[:, :, imf]

            # Incrementing IMF counter
            imf = imf + 1

        # Checking for oversifting
        if np.any(sift_count >= 5 * np.ones(shape=max_imfs)):
            print(
                "Decomposition may be oversifted. Checking if window size increases monotonically..."
            )

            if np.any(
                np.diff(windows[self.window_type, :]) <= np.zeros(shape=max_imfs - 1)
            ):
                print("Filter window size does not increase monotonically")

        # Record residual components
        imfs[:, :, -1] = Residue

        # Adjust the channel order of the output
        imfs = check_outputs(imfs=imfs, inputs_shape=inputs_shape)

        # Saving the results
        self.imfs = imfs[:-1, :, :] if len(imfs.shape) == 3 else imfs[:-1, :]
        self.residue = Residue

        # Organising results
        if return_all is True:
            return imfs, Residue, windows, sift_count
        return imfs

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and residue from recently analysed signal
        :return: obtained IMFs and residue through EMD
        """
        if self.imfs is None or self.residue is None:
            # If the algorithm has not been executed yet, there is no result for this decomposition.
            raise ValueError(
                "No IMF found. Please, run `fit_transform` method or its variant first."
            )
        return self.imfs, self.residue

    def get_imfs_and_trend(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and trend from recently analysed signal.
        Note that this may differ from the `get_imfs_and_residue` as the trend isn't
        necessarily the residue. Residue is a point-wise difference between input signal
        and all obtained components, whereas trend is the slowest component (can be zero).
        :return: obtained IMFs and main trend through EMD
        """
        if self.imfs is None or self.residue is None:
            # There is no decomposition result for this storage yet
            raise ValueError(
                "No IMF found. Please, run `fit_transform` method or its variant first."
            )

        # Get the intrinsic mode function and residual respectively
        imfs, residue = self.get_imfs_and_residue()
        if np.allclose(residue, 0):
            return imfs[:-1].copy(), imfs[-1].copy()
        else:
            return imfs, residue


def immse(a: np.ndarray, b: np.ndarray) -> floating[Any]:
    """
    Reimplementation of MATLAB's immse function.
    Calculates the Mean Squared Error (MSE) between two arrays.

    :param a: Input array a
    :param b: Input array b
    :return: Computed Mean Squared Error value
    """
    assert a.shape == b.shape, "The shape of `a` and `b` must be identical"
    return np.mean((a - b) ** 2)


def env_smoothing(
    env_max: np.ndarray, env_min: np.ndarray, w_sz: Union[int, float]
) -> np.ndarray:
    """
    Smooth upper and lower envelope signals using moving average.

    :param env_max: Upper envelope signal
    :param env_min: Lower envelope signal
    :param w_sz: Window size for moving average (will be converted to integer)
    :return: Smoothed mean of upper and lower envelopes
    """
    # Ensure window size is integer
    w_sz = int(w_sz)

    # Apply smoothing
    env_maxs = simple_moving_average(signal=env_max, window_size=w_sz)
    env_mins = simple_moving_average(signal=env_min, window_size=w_sz)

    # Calculate mean envelope
    return (env_maxs + env_mins) / 2


def check_inputs(signal: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    """
    Check the specific shape of the input signal.

    Args:
        signal (np.ndarray): Input signal to be checked.

    Returns:
        tuple: Transposed signal and original input shape.
    """
    # Get the shape of the input signal
    inputs_shape = signal.shape
    len_shape = len(inputs_shape)

    if len_shape == 1:
        # If the input is a 1D signal, add an additional channel
        signal = signal[np.newaxis, :]
    elif len_shape == 2:
        # Input is a multivariate signal
        pass
    else:
        # Input signal is in an incorrect format
        raise ValueError(
            "The input signal must be a NumPy ndarray univariate or multivariate signal with shape [seq_len, ] or [num_vars, seq_len]"
        )

    return signal.T, inputs_shape


def check_outputs(imfs: np.ndarray, inputs_shape: Tuple) -> np.ndarray:
    """
    Check and standardize the shape and channel order of the output intrinsic mode functions (IMFs).

    Args:
        imfs (np.ndarray): Extracted IMFs to be checked.
        inputs_shape: Original shape of the input signal.

    Returns:
        np.ndarray: Reshaped and reordered IMFs.
    """
    len_shape = len(inputs_shape)

    if len_shape == 1:
        # Original input was a univariate signal
        imfs = imfs[:, 0, :]
        imfs = np.transpose(imfs, axes=[1, 0])
    else:
        # Original input was a multivariate signal
        imfs = np.transpose(imfs, axes=[2, 0, 1])
    return imfs


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from pysdkit.data import test_emd, test_multivariate_signal
    from pysdkit.plot import plot_IMFs

    faemd = FAEMD(max_imfs=3)

    time, signal = test_emd()

    IMFs = faemd.fit_transform(signal)

    plot_IMFs(signal, IMFs)
    plt.show()

    time, signal = test_multivariate_signal()
    IMFs = faemd.fit_transform(signal)

    plot_IMFs(signal, IMFs)

    sum_IMFs = np.sum(IMFs, axis=0).T
    print(sum_IMFs.shape)
    for i in range(2):
        print(np.allclose(sum_IMFs[i], signal[i]))
    plt.show()
