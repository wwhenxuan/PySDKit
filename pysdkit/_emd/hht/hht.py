# -*- coding: utf-8 -*-
"""
Created on 2025/02/06 10:29:05
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from matplotlib import pyplot as plt

from typing import Optional, Tuple, Any

from pysdkit.plot import plot_IMFs, plot_HilbertSpectrum
from pysdkit.utils import (
    hilbert_real,
    hilbert_imaginary,
    hilbert_transform,
    hilbert_spectrum,
)
from pysdkit import EMD, REMD, EEMD, CEEMDAN
from pysdkit._emd.hht.frequency import get_envelope_frequency


class HHT(object):
    """Perform Hilbert-Huang transform on the input signal"""

    def __init__(
        self,
        algorithm: Optional[str] = "EMD",
        max_imfs: Optional[int] = -1,
    ) -> None:
        self.algorithm = algorithm
        self.max_imfs = max_imfs

        self.emd = self._get_emd()

        # Store original signal and sampling frequency
        self.signal, self.fs = None, None
        # Storing decomposition results
        self.imfs, self.imfs_env, self.imfs_freq = None, None, None

    def __call__(
        self,
        signal: np.ndarray,
        fs: Optional[float] = None,
        return_all: Optional[bool] = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Hilbert-Huang transform on the signal `x`, and return the amplitude and
        instantaneous frequency function of each intrinsic mode.

        :param signal: the input signal of Numpy 1D array.
        :param fs: Sampling frequencies in Hz.
        :param return_all: Whether to return all results

        :return: The intrinsic mode function and other information obtained by Hilbert-Huang transform
        """
        return self.fit_transform(signal, fs, return_all=return_all)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Hilbert-Huang Transform (HHT)"

    def _get_emd(self) -> EMD | REMD | EEMD | CEEMDAN:
        """选择对应的经验模态分解算法"""
        if self.algorithm == "EMD":
            return EMD(max_imfs=self.max_imfs)
        elif self.algorithm == "REMD":
            return REMD(max_imfs=self.max_imfs)
        elif self.algorithm == "EEMD":
            return EEMD(self.max_imfs)
        elif self.algorithm == "CEEMDAN":
            return CEEMDAN(self.max_imfs)
        else:
            raise ValueError("algorithm must be EMD, REMD, EEMD or CEEMDAN!")

    def save_decompsition(
        self,
        signal: np.ndarray,
        fs: float,
        imfs: np.ndarray,
        imfs_env: np.ndarray,
        imfs_freq: np.ndarray,
    ) -> None:
        """
        Record the results of this Hilbert-Huang transform

        :param imfs:
        :param signal: Input Numpy signal.
        :param fs:
        :param imfs_env:
        :param imfs_freq:

        :return: None
        """
        self.signal = signal
        self.fs = fs
        self.imfs = imfs
        self.imfs_freq = imfs_freq
        self.imfs_env = imfs_env

    def fit_transform(
        self,
        signal: np.ndarray,
        fs: Optional[float] = None,
        return_all: Optional[bool] = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Hilbert-Huang transform on the signal `x`, and return the amplitude and
        instantaneous frequency function of each intrinsic mode.

        :param signal: the input signal of Numpy 1D array.
        :param fs: Sampling frequencies in Hz.
        :param return_all: Whether to return all results

        :return: The intrinsic mode function and other information obtained by Hilbert-Huang transform
        """
        # Using the empirical mode decomposition algorithm to obtain the intrinsic mode function
        imfs = self.emd.fit_transform(signal)
        # Analyze the envelope spectrum frequency characteristics of each mode
        imfs_env, imfs_freq = get_envelope_frequency(imfs, fs=fs)

        # Save the result of this Hilbert-Huang transform
        self.save_decompsition(
            signal=signal, fs=fs, imfs=imfs, imfs_env=imfs_env, imfs_freq=imfs_freq
        )

        if return_all is True:
            return imfs, imfs_env, imfs_freq
        return imfs

    def plot_IMFs(
        self, signal: Optional[np.ndarray] = None, imfs: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """
        Visualize the decomposition results.

        :param signal: The NumPy signal to be visualized.
        :param imfs: The intrinsic mode functions obtained by decomposing the signal to be visualized.

        :return: Pyplot drawing Figure object in Matplotlib
        """

        # Get the original signal and the decomposed result
        signal = signal if signal is not None else self.signal
        imfs = imfs if imfs is not None else self.imfs

        # Handling exception information
        if signal is None:
            raise ValueError(
                "Please input the result after Hilbert-Huang transform or execute the `fit_transform` method once to record the decomposition result!"
            )

        # Visualize the decomposed intrinsic mode functions
        figure = plot_IMFs(signal, imfs, return_figure=True)

        return figure

    def hilbert_spectrum(
        self,
        imfs_env: Optional[np.ndarray] = None,
        imfs_freq: Optional[np.ndarray] = None,
        fs: Optional[float] = None,
        freq_lim: Optional[tuple[float, float]] = None,
        freq_res: Optional[float] = None,
        time_scale: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
         Compute the Hilbert spectrum H(t, f) using numpy.

        :param imfs_env: The envelope functions of all IMFs.
        :param imfs_freq: The instantaneous frequency functions.
        :param fs: Sampling frequency in Hz.
        :param freq_lim: Frequency range (min, max). Defaults to (0, fs/2).
        :param time_scale: Frequency resolution. Defaults to (freq_max - freq_min)/200.
        :param freq_res: Temporal scaling factor (Default: 1).

        :return: (spectrum, time_axis, freq_axis)
                 - spectrum : ndarray, shape (..., time_bins, freq_bins), Hilbert spectrum matrix
                 - time_axis : ndarray, 1D, Time axis labels
                 - freq_axis : ndarray, 1D, Frequency axis labels
        """
        # Read the results saved during the decomposition process
        imfs_env = imfs_env if imfs_env is not None else self.imfs_env
        imfs_freq = imfs_freq if imfs_freq is not None else self.imfs_freq
        fs = fs if fs is not None else self.fs

        # Compute the Hilbert spectrum
        spectrum, t, f = hilbert_spectrum(
            imfs_env=imfs_env,
            imfs_freq=imfs_freq,
            fs=fs,
            freq_lim=freq_lim,
            time_scale=time_scale,
            freq_res=freq_res,
        )
        return spectrum, t, f

    def plot_spectrum(
        self,
        imfs_env: Optional[np.ndarray] = None,
        imfs_freq: Optional[np.ndarray] = None,
        fs: Optional[float] = None,
        freq_lim: Optional[tuple[float, float]] = None,
        freq_res: Optional[float] = None,
        time_scale: int = 1,
    ) -> tuple[list[Any] | Any, list[Any]] | list[Any] | Any:
        """
        Obtaining and visualizing the Hilbert spectrum.

        :param imfs_env: The envelope functions of all IMFs.
        :param imfs_freq: The instantaneous frequency functions.
        :param fs: Sampling frequency in Hz.
        :param freq_lim: Frequency range (min, max). Defaults to (0, fs/2).
        :param freq_res: Frequency resolution. Defaults to (freq_max - freq_min)/200.
        :param time_scale: Temporal scaling factor (Default: 1).

        :return: The plotting results.
        """
        # Read the results saved during the decomposition process
        imfs_env = imfs_env if imfs_env is not None else self.imfs_env
        imfs_freq = imfs_freq if imfs_freq is not None else self.imfs_freq

        # Get the Hilbert spectrum
        spectrum, t, f = self.hilbert_spectrum(
            imfs_env=imfs_env,
            imfs_freq=imfs_freq,
            fs=fs,
            freq_lim=freq_lim,
            time_scale=time_scale,
            freq_res=freq_res,
        )
        return plot_HilbertSpectrum(spectrum, t, f)


if __name__ == "__main__":
    from pysdkit.data import test_hht

    t, s = test_hht()
    fs = 1000

    hht = HHT(max_imfs=4)

    imfs, imfs_env, imfs_freq = hht.fit_transform(s, fs=fs, return_all=True)
    plot_IMFs(s, imfs)
    hht.hilbert_spectrum()
    hht.plot_spectrum(imfs_env=imfs_env, imfs_freq=imfs_freq)
    plt.show()
