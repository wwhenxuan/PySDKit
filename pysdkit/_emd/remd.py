# -*- coding: utf-8 -*-
"""
Created on 2025/02/05 13:15:49
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Tuple

from scipy.interpolate import interp1d
from scipy.stats import kurtosis

from pysdkit._emd import (
    akima,
    cubic,
    pchip,
    cubic_hermite,
    cubic_spline_3pts,
    prepare_points_parabol,
    prepare_points_simple,
)
from pysdkit._emd import find_extrema_parabol, find_extrema_simple
from pysdkit.utils import get_timeline, normalize_signal


class REMD(object):
    """
    Robust Empirical Mode Decomposition

    A useful adaptive signal processing tool for multi-component signal separation, non-stationary signal processing.

    The REMD is an improved empirical mode decomposition powered by soft sifting stopping criterion (SSSC).
    The SSSC is an adaptive sifting stop criterion to stop the sifting process automatically for the EMD.
    It extracts a set of mono-component signals (called intrinsic mode functions) from a temporal mixed signal.
    It can be used together with Hilbert transform (or other demodulation techniques) for advanced time-frequency analysis.

    Dandan Peng, Zhiliang Liu, Yaqiang Jin, Yong Qin.
    Improved EMD with a Soft Sifting Stopping Criterion and Its Application to Fault Diagnosis of Rotating Machinery.
    Journal of Mechanical Engineering. Accepted on Jan. 01, 2019.

    Zhiliang Liu*, Dandan Peng, Ming J. Zuo, Jiansuo Xia, and Yong Qin.
    Improved Hilbert-Huang transform with soft sifting stopping criterion and its application to fault diagnosis of wheelset bearings.
    ISA Transactions. 125: 426-444, 2022.

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/70032-robust-empirical-mode-decomposition-remd
    """

    def __init__(
        self,
        max_imfs: int = 3,
        max_iter: Optional[int] = 32,
        nbsym: int = 2,
        ssc: Optional[str] = "liu",
        extrema_detection: Optional[str] = "simple",
        spline_kind: Optional[str] = "cubic",
        ext_ratio: Optional[float] = 0.2,
    ) -> None:
        """
        Initialize REMD parameters.
        A useful adaptive signal processing tool for multi-component signal separation, non-stationary signal processing.
        
        :param max_imfs: max number of imf to be decomposed
        :param max_iter: max iteration number in a IMF sifting process
        :param nbsym:
        :param ssc: sifting stopping criterion
        :param extrema_detection: method used to finding extrema, choices: ['parabol', 'simple']
        :param spline_kind: The specific interpolation algorithm used to construct the upper and lower envelope spectra, optional:
                            [akima, cubic, pchip, cubic_hermite, slinear, quadratic, linear]
        :param ext_ratio:
        """
        self.max_imfs, self.max_iter = max_imfs, max_iter
        self.nbsym = nbsym
        # Stopping criteria
        self.ssc = ssc
        # How to find extreme points
        self.extrema_detection = extrema_detection

        self.spline_kind = spline_kind
        self.ext_ratio = ext_ratio

        # Record the decomposition process and results
        self.imfs, self.iterNum, self.fvs = None, None, None

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Robust Empirical Mode Decomposition (REMD)"

    def init_imfs(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The IMFs obtained from the initial decomposition and the variables recording the decomposition process"""
        # Recording the IMFs
        imfs = np.zeros(shape=(self.max_imfs, seq_len))
        # Record the number of decompositions for each mode
        iterNum = np.zeros(self.max_imfs)
        fvs = np.zeros(shape=(self.max_imfs, self.max_iter))

        return imfs, iterNum, fvs

    def stop_emd(self, time: np.ndarray, signal: np.ndarray, x_energy: float) -> bool:
        """Check whether there are enough (3) extrema to continue the decomposition"""
        # Get various extreme points of the input signal
        indmax, local_max_val, indmin, local_min_val, indzer = self.extr(
            time=time, signal=signal
        )
        peak = len(indmin) + len(indmax)
        ratio = np.sum(signal**2) / x_energy

        # Is it possible to set a threshold here?  ---> Not yet
        stop_flag = peak < 3 or ratio < 0.001

        return stop_flag

    def extr(
        self, time: np.ndarray, signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts the indices of extrema for the input signal
        Copied from _emd toolbox by G.Rilling and P.Flandrin
        https://perso.ens-lyon.fr/patrick.flandrin/emd.html
        """
        # Get the position of the extreme point by calling the function to find the extreme point
        if self.extrema_detection == "parabol":
            local_max_pos, local_max_val, local_min_pos, local_min_val, indzer = (
                find_extrema_parabol(time=time, signal=signal)
            )
        elif self.extrema_detection == "simple":
            local_max_pos, local_max_val, local_min_pos, local_min_val, indzer = (
                find_extrema_simple(time=time, signal=signal)
            )
        else:
            raise ValueError("`find_extrema` must be 'parabol' or 'simple'")

        return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer

    def extend(
        self, signal: np.ndarray, indmin: np.ndarray, indmax: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extend original data to refrain end effect
        Modified on _emd by G.Rilling and P.Flandrin
        https://perso.ens-lyon.fr/patrick.flandrin/emd.html

        :param signal: Input signal to be processed
        :param indmin: The position array of the maximum points of the input signal
        :param indmax: An array of positions of the minimum points of the input signal
        """
        # do not extend x
        if self.ext_ratio == 0:
            ext_indmin = indmin
            ext_indmax = indmax
            ext_x = signal.copy()
            cut_index = np.array([0, len(signal)])
            return ext_indmin, ext_indmax, ext_x, cut_index

        # number of extrema in extending end
        nbsym = np.ceil(self.ext_ratio * len(indmax)) - 1
        xlen = len(signal)

        # 创建时间戳数组
        time = np.arange(0, xlen)

        # Boundary conditions for interpolations:
        end_max = len(indmax)
        end_min = len(indmin)

        # left end extend first
        if indmax[0] < indmin[0]:
            # first extrema is max

            if signal[0] > signal[indmax[0]]:
                # first point > first min extrema
                lmax = np.fliplr(indmax[1 : min(end_max, nbsym + 1)])
                lmin = np.fliplr(indmin[: min(end_min, nbsym)])
                lsym = indmax[0]

            else:
                # first point < first min extrema
                lmax = np.flipud(indmax[: min(end_max, nbsym)])
                lmin = np.concatenate([np.fliplr(indmin[: min(end_min, nbsym - 1)]), 1])
                lsym = 1

        else:
            # first extrema is maximum

            if signal[0] < signal[indmax[0]]:
                # first point < first maximum
                lmax = np.fliplr(indmax[: min(end_max, nbsym)])
                lmin = np.fliplr(indmin[1 : min(end_min, nbsym + 1)])
                lsym = indmax[0]

            else:
                # first point > first minimum
                lmax = np.concatenate([np.fliplr(indmax[: min(end_max, nbsym - 1)]), 1])
                lmin = np.fliplr(indmin[: min(end_min, nbsym)])
                lsym = 1

        # right end second
        if indmax[-1] < indmin[-1]:
            # last extrema is minimum

            if signal[-1] < signal[indmax[-1]]:
                # last point < last maximum
                rmax = np.fliplr(indmax[max(end_max - nbsym + 1, 0) :])
                rmin = np.fliplr(indmin[max(end_min - nbsym, 0) : -1])
                rsym = indmin[-1]

            else:
                # last point > last maximum
                rmax = np.concatenate(
                    [np.array([xlen]), np.fliplr(indmax[max(end_max - nbsym + 2, 0) :])]
                )
                rmin = np.fliplr(indmin[max(end_min - nbsym + 1, 0) :])
                rsym = xlen

        else:
            # last extrema is maximum

            if signal[-1] > signal[indmax[-1]]:
                # last point > last minimum
                rmax = np.fliplr(indmax[max(end_max - nbsym, 0) : -1])
                rmin = np.fliplr(indmin[max(end_min - nbsym + 1, 0) :])
                rsym = indmax[-1]

            else:
                # last point < last minimum
                rmax = np.fliplr(indmax[max(end_max - nbsym + 1, 0) :])
                rmin = np.concatenate(
                    [np.array([xlen]), np.fliplr(indmin[max(end_min - nbsym + 2, 0) :])]
                )
                rsym = xlen

        tl_min = 2 * time[lsym] - time[lmin]
        tl_max = 2 * time[lsym] - time[lmax]

        tr_min = 2 * time[rsym] - time[rmin]
        tr_max = 2 * time[rsym] - time[rmax]

        # in case symmetrized parts do not extend enough
        if tl_min[0] > time[0] or tl_max[0] > time[0]:

            if lsym == indmax[0]:
                lmax = np.fliplr(indmax[: min(end_max, nbsym)])

            else:
                lmin = np.fliplr(indmin[: min(end_min, nbsym)])

            if lsym == 0:
                raise ValueError("bug")

            lsym = 0

        if tr_min[-1] < time[xlen] or tr_max[-1] < time[xlen]:

            if rsym == indmax[-1]:
                rmax = np.fliplr(indmax[max(end_max - nbsym + 1, 0) :])

            else:
                rmin = np.fliplr(indmin[max(end_min - nbsym + 1, 0) :])

            if rsym == xlen:
                raise ValueError("bug")

            rsym = xlen

        l_end = np.max(np.max(lmax, lmin))
        r_end = np.min(np.min(rmax, rmin))

        new_lmax = l_end - lmax
        new_lmin = l_end - lmin

        new_rmax = rsym - rmax
        new_rmin = rsym - rmin

        lx_length = l_end - lsym

        lx = np.fliplr(signal[lsym + 1 : l_end])
        rx = np.fliplr(signal[r_end : rsym - 1])

        ext_x = np.concatenate([lx, signal[lsym:rsym], rx])

        ext_indmin = np.concatenate(
            [
                new_lmin,
                indmin + lx_length - lsym + 1,
                new_rmin + lx_length - lsym + 1 + rsym,
            ]
        )
        ext_indmax = np.concatenate(
            [
                new_lmax,
                indmax + lx_length - lsym + 1,
                new_rmax + lx_length - lsym + 1 + rsym,
            ]
        )

        # Index for cutting extension of x
        cut_index = np.concatenate(
            [lx_length - lsym + 2, len(signal) + lx_length - lsym + 1]
        )

        return ext_indmin, ext_indmax, ext_x, cut_index

    def spline_points(
        self, time: np.ndarray, extrema: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs spline over given points.
        Generates spline curves using different interpolation methods (depending on the spline_kind parameter) for given extreme points.
        These curves are used as the upper and lower envelopes of the signal.

        :param time: position or time array of numpy
        :param extrema: the max_extrema and min_extrema for input signal in numpy ndarray
        :return: spline_points array of numpy ndarray
        """

        kind = self.spline_kind.lower()
        t = time[np.r_[time >= extrema[0, 0]] & np.r_[time <= extrema[0, -1]]]

        if kind == "akima":
            # Akima interpolation is known for its smoothness and is suitable for curves with rapid changes
            return t, akima(extrema[0], extrema[1], t)

        elif kind == "cubic":
            # Cubic Spline interpolation ensures that the second derivatives are continuous, resulting in a smooth curve
            if extrema.shape[1] > 3:
                return t, cubic(extrema[0], extrema[1], t)
            else:
                # Custom Cubic Spline Interpolation for 3 Data Points
                return cubic_spline_3pts(extrema[0], extrema[1], t)

        elif kind == "pchip":
            # Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) Interpolation
            return t, pchip(extrema[0], extrema[1], t)

        elif kind == "cubic_hermite":
            # Cubic Hermite Spline interpolation ensures that the first derivatives are continuous
            return t, cubic_hermite(extrema[0], extrema[1], t)

        elif kind in ["slinear", "quadratic", "linear"]:
            # Simple linear interpolation
            return time, interp1d(extrema[0], extrema[1], kind=kind)(t)

        else:
            raise ValueError("No such interpolation method!")

    def prepare_points(
        self,
        time: np.ndarray,
        signal: np.ndarray,
        max_pos: np.ndarray,
        max_val: np.ndarray,
        min_pos: np.ndarray,
        min_val: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Further processing of the maximum and minimum points of the input signal makes the upper and lower envelope spectra smoother.
        :param time: position or time array of numpy
        :param signal: input signal of numpy ndarray
        :param max_pos: numpy array, Position of local maxima.
        :param max_val: numpy array, Values of local maxima
        :param min_pos: numpy array, Position of local minima
        :param min_val: numpy array, Values of local minima
        :return: max_extrema and min_extrema for input signal in numpy ndarray
        """
        if self.extrema_detection == "parabol":
            return prepare_points_parabol(
                time,
                signal,
                max_pos,
                max_val,
                min_pos,
                min_val,
                self.nbsym,
            )
        elif self.extrema_detection == "simple":
            return prepare_points_simple(
                time, signal, max_pos, max_val, min_pos, min_val, self.nbsym
            )
        else:
            raise ValueError(
                "Incorrect extrema detection type. Please try: 'simple' or 'parabol'."
            )

    def _emd_mean(self, time: np.ndarray, signal: np.ndarray) -> Tuple[np.ndarray, int]:
        """Get the mean of the upper and lower envelope spectra and the number of all extreme points"""
        # Get extreme points
        local_max_pos, local_max_val, local_min_pos, local_min_val, indzer = self.extr(
            time=time, signal=signal
        )
        # Calculate the number of extreme points
        n_extr = len(local_max_pos) + len(local_min_pos)

        max_extrema, min_extrema = self.prepare_points(
            time, signal, local_max_pos, local_max_val, local_min_pos, local_min_val
        )

        # Get the upper and lower envelope spectra
        _, max_spline = self.spline_points(time, max_extrema)
        _, min_spline = self.spline_points(time, min_extrema)

        # Get the upper and lower envelope spectra
        m_j = np.zeros(len(signal))
        m_j[:] = 0.5 * (max_spline + min_spline)

        return m_j, n_extr

    def is_sifting_process_stop(self, time, m, s, j, fv_i) -> Tuple[bool, np.ndarray]:
        """sifting stopping criterion"""
        df = m.copy()

        # Get the maximum, minimum and zero points
        local_max_pos, local_max_val, local_min_pos, local_min_val, indzer = self.extr(
            time=time, signal=s
        )

        # The number of minimum points obtained
        lm = len(local_min_pos)
        # The number of maximum points obtained
        LM = len(local_max_pos)

        # The total number of extreme points
        nem = lm + LM

        # Number of zeros
        nzm = len(indzer)

        stop_flag_sifting_process = False
        if self.ssc == "liu":
            # local optimal iteration
            fv_i[j] = np.sqrt(np.mean(df**2)) + np.abs(kurtosis(df) - 3)
            if j >= 2 > abs(nzm - nem):
                if (fv_i[j] >= fv_i[j - 1]) and (fv_i[j - 1] >= fv_i[j - 2]):
                    stop_flag_sifting_process = True

        return stop_flag_sifting_process, fv_i[j]

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Start executing the REMD algorithm according to the input parameters
        :param signal: The input 1D signal of ndarray to be decomposed
        :return: the Intrinsic Mode Function of the input signal
        """
        # To ensure that the input signal is one-dimensional
        seq_len = signal.shape[0]

        # Initialize timestamp array
        time = get_timeline(range_max=seq_len, dtype=signal.dtype)

        # Normalize T so that it doesn't explode
        time = normalize_signal(time)

        # energy = square summation
        signal_energy = np.sum(signal**2)

        # 初始分解得到的本征模态函数以及记录分解过程的变量
        imfs, iterNum, fvs = self.init_imfs(seq_len)

        # Initialize the main loop
        i = 0

        # Copy the signal for sifting process, reserve original input as signal
        x = signal.copy()

        while i < self.max_imfs - 1 and not self.stop_emd(
            time, x, x_energy=float(signal_energy)
        ):
            # outer loop for imf selection

            # initialize variables used in imf sifting loop
            s_j = np.zeros(shape=(self.max_iter, seq_len))

            # imf sifting iteration loop
            j = 0
            stop_flag_sifting_process = 0

            s = x.copy()

            while j < self.max_iter and not stop_flag_sifting_process:
                # inner loop for sifting process

                if j == 0:
                    # 获取上下包络谱
                    m_j, n_extr = self._emd_mean(time, s)

                # force to stop iteration if number of extrema of s is smaller than 3
                if n_extr < 3:
                    break

                # The algorithm can continue to execute if the number of extreme points meets the requirements
                # Remove the mean of the envelope spectrum from the original signal
                s = s - m_j
                s_j[j, :] = s

                m_j, n_extr = self._emd_mean(time, s)
                stop_flag_sifting_process, fvs[i, :] = self.is_sifting_process_stop(
                    time, m_j, s, j, fvs[i, :]
                )

                j += 1

            # Critical Step
            opt0 = np.min(fvs[i, :j])
            # in case iteration stop for n_extr<3
            opt_IterNum = int(min(j - 1, opt0))

            # gain intrinsic mode function
            imfs[i, :] = s_j[opt_IterNum, :]

            # remove imf just obtained from input signal
            x = x - imfs[i, :]

            # record the iteration number taken by each imf sifting
            iterNum[i] = opt_IterNum

            i += 1

        # save residual in the last row of imf matrix
        imfs[i, :] = x

        return imfs


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from pysdkit.data import test_emd, test_univariate_signal
    from pysdkit.plot import plot_IMFs

    time, signal = test_univariate_signal()

    remd = REMD(max_imfs=4, ext_ratio=0.0)

    IMFs = remd.fit_transform(signal)

    plot_IMFs(signal, IMFs)
    plt.show()
