# -*- coding: utf-8 -*-
"""
Created on Sat Mar 4 21:58:54 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
Code taken from https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EMD.py
"""
import numpy as np
from numpy import ndarray
from scipy.interpolate import interp1d
from typing import Optional, Tuple

from pysdkit.emd import akima, cubic, pchip, cubic_hermite, cubic_spline_3pts
from pysdkit.emd import find_extrema_parabol, find_extrema_simple
from pysdkit.emd import prepare_points_parabol, prepare_points_simple

from pysdkit.utils import get_timeline, normalize_signal, common_dtype


class EMD(object):
    """
    Empirical Mode Decomposition

    Huang, Norden E., et al.
    "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis."
    Proceedings of the Royal Society of London.
    Series A: mathematical, physical and engineering sciences 454.1971 (1998): 903-995.

    The algorithm first interpolates the signal extreme values and averages the upper and lower envelopes to obtain the local mean of the signal,
    which can be regarded as an estimate of the low-frequency components in the signal;
    then, the low-frequency components are iteratively separated from the input signal to obtain the high-frequency (fast oscillation) components.
    This completes a screening. Repeat the screening process until all the main oscillation modes in the input signal are extracted.

    Python code: https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EMD.py
    MATLAB code: https://www.mathworks.com/help/signal/ref/emd.html
    R code: https://rdrr.io/github/PierreMasselot/Library--emdr/f/README.md
    """

    def __init__(
        self,
        spline_kind: str = "cubic",
        energy_ratio_thr: Optional[float] = 0.2,
        nbsym: int = 2,
        std_thr: Optional[float] = 0.2,
        svar_thr: Optional[float] = 0.001,
        total_power_thr: Optional[float] = 0.005,
        range_thr: Optional[float] = 0.001,
        extrema_detection: Optional[str] = "simple",
        max_imfs: Optional[int] = -1,
        max_iteration: int = 1000,
        **kwargs,
    ) -> None:
        """
        Configuration, such as threshold values, can be passed as kwargs (keyword arguments).
        :param spline_kind: The specific interpolation algorithm used to construct the upper and lower envelope spectra, optional:
                            [akima, cubic, pchip, cubic_hermite, slinear, quadratic, linear]
        :param extrema_detection: method used to finding extrema, choices: ['parabol', 'simple']
        :param nbsym:
        :param max_imfs: maximum number of the IMFs to be decomposed
        :param max_iteration: maximum number of iterations per single sifting in EMD
        :param std_thr: threshold value on standard deviation per IMF check
        :param svar_thr: threshold value on scaled variance per IMF check
        :param total_power_thr: threshold value on total power per EMD decomposition
        :param range_thr: threshold for amplitude range (after scaling) per EMD decomposition
        :param energy_ratio_thr: threshold value on energy ratio per IMF check
        :param kwargs: other parameters passed to EMD
        """
        # Interpolation algorithm and algorithm for finding extreme points used in empirical mode decomposition
        self.spline_kind = spline_kind
        self.extrema_detection = extrema_detection
        self.nbsym = nbsym
        self.MAX_ITERATION = max_iteration

        # Thresholds in Empirical Mode Decomposition
        self.std_thr, self.svar_thr = std_thr, svar_thr
        self.total_power_thr, self.range_thr = total_power_thr, range_thr
        self.energy_ratio_thr = energy_ratio_thr

        # Other configuration parameters
        self.DTYPE = kwargs.get("DTYPE", np.float64)
        self.FIXE = kwargs.get("FIXE", 0)
        self.FIXE_H = int(kwargs.get("FIXE_H", 0))

        # Saving imfs and residue for external references
        self.imfs = None
        self.residue = None

        self.max_imfs = max_imfs

    def __call__(
        self, signal, time: Optional[np.ndarray] = None, max_imfs: Optional[int] = None
    ) -> np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal, time=time, max_imfs=max_imfs)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Empirical Mode Decomposition (EMD)"

    @staticmethod
    def _check_length(signal: np.ndarray, time: Optional[np.ndarray] = None):
        """Check input timing and signal length are equal"""
        if time is not None and len(signal) != len(time):
            raise ValueError(
                f"Signal have different size: len(signal)={len(signal)}, len(time)={len(time)}"
            )

    @staticmethod
    def _check_shape(signal: np.ndarray, time: np.ndarray) -> None:
        """Check input timing and signal shape are equal"""
        if signal.shape != time.shape:
            raise ValueError(
                "Position or time array should be the same size as signal."
            )

    def find_extrema(
        self, time: np.ndarray, signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns extrema (minima and maxima) for given signal S.
        Detection and definition of the extrema depends on
        ``extrema_detection`` variable, set on initiation of EMD.
        :param time: position or time array of numpy
        :param signal: input signal of numpy ndarray
        :return: local_max_pos : numpy array, Position of local maxima
                 local_max_val : numpy array, Values of local maxima
                 local_min_pos : numpy array, Position of local minima
                 local_min_val : numpy array, Values of local minima
        """
        if self.extrema_detection == "parabol":
            return find_extrema_parabol(time=time, signal=signal)
        elif self.extrema_detection == "simple":
            # use the simple extrema detection
            return find_extrema_simple(time=time, signal=signal)
        else:
            raise ValueError(
                "Incorrect extrema detection type. Please try: 'simple' or 'parabol'."
            )

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
                DTYPE=self.DTYPE,
            )
        elif self.extrema_detection == "simple":
            return prepare_points_simple(
                time, signal, max_pos, max_val, min_pos, min_val, self.nbsym
            )
        else:
            raise ValueError(
                "Incorrect extrema detection type. Please try: 'simple' or 'parabol'."
            )

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
            return time, interp1d(extrema[0], extrema[1], kind=kind)(t).astype(
                self.DTYPE
            )

        else:
            raise ValueError("No such interpolation method!")

    def extract_max_min_spline(
        self, time: np.ndarray, signal: np.ndarray
    ) -> list[int] | tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Extracts top and bottom envelopes based on the signal,
        which are constructed based on maxima and minima, respectively.

        Based on the local maximum and minimum values of the signal,
        the upper envelope and the lower envelope are constructed.
        This method is one of the key steps in constructing IMF,
        because IMF is obtained by the difference between the original signal and its mean.

        :param time: position or time array of numpy
        :param signal: input signal of numpy ndarray
        :return: local_max_pos : numpy array, Position of local maxima
                 local_max_val : numpy array, Values of local maxima
                 local_min_pos : numpy array, Position of local minima
                 local_min_val : numpy array, Values of local minima
        """

        # Get indexes and values of extrema
        ext_res = self.find_extrema(time, signal)
        max_pos, max_val = ext_res[0], ext_res[1]
        min_pos, min_val = ext_res[2], ext_res[3]

        # Checks if there are enough extreme points to construct a valid envelope.
        # Usually at least three points are needed to construct a meaningful envelope.
        # If not, an error or other flag may be returned.
        if len(max_pos) + len(min_pos) < 3:
            return [-1] * 4

        # Extrapolation of signal (over boundaries)
        max_extrema, min_extrema = self.prepare_points(
            time, signal, max_pos, max_val, min_pos, min_val
        )

        # The processed extreme points are interpolated using the spline_points method to construct the upper and lower envelopes.
        # These envelopes are used in the subsequent IMF extraction process, especially to calculate the difference between each IMF and the signal mean.
        _, max_spline = self.spline_points(time, max_extrema)
        _, min_spline = self.spline_points(time, min_extrema)

        return max_spline, min_spline, max_extrema, min_extrema

    def check_imf(
        self,
        imf_new: np.ndarray,
        imf_old: np.ndarray,
        eMax: np.ndarray,
        eMin: np.ndarray,
    ) -> bool:
        """
        Evaluate if the current IMF (Intrinsic Mode Function) satisfies the end condition
        based on Huang's criteria, similar to the Cauchy convergence test.

        This function ensures that consecutive siftings of the signal have minimal impact,
        indicating that the IMF component has been properly extracted. The criteria include
        checking if all local maxima are positive, all local minima are negative, and various
        convergence tests based on the differences between consecutive IMFs.

        :param imf_new: np.ndarray - the newly extracted IMF in the current iteration.
        :param imf_old: np.ndarray - the previously extracted IMF from the last iteration.
        :param eMax: np.ndarray - array of values at local maxima points, used for validation.
        :param eMin: np.ndarray - array of values at local minima points, used for validation.
        :return: bool - True if the current IMF meets the stopping criteria, False otherwise.
        """

        # Check that all local maxima are positive and all local minima are negative
        if np.any(eMax[1] < 0) or np.any(eMin[1] > 0):
            return False

        # Convergence check based on the energy of the new IMF
        if np.sum(imf_new**2) < 1e-10:
            return False

        # Precompute differences between the new and old IMF
        imf_diff = imf_new - imf_old
        imf_diff_sqrd_sum = np.sum(imf_diff * imf_diff)

        # Scaled variance test
        svar = imf_diff_sqrd_sum / (max(imf_old) - min(imf_old))
        if svar < self.svar_thr:
            return True

        # Standard deviation test
        std = np.sum((imf_diff / imf_new) ** 2)
        if std < self.std_thr:
            return True

        # Energy ratio test
        energy_ratio = imf_diff_sqrd_sum / np.sum(imf_old * imf_old)
        if energy_ratio < self.energy_ratio_thr:
            return True

        return False

    def end_condition(self, signal: np.ndarray, IMF: np.ndarray) -> bool:
        """
        Evaluate whether the Empirical Mode Decomposition (EMD) process should terminate.
        The process stops when either the absolute amplitude of the residue is below a
        threshold or the mean absolute difference of the residue is below another threshold.

        This function ensures that the decomposition stops when further significant
        intrinsic mode functions (IMFs) cannot be extracted reliably due to the minimal
        variation in the remaining signal.

        :param signal: np.ndarray - The original signal on which EMD was performed.
        :param IMF: np.ndarray - A 2D array containing all extracted IMFs, where each row represents an IMF.
        :return: bool - True if the EMD process should terminate, False otherwise.
        """
        # Calculate the residue from the original signal minus all extracted IMFs
        tmp = signal - np.sum(IMF, axis=0)

        # Check if the range of the residue is below the threshold
        if np.max(tmp) - np.min(tmp) < self.range_thr:
            return True

        # Check if the sum of the absolute differences of the residue is below the threshold
        if np.sum(np.abs(tmp)) < self.total_power_thr:
            return True

        return False

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

    def fit_transform(
        self,
        signal: np.ndarray,
        time: Optional[np.ndarray] = None,
        max_imfs: Optional[int] = 4,
    ) -> np.ndarray:
        """
        Signal decomposition using EMD algorithm
        :param signal: the time domain signal (1D numpy array) to be decomposed
        :param time: the time array of the input signal
        :param max_imfs: the maximum number of IMFs to extract
        :return: the decomposed results of IMFs
        """

        # Define the length of signal
        N = len(signal)

        max_imfs = self.max_imfs if max_imfs is None else max_imfs

        # The length of the sequence to be decomposed is inconsistent with the length of the input token sequence
        self._check_length(signal=signal, time=time)

        if time is None or self.extrema_detection == "simple":
            time = get_timeline(len(signal), signal.dtype)

        # Normalize T so that it doesn't explode
        time = normalize_signal(time)

        # Make sure same types are dealt
        signal, time = common_dtype(signal, time)
        self.DTYPE = signal.dtype

        # Initialize the residual signal, which is initially set to be the same as the original signal S.
        # It is used to store the remaining part after removing an IMF from the signal each time during the iteration process.
        residue = signal.astype(self.DTYPE)

        # Make sure both are the same shape
        self._check_shape(signal=signal, time=time)

        # Create arrays
        # Used to count the number of intrinsic mode functions (IMFs) that have been successfully extracted from the original signal.
        # Initialization to 0 means that no IMF has been successfully extracted.
        imfNo = 0
        # Used to record the total number of extreme values (maxima and minima) found in the current iteration, indicating an undefined state
        extNo = -1
        # This is a NumPy array used to store all extracted IMFs. New IMFs will be added to it row by row.
        IMF = np.empty((imfNo, N))  # Numpy container for IMF
        # Control when the main program stops
        finished = False

        while not finished:
            residue[:] = signal - np.sum(IMF[:imfNo], axis=0)
            imf = residue.copy()

            mean = np.zeros(N, dtype=self.DTYPE)

            # Counters
            # Record the number of iterations in the current IMF extraction process,
            # which cannot exceed the specified maximum number of iterations
            n_h = 0

            for n in range(1, self.MAX_ITERATION + 1):
                ext_res = self.find_extrema(time, imf)

                max_pos, min_pos, indzer = ext_res[0], ext_res[2], ext_res[4]

                # Record the number of extreme points and the number of zero crossing points separately
                extNo = len(min_pos) + len(max_pos)

                if extNo > 2:
                    max_env, min_env, eMax, eMin = self.extract_max_min_spline(
                        time, imf
                    )
                    mean[:] = 0.5 * (max_env + min_env)

                    imf_old = imf.copy()
                    imf[:] = imf - mean

                    # Fix number of iterations
                    if self.FIXE:
                        if n >= self.FIXE:
                            break

                    # Fix number of iterations after number of zero-crossings and extrema differ at most by one.
                    elif self.FIXE_H:
                        tmp_residue = self.find_extrema(time, imf)
                        max_pos, min_pos, ind_zer = (
                            tmp_residue[0],
                            tmp_residue[2],
                            tmp_residue[4],
                        )
                        extNo = len(max_pos) + len(min_pos)
                        nzm = len(ind_zer)

                        # If proto-IMF add one, or reset counter otherwise
                        n_h = n_h + 1 if abs(extNo - nzm) < 2 else 0

                        # STOP
                        if n_h >= self.FIXE_H:
                            break

                    # Stops after default stopping criteria are met
                    else:
                        ext_res = self.find_extrema(time, imf)
                        max_pos, _, min_pos, _, ind_zer = ext_res
                        extNo = len(max_pos) + len(min_pos)
                        nzm = len(ind_zer)

                        if imf_old is np.nan:
                            continue

                        f1 = self.check_imf(imf, imf_old, eMax, eMin)
                        f2 = abs(extNo - nzm) < 2

                        # STOP
                        if f1 and f2:
                            break

                else:  # Less than 2 ext, i.e. trend
                    finished = True
                    break

            # END OF IMF SIFTING
            IMF = np.vstack((IMF, imf.copy()))
            imfNo += 1

            if self.end_condition(signal, IMF) or imfNo == max_imfs - 1:
                # Determine whether to stop iterative decomposition based on the original input signal and the intrinsic mode function
                break

        # If the last sifting had 2 or less extrema then that's a trend (residue)
        if extNo <= 2:
            IMF = IMF[:-1]

        # Saving imfs and residue for external references
        self.imfs = IMF.copy()
        self.residue = signal - np.sum(self.imfs, axis=0)

        # If residue isn't 0 then add it to the output
        if not np.allclose(self.residue, 0):
            IMF = np.vstack((IMF, self.residue))

        return IMF
