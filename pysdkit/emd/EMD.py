# -*- coding: utf-8 -*-
"""
Created on Sat Mar 4 21:58:54 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
Code taken from https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EMD.py
"""
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Tuple
from ._splines import akima, cubic, pchip, cubic_hermite, cubic_spline_3pts
from ..utils import get_timeline, normalize_signal, common_dtype
from ._find_extrema import find_extrema_parabol, find_extrema_simple
from ._prepare_points import prepare_points_parabol, prepare_points_simple


class EMD(object):

    def __init__(self, spline_kind: str = "cubic", nbsym: int = 2, max_iteration: int = 1000, **kwargs):
        self.spline_kind = spline_kind
        self.nbsym = nbsym
        self.MAX_ITERATION = max_iteration

        self.energy_ratio_thr = float(kwargs.get("energy_ratio_thr", 0.2))
        self.std_thr = float(kwargs.get("std_thr", 0.2))
        self.svar_thr = float(kwargs.get("svar_thr", 0.001))
        self.total_power_thr = float(kwargs.get("total_power_thr", 0.005))
        self.range_thr = float(kwargs.get("range_thr", 0.001))

        self.extrema_detection = kwargs.get("extrema_detection", "simple")  # simple, parabol
        self.DTYPE = kwargs.get("DTYPE", np.float64)
        self.FIXE = kwargs.get("FIXE", 0)
        self.FIXE_H = int(kwargs.get("FIXE_H", 0))

        # Saving imfs and residue for external references
        self.imfs = None
        self.residue = None

    def __call__(self, signal, time: Optional[np.ndarray] = None, max_imf=-1):
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal, time=time, max_imf=max_imf)

    @staticmethod
    def _check_length(signal: np.ndarray, time: Optional[np.ndarray] = None):
        """Check input timing and signal length are equal"""
        if time is not None and len(signal) != len(time):
            raise ValueError(f"Signal have different size: len(signal)={len(signal)}, len(time)={len(time)}")

    @staticmethod
    def _check_shape(signal: np.ndarray, time: np.ndarray):
        """Check input timing and signal shape are equal"""
        if signal.shape != time.shape:
            raise ValueError("Position or time array should be the same size as signal.")

    def find_extrema(self, time: np.ndarray, signal: np.ndarray):
        """
        Returns extrema (minima and maxima) for given signal S.
        Detection and definition of the extrema depends on
        ``extrema_detection`` variable, set on initiation of EMD.

        在给定的信号中找到所有的局部极大值和极小值。
        根据 extrema_detection 参数的不同，可以选择简单检测或抛物线检测方法。

        Parameters
        ----------
        T : numpy array
            Position or time array.
        S : numpy array
            Input data S(T).

        Returns
        -------
        local_max_pos : numpy array
            Position of local maxima.
        local_max_val : numpy array
            Values of local maxima.
        local_min_pos : numpy array
            Position of local minima.
        local_min_val : numpy array
            Values of local minima.
        """

        if self.extrema_detection == "parabol":
            return find_extrema_parabol(T=time, S=signal)
        elif self.extrema_detection == "simple":
            return find_extrema_simple(T=time, S=signal)
        else:
            raise ValueError("Incorrect extrema detection type. Please try: 'simple' or 'parabol'.")

    def prepare_points(self, T: np.ndarray,
                       S: np.ndarray,
                       max_pos: np.ndarray,
                       max_val: np.ndarray,
                       min_pos: np.ndarray,
                       min_val: np.ndarray,
                       ):
        if self.extrema_detection == "parabol":
            return prepare_points_parabol(T, S, max_pos, max_val, min_pos, min_val, self.nbsym, DTYPE=self.DTYPE)
        elif self.extrema_detection == "simple":
            return prepare_points_simple(T, S, max_pos, max_val, min_pos, min_val, self.nbsym)
        else:
            raise ValueError("Incorrect extrema detection type. Please try: 'simple' or 'parabol'.")

    def spline_points(self, T: np.ndarray, extrema: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs spline over given points.

        对给定的极值点使用不同的插值方法（根据 spline_kind 参数）生成样条曲线，
        这些曲线用作信号的上包络和下包络。

        Parameters
        ----------
        T : numpy array
            Position or time array.
        extrema : numpy array
            Position (1st row) and values (2nd row) of points.

        Returns
        -------
        T : numpy array
            Position array (same as input).
        spline : numpy array
            Spline array over given positions T.
        """

        kind = self.spline_kind.lower()
        t = T[np.r_[T >= extrema[0, 0]] & np.r_[T <= extrema[0, -1]]]

        if kind == "akima":
            return t, akima(extrema[0], extrema[1], t)

        elif kind == "cubic":
            if extrema.shape[1] > 3:
                return t, cubic(extrema[0], extrema[1], t)
            else:
                return cubic_spline_3pts(extrema[0], extrema[1], t)

        elif kind == "pchip":
            return t, pchip(extrema[0], extrema[1], t)

        elif kind == "cubic_hermite":
            return t, cubic_hermite(extrema[0], extrema[1], t)

        elif kind in ["slinear", "quadratic", "linear"]:
            return T, interp1d(extrema[0], extrema[1], kind=kind)(t).astype(self.DTYPE)

        else:
            raise ValueError("No such interpolation method!")

    def extract_max_min_spline(self, T: np.ndarray, S: np.ndarray
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts top and bottom envelopes based on the signal,
        which are constructed based on maxima and minima, respectively.

        基于信号的局部极大值和极小值，构造上包络和下包络。
        这个方法是构建IMF的关键步骤之一，因为IMF是通过原信号和其均值之差得到的。

        Parameters
        ----------
        T : numpy array
            Position or time array.
        S : numpy array
            Input data S(T).

        Returns
        -------
        max_spline : numpy array
            Spline spanned on S maxima.
        min_spline : numpy array
            Spline spanned on S minima.
        max_extrema : numpy array
            Points indicating local maxima.
        min_extrema : numpy array
            Points indicating local minima.
        """

        # Get indexes of extrema
        """这里进行极大值和极小值点的检测，并获得相应的信息"""
        ext_res = self.find_extrema(T, S)
        max_pos, max_val = ext_res[0], ext_res[1]
        min_pos, min_val = ext_res[2], ext_res[3]

        """
        检查是否有足够的极值点来构造有效的包络线。通常需要至少三个点来构建一个有意义的包络。
        如果不足，可能返回错误或其他标记。
        """
        if len(max_pos) + len(min_pos) < 3:
            return [-1] * 4  # TODO: Fix this. Doesn't match the signature.

        #########################################
        # Extrapolation of signal (over boundaries)
        max_extrema, min_extrema = self.prepare_points(T, S, max_pos, max_val, min_pos, min_val)

        """
        对处理后的极值点使用 spline_points 方法进行插值，构建上包络和下包络。
        这些包络线用于后续的IMF提取过程，特别是计算每一个IMF和信号均值的差异。
        """
        _, max_spline = self.spline_points(T, max_extrema)
        _, min_spline = self.spline_points(T, min_extrema)

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
        if np.sum(imf_new ** 2) < 1e-10:
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

    def fit_transform(self, signal: np.ndarray, time: Optional[np.ndarray] = None, max_imf: int = -1) -> np.ndarray:

        # Define the length of signal
        N = len(signal)

        # 待分解序列与输入标记序列的长度不一致
        self._check_length(signal=signal, time=time)

        if time is None or self.extrema_detection == "simple":
            time = get_timeline(len(signal), signal.dtype)

        # Normalize T so that it doesn't explode
        time = normalize_signal(time)

        # Make sure same types are dealt
        # print("signal", signal)
        signal, time = common_dtype(signal, time)
        # print("signal", signal)
        self.DTYPE = signal.dtype

        # print("time", time)

        # 初始化剩余信号，开始时设置为与原信号 S 相同，用于在迭代过程中存储每次从信号中去除一个 IMF 后的剩余部分。
        residue = signal.astype(self.DTYPE)

        # 确保两者的形状相同
        self._check_shape(signal=signal, time=time)

        # Create arrays
        # 用来计数已经从原始信号中成功提取的固有模态函数（IMF）的数量,初始化为0表示还没有任何IMF被成功提取
        imfNo = 0
        # 用于记录在当前迭代中找到的极值（极大值和极小值）的总数，表示一种未定义状态
        extNo = -1
        # 这是一个用于存储所有提取的IMFs的NumPy数组，后续将逐行添加新的IMF到里面
        IMF = np.empty((imfNo, N))  # Numpy container for IMF
        # 控制主程序何时停止
        finished = False

        while not finished:
            residue[:] = signal - np.sum(IMF[:imfNo], axis=0)
            imf = residue.copy()
            mean = np.zeros(N, dtype=self.DTYPE)

            # Counters
            """记录当前IMF提取过程中的迭代次数，不能超过规定的最大迭代次数self.MAX_ITERATION"""
            n_h = 0  # counts when |#zero - #ext| <=1

            for n in range(1, self.MAX_ITERATION + 1):
                ext_res = self.find_extrema(time, imf)
                max_pos, min_pos, indzer = ext_res[0], ext_res[2], ext_res[4]
                """extNo和nzm分别记录极值点的数量和零交叉点的数量"""
                extNo = len(min_pos) + len(max_pos)

                if extNo > 2:
                    max_env, min_env, eMax, eMin = self.extract_max_min_spline(time, imf)
                    mean[:] = 0.5 * (max_env + min_env)

                    imf_old = imf.copy()
                    imf[:] = imf - mean

                    # Fix number of iterations
                    if self.FIXE:
                        if n >= self.FIXE:
                            break

                    # Fix number of iterations after number of zero-crossings
                    # and extrema differ at most by one.
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

            if self.end_condition(signal, IMF) or imfNo == max_imf - 1:
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

    def get_imfs_and_residue(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and residue from recently analysed signal.

        Returns
        -------
        imfs : np.ndarray
            Obtained IMFs
        residue : np.ndarray
            Residue.

        """
        if self.imfs is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")
        return self.imfs, self.residue

    def get_imfs_and_trend(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Provides access to separated imfs and trend from recently analysed signal.
        Note that this may differ from the `get_imfs_and_residue` as the trend isn't
        necessarily the residue. Residue is a point-wise difference between input signal
        and all obtained components, whereas trend is the slowest component (can be zero).

        Returns
        -------
        imfs : np.ndarray
            Obtained IMFs
        trend : np.ndarray
            The main trend.

        """
        if self.imfs is None or self.residue is None:
            raise ValueError("No IMF found. Please, run EMD method or its variant first.")

        imfs, residue = self.get_imfs_and_residue()
        if np.allclose(residue, 0):
            return imfs[:-1].copy(), imfs[-1].copy()
        else:
            return imfs, residue


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    x = np.linspace(0, 1, 1024)
    f = np.cos(22 * np.pi * x ** 2) + 6 * x ** 2
    emd = EMD(spline_kind="akima")
    IMFs = emd.fit_transform(f, max_imf=2)
    print(IMFs.shape)
