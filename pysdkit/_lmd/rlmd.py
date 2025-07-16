# -*- coding: utf-8 -*-
"""
Created on 2025/01/31 23:33:43
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Tuple


class RLMD(object):
    """
    Robust Local Mean Decomposition

    The RLMD is an improved local mean decomposition powered by a set of optimization strategies.
    The optimization strategies can deal with boundary condition, envelope estimation, and sifting stopping criterion in the LMD.
    It simultaneously extracts a set of mono-component signals (called product functions) and
    their associated demodulation signals (i.e. AM signal and FM signal) from a mixed signal,
    which is the most attracting feature comparing with other adaptive signal processing methods,
    such as the EMD. The RLMD can be used for time-frequency analysis.

    Zhiliang Liu, Yaqiang Jin, Ming J. Zuo, and Zhipeng Feng.
    Time-frequency representation based on robust local mean decomposition for multi-component AM-FM signal analysis.
    Mechanical Systems and Signal Processing. 95: 468-487, 2017.
    Liu Zhiliang (2025). Robust Local Mean Decomposition (RLMD)
    (https://www.mathworks.com/matlabcentral/fileexchange/66935-robust-local-mean-decomposition-rlmd),
    MATLAB Central File Exchange. Retrieved July 13, 2025.
    """

    def __init__(
        self,
        max_imfs: Optional[int] = 10,
        max_iter: Optional[int] = 30,
        smooth_mode: Optional[str] = "ma",
        ma_span: Optional[str] = "liu",
        ma_iter_mode: Optional[str] = "fixed",
        stop_threshold: Optional[Tuple[float, float, float]] = (0.005, 0.7, 0.005),
        extd_r: Optional[float] = 0.2,
        sifting_stopping_mode: Optional[str] = "liu",
        min_peak: Optional[int] = 3,
        min_ratio: Optional[float] = 0.001,
    ):
        """
        The above parameters are the default parameters of the algorithm.
        Please modify and adjust them according to the specific data when using them.

        :param max_imfs: Maximum number of intrinsic mode functions to be decomposed.
        :param max_iter: max iteration number in a PF sifting process.
        :param smooth_mode: ma - moving average, spline - pchip.
        :param ma_span: pdmax span method, see function ma_span.
        :param ma_iter_mode: fixed or dynamic span for iterate ma.
        :param stop_threshold: sifting stopping thresholds for Rilling's criterion.
        :param extd_r: end extension length to original data.
        :param sifting_stopping_mode: the sifting stoppling optimizaion.
        :param min_peak: When the number of remaining extreme points of the input signal is less than a certain value,
                         the algorithm stops iterating.
        :param min_ratio: When the energy of the remaining signal is less than a certain value,
                          the algorithm stops iterative updating.
        """
        self.max_imfs = max_imfs
        self.max_iter = max_iter

        self.smooth_mode = smooth_mode
        self.ma_span = ma_span
        self.ma_iter_mode = ma_iter_mode

        self.stop_threshold = stop_threshold
        self.extd_r = extd_r
        self.sifting_stopping_mode = sifting_stopping_mode

        # 与算法停止迭代更新相关的参数
        self.min_peak = min_peak
        self.min_ratio = min_ratio

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Allow instances to be called like functions"""
        return self.fit_transform(signal)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Robust Local Mean Decomposition (RLMD)"

    def _initial_inputs(
        self, signal: np.ndarray
    ) -> Tuple[
        np.ndarray | float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """"""
        # get the length of inputs signal
        nx = signal.shape[0]

        # 初始化存放中间结果的数组
        ams = np.zeros(shape=(self.max_imfs, nx))
        fms = np.zeros(shape=(self.max_imfs, nx))

        # 初始化本征模态函数
        pfs = np.zeros(shape=(self.max_imfs, nx))

        # 每轮迭代
        iterNum = np.zeros(self.max_imfs)
        fvs = np.zeros(shape=(self.max_imfs, self.max_iter))

        # energy = square summation.
        signal_energy = np.sum(signal**2)

        return signal_energy, ams, fms, pfs, iterNum, fvs

    def _stop_lmd(self, signal: np.ndarray, signal_energy: np.ndarray | float) -> bool:
        """
        LMD算法的停止准则
        Check if there are enough (3) extrema to continue the decomposition

        :param signal: 输入的原始信号或是信号的剩余分量
        :param signal_energy: 原信号的能量大小
        :return:
        """
        # 获得输入信号的极大值和极小值索引位置
        indmin, indmax, _ = extr(x=signal)

        # 计算极值点的数目
        number_peak = len(indmax) + len(indmin)

        ratio = np.sum(signal**2) / signal_energy

        # 判断是否可以停止算法
        stop = number_peak < self.min_peak or ratio < self.min_ratio

        if stop:
            return True
        return False

    def _lmd_mean_amp(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | int]:
        """
        Compute mean function and amplitude function of x in LMD

        :param x:
        :return:
        """

        # Find extremum indices
        indmin, indmax, _ = extr(x=x)

        # Total amount of extrema
        n_extr = len(indmin) + len(indmax)

        if n_extr < self.min_peak:
            m = np.array([])
            a = np.array([])
            return m, a, n_extr

        # extend original data to refrain end effect
        # ext_indmin(max) contains the end point's index
        ext_indmin, ext_indmax, ext_x, cut_index = extend(
            x=x, indmin=indmin, indmax=indmax, extd_r=self.extd_r
        )

        # preparation
        m0 = np.zeros(len(ext_x))
        a0 = np.zeros(len(ext_x))

        # TODO: 这里对这个索引数组进行了-1处理
        ind_extextr = np.sort(np.hstack([ext_indmin, ext_indmax])) - 1

        # Compute local mean and amplititude sequence
        if self.smooth_mode == "ma":
            for k in range(0, len(ind_extextr) - 1):  # TODO: 注意这里是不想需要减1
                subm1 = ind_extextr[k]
                subm2 = ind_extextr[k + 1]
                # print(subm1, subm2, len(m0), len(a0), len(ext_x))
                m0[subm1:subm2] = (ext_x[subm1] + ext_x[subm2]) / 2
                a0[subm1:subm2] = np.abs(ext_x[subm1] + ext_x[subm2]) / 2

            span, smax = self._get_best_span(ind_extextr, x=ext_x)

            # iterative moving average
            m = self._itrma(x=m0, span=span, smax=smax)
            a = self._itrma(x=a0, span=span, smax=smax)

            # cut extension
            # TODO: 这里在索引位置上进行了+1
            m = m[cut_index[0] : cut_index[1] + 1]
            a = a[cut_index[0] : cut_index[1] + 1]

        else:
            raise ValueError(f"No specifications {self.smooth_mode} for smooth_mode!")

        return m, a, n_extr

    def _itrma(self, x: np.ndarray, span: int, smax: int) -> np.ndarray:
        """
        Iterative moving average dynamic step

        :param x:
        :param span:
        :param smax:
        :return:
        """

        x = smooth(x, span=span)
        nm = len(x)

        if self.ma_iter_mode == "fixed":
            # Stick Step
            cntr = 0  # count times of moving average

            # theoretic
            max_c = np.ceil(smax / span) * 15

            k = int((span + 1) / 2)
            kmax = int(nm - (span - 1) / 2)

            while k < kmax and cntr < max_c:
                # find flat step
                if x[k] == x[k + 1]:
                    x = smooth(x, span=span)
                    cntr = cntr + 1
                    k -= 1
                k += 1

        else:
            raise ValueError(f"No specifications {self.ma_iter_mode} for ma_iter_mode!")

        return x

    def _get_best_span(
        self, ind_extr: np.ndarray, x: np.ndarray
    ) -> Tuple[np.ndarray | int, int]:
        """
        Moving average span selection.
        Return the best span of moving average

        :param ind_extr: indices of extremum of x
        :param x: the input signal to be decomposed
        :return: - span : int, Optimal moving-average span (always odd).
                 - smax : int, Maximum step between consecutive extrema.
        """
        ind_extr = np.asarray(ind_extr, dtype=int)
        x = np.asarray(x, dtype=float)

        x_extr = x[ind_extr]

        # 0 elements' indices of eql_extr are the first indices of identical maximum ,or minimum, pairs
        eql_extr = x_extr[2:] - x_extr[:-2]

        # delete the extremum indecies between two identical maximum(minimum)
        # eql_extr == 0 indicates identical values separated by one point
        to_remove = np.where(eql_extr == 0)[0] + 1  # +1 because diff shrinks length
        ind_extr = np.delete(ind_extr, to_remove)

        # Steps between successive extrema
        step_vec = ind_extr[1:] - ind_extr[:-1] + 1
        smax = int(step_vec.max())
        smean = step_vec.mean()

        if self.ma_span.lower() == "liu":
            # Probability density via histogram
            counts, bins = np.histogram(step_vec, bins="auto", density=True)
            bin_centers = bins[:-1] + np.diff(bins) / 2.0
            span_c = np.sum(bin_centers * counts * np.diff(bins))
            span_std = np.sqrt(
                np.sum((bin_centers - span_c) ** 2 * counts * np.diff(bins))
            )
            span = int(np.ceil(span_c + 3 * span_std))
        else:
            raise ValueError("Unsupported ma_span specification.")

        # Force span to be odd
        span = int(np.ceil(span))
        if span % 2 == 0:
            span += 1

        return span, smax

    def _is_sifting_stopping(
        self, a_j: np.ndarray, j: np.ndarray, fv_i: np.ndarray
    ) -> Tuple[bool, np.ndarray]:
        """
        Sifting stopping criterion of Robust Local Mean Decomposition.

        :param a_j:
        :param j:
        :param fv_i:
        :return:
        """
        # baseline is y = 1.
        base = np.ones(shape=a_j.shape)

        # df = abs(a_j - base); % difference between a_i and baseline.
        df = a_j - base

        # 初始化算法停止的标识
        stop_sifting = False

        if self.sifting_stopping_mode.lower() == "liu":
            # local optimal iteration.
            fv_i[j] = rms(df) + np.abs(kurtosis(x=df) - 3)

            if j >= 3:
                if (fv_i[j] >= fv_i[j - 1]) and (fv_i[j - 1] >= fv_i[j - 2]):
                    stop_sifting = True
                    return stop_sifting, fv_i

        else:
            raise ValueError(
                f"No specifications {self.sifting_stopping_mode} for sifting_stopping_mode!"
            )

        return stop_sifting, fv_i

    def fit_transform(self, signal: np.ndarray):
        """
        This funcion perform the local mean decompose(LMD) on the input signal, and return the product function (pfs),
        and their corresponding instantaneous amplititde(ams) and frequency modulation signal(fms).

        :param signal:
        :return:
        """
        # 初始化
        signal_energy, ams, fms, pfs, iterNum, fvs = self._initial_inputs(signal=signal)

        # Initialize Main Loop
        i = 0

        # Copy input signal to xs for sifting process, reserve original inputs as x
        xs = np.copy(a=signal)
        nx = signal.shape[0]

        # start the main loop
        while i < self.max_imfs and not self._stop_lmd(
            signal=xs, signal_energy=signal_energy
        ):

            # initialize variables used in PF sifting loop
            a_i = np.ones(nx)
            s_j = np.zeros(shape=(self.max_iter, nx))
            a_ij = np.zeros(shape=(self.max_iter, nx))

            # PF sifting iteration loop
            j = 0
            stop_sifting = False
            s = xs.copy()

            while j < self.max_imfs and not stop_sifting:
                # inner loop for sifting process

                # Compute mean function and amplitude function
                m_j, a_j, n_extr = self._lmd_mean_amp(x=s)

                # force to stop iter if number of extrema of s is smaller than 3.
                if n_extr < self.min_peak:
                    # 极值点数目过少停止算法迭代
                    print("剩余残差分量的极值点数目较少，算法停止!")
                    break

                # remove mean
                h_j = s - m_j

                # demodulate amplitude
                s = h_j / a_j

                # mulitiply every a_i
                a_i = a_i * a_j

                a_ij[j, :] = a_i
                s_j[j, :] = s

                # 判断算法的停止准则
                stop_sifting, fvs_array = self._is_sifting_stopping(
                    a_j=a_j, j=j, fv_i=fvs[i, :]
                )
                fvs[i, :] = fvs_array

                j = j + 1

            if self.sifting_stopping_mode.lower() == "liu":
                # 获得最小值位置处的索引
                opt0 = np.argmin(fvs[i, :j])

                # in case iteration stop for n_extr<3
                opt_IterNum = min(j, opt0)

            else:
                raise ValueError("No specifications for sifting_stopping_mode.")

            # save each amplitude modulation function in ams.
            ams[i, :] = a_ij[opt_IterNum, :]

            # save each pure frequency modulation function in fms.
            fms[i, :] = s_j[opt_IterNum, :]

            # gain Product Funcion.
            pfs[i, :] = a_ij[opt_IterNum, :]

            # remove PF just obtained from the input signal
            xs = xs - pfs[i, :]
            # print(pfs[i, :])

            # record the iteration times taken by of each PF sifing
            iterNum[i] = opt_IterNum

            i = i + 1

        # save residual in the last row of PFs matrix
        # pfs[i + 1, :] = xs  # FIX: IndexError
        residual = xs
        # Append the residual
        pfs = np.vstack([pfs, residual])

        return pfs


def extr(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Python implementation of the MATLAB `extr` function from the EMD toolbox.

    Extracts the indices of extrema
    Copied from emd toolbox by G.Rilling and P.Flandrin
    http://perso.ens-lyon.fr/patrick.flandrin/emd.html

    :param x: 1-D array_like Input signal.
    :return: indmin : ndarray Indices of local minima.
             indmax : ndarray Indices of local maxima.
             indzer : ndarray Indices of zero-crossings (including mid-points of flat zero segments).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    # ------------------------------------------------------------------
    # 1. Zero-crossings
    # ------------------------------------------------------------------
    # Detect sign changes between consecutive samples
    x1, x2 = x[:-1], x[1:]
    indzer = np.where(x1 * x2 < 0)[0]

    # Handle exact zeros and flat zero segments
    iz = np.where(x == 0)[0]
    if iz.size > 0:
        # Detect contiguous zero segments
        zeros = x == 0
        dz = np.diff(np.r_[0, zeros, 0])
        debz = np.where(dz == 1)[0]
        finz = np.where(dz == -1)[0] - 1

        # Middle of each flat zero segment
        midz = np.round((debz + finz) / 2).astype(int)
        indzer = np.unique(np.concatenate((indzer, midz)))

    # ------------------------------------------------------------------
    # 2. Extrema
    # ------------------------------------------------------------------
    d = np.diff(x)
    d1, d2 = d[:-1], d[1:]
    indmin = np.where((d1 * d2 < 0) & (d1 < 0))[0] + 1
    indmax = np.where((d1 * d2 < 0) & (d1 > 0))[0] + 1

    # Handle constant (plateau) segments
    if np.any(d == 0):
        bad = d == 0
        dd = np.diff(np.r_[0, bad, 0])
        debs = np.where(dd == 1)[0]
        fins = np.where(dd == -1)[0] - 1

        # Discard segments at the very edges
        if debs.size and debs[0] == 0:
            debs, fins = debs[1:], fins[1:]
        if fins.size and fins[-1] == n - 1:
            debs, fins = debs[:-1], fins[:-1]

        imin, imax = [], []
        for k in range(debs.size):
            if d[debs[k] - 1] > 0 > d[fins[k]]:
                imax.append((debs[k] + fins[k]) // 2)
            elif d[debs[k] - 1] < 0 < d[fins[k]]:
                imin.append((debs[k] + fins[k]) // 2)

        indmin = np.unique(np.concatenate((indmin, imin)))
        indmax = np.unique(np.concatenate((indmax, imax)))

    return indmin.astype(int), indmax.astype(int), indzer.astype(int)


def extend(
    x: np.ndarray, indmin: np.ndarray, indmax: np.ndarray, extd_r: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Python implementation of the MATLAB `extend` routine from the EMD toolbox.

    :param x: 1-D ndarray, Original signal.
    :param indmin: 1-D array_like, Indices of local minima (0-based).
    :param indmax: 1-D array_lik, Extension ratio.  0 → no extension.
    :param extd_r: float, Extension ratio.  0 → no extension.
    :return: - ext_indmin : ndarray, Indices of minima in the extended signal.
             - ext_indmax : ndarray, Indices of maxima in the extended signal.
             - ext_x : ndarray, Extended signal.
             - cut_index : ndarray, Two-element array [start, end] giving indices to cut back to the original range in the extended signal.
    """
    x = np.asarray(x, dtype=float)
    indmin = np.asarray(indmin, dtype=int)
    indmax = np.asarray(indmax, dtype=int)

    xlen = x.size
    if extd_r == 0:
        return indmin, indmax, x, np.array([0, xlen - 1])

    # do not extend x
    nbsym = int(np.ceil(extd_r * indmax.size))  # number of extrema to use
    t = np.arange(xlen)

    # boundary conditions for interpolations :

    # ------------------------------------------------------------
    # 1. Left boundary
    # ------------------------------------------------------------
    if indmax[0] < indmin[0]:
        # first extremum is a maximum
        if x[0] > x[indmin[0]]:
            lmax = indmax[2 : min(indmax.size, nbsym + 1)][::-1]
            lmin = indmin[1 : min(indmin.size, nbsym)][::-1]
            lsym = indmax[0]
        else:
            lmax = indmax[1 : min(indmax.size, nbsym + 1)][::-1]
            lmin = np.r_[indmin[1 : min(indmin.size, nbsym)][::-1], 0]
            lsym = 0
    else:  # first extremum is a minimum
        if x[0] < x[indmax[0]]:
            lmax = indmax[1 : min(indmax.size, nbsym)][::-1]
            lmin = indmin[2 : min(indmin.size, nbsym + 1)][::-1]
            lsym = indmin[0]
        else:
            lmax = np.r_[indmax[1 : min(indmax.size, nbsym)][::-1], 0]
            lmin = indmin[1 : min(indmin.size, nbsym + 1)][::-1]
            lsym = 0

    # ------------------------------------------------------------
    # 2. Right boundary
    # ------------------------------------------------------------
    if indmax[-1] < indmin[-1]:  # last extremum is a minimum
        if x[-1] < x[indmax[-1]]:
            rmax = indmax[max(indmax.size - nbsym, 0) :][::-1]
            rmin = indmin[max(indmin.size - nbsym, 0) : -1][::-1]
            rsym = indmin[-1]
        else:
            rmax = np.r_[xlen - 1, indmax[max(indmax.size - nbsym + 1, 0) :][::-1]]
            rmin = indmin[max(indmin.size - nbsym, 0) :][::-1]
            rsym = xlen - 1
    else:  # last extremum is a maximum
        if x[-1] > x[indmin[-1]]:
            rmax = indmax[max(indmax.size - nbsym, 0) : -1][::-1]
            rmin = indmin[max(indmin.size - nbsym + 1, 0) :][::-1]
            rsym = indmax[-1]
        else:
            rmax = indmax[max(indmax.size - nbsym, 0) :][::-1]
            rmin = np.r_[xlen - 1, indmin[max(indmin.size - nbsym + 1, 0) :][::-1]]
            rsym = xlen - 1

    # Mirror locations
    tlmin = 2 * t[lsym] - t[lmin]
    tlmax = 2 * t[lsym] - t[lmax]
    trmin = 2 * t[rsym] - t[rmin]
    trmax = 2 * t[rsym] - t[rmax]

    # Ensure the extension reaches far enough
    if tlmin.size and tlmin[0] > t[0] or tlmax.size and tlmax[0] > t[0]:
        if lsym == indmax[0]:
            lmax = indmax[1 : min(indmax.size, nbsym + 1)][::-1]
        else:
            lmin = indmin[1 : min(indmin.size, nbsym + 1)][::-1]
        if lsym == 0:
            raise RuntimeError("Left extension bug")
        lsym = 0

    if trmin.size and trmin[-1] < t[-1] or trmax.size and trmax[-1] < t[-1]:
        if rsym == indmax[-1]:
            rmax = indmax[max(indmax.size - nbsym, 0) :][::-1]
        else:
            rmin = indmin[max(indmin.size - nbsym, 0) :][::-1]
        if rsym == xlen - 1:
            raise RuntimeError("Right extension bug")
        rsym = xlen - 1

    # ------------------------------------------------------------
    # 3. Build extended signal
    # ------------------------------------------------------------
    l_end = max(np.max(lmax) if lmax.size else 0, np.max(lmin) if lmin.size else 0)
    r_end = min(
        np.min(rmax) if rmax.size else (xlen - 1),
        np.min(rmin) if rmin.size else (xlen - 1),
    )

    new_lmax = l_end + 1 - lmax
    new_lmin = l_end + 1 - lmin
    new_rmax = rsym - rmax
    new_rmin = rsym - rmin

    lx_len = l_end - lsym
    lx = x[lsym + 1 : l_end + 1][::-1]
    rx = x[r_end:rsym][::-1]

    ext_x = np.r_[lx, x[lsym : rsym + 1], rx]

    lx_shift = lx_len - lsym
    ext_indmin = np.r_[
        new_lmin, indmin + lx_shift + 1, new_rmin + lx_shift + 1 + rsym
    ].astype(int)
    ext_indmax = np.r_[
        new_lmax, indmax + lx_shift + 1, new_rmax + lx_shift + 1 + rsym
    ].astype(int)

    cut_index = np.array([lx_shift + 1, xlen + lx_shift], dtype=int)

    return ext_indmin, ext_indmax, ext_x, cut_index


def smooth(x: np.ndarray, span: int) -> np.ndarray:
    """
    Python 版 MATLAB smooth(x, span) —— centered moving average.

    :param x: 1-D array_like
    :param span: int, 窗宽（点数）。必须是奇数；若为偶数自动加 1
    :return: y: np.ndarray, 平滑后的输入序列
    """
    x = np.asarray(x, dtype=float).ravel()
    N = x.size

    # 保证 span 为奇数，且不超过数据长度
    span = int(span)
    if span <= 0:
        raise ValueError("span must be a positive integer")
    if span % 2 == 0:
        span += 1
    if span > N:
        span = N if N % 2 else N - 1  # 取小于等于 N 的最大奇数

    k = (span - 1) // 2
    y = np.empty_like(x)

    for i in range(N):
        # 计算当前左右可取的点数
        left = min(i, k)
        right = min(N - 1 - i, k)
        win = x[i - left : i + right + 1]
        y[i] = win.mean()

    return y


def rms(
    x: np.ndarray,
    dim: Optional[int] = 0,
    *,
    keepdims: bool = False,
    omitnan: bool = False,
) -> np.ndarray | float:
    """
    等效 MATLAB 的 rms 函数
    Parameters
    ----------
    x : array_like
        输入数组
    dim : int, optional
        沿哪一条轴计算，默认 0（按列）
    keepdims : bool, optional
        是否保留被压缩的维度（False 与 MATLAB 默认一致）
    omitnan : bool, optional
        是否跳过 NaN（False 时与 MATLAB 默认一致）
    Returns
    -------
    y : ndarray
        RMS 结果
    """
    x = np.asarray(x, dtype=float)

    if omitnan:
        # 跳过 NaN：用 nanmean / nanstd 思路
        mean_sq = np.nanmean(x**2, axis=dim, keepdims=keepdims)
    else:
        mean_sq = np.mean(x**2, axis=dim, keepdims=keepdims)

    return np.sqrt(mean_sq)


def kurtosis(
    x: np.ndarray,
    axis: Optional[int] = 0,
    bias: bool = False,
    nan_policy: Optional[str] = "propagate",
) -> np.ndarray | float:
    """
    复现 MATLAB kurtosis(x) 行为（总体公式，flag=0）
    Parameters
    ----------
    x : array_like
    axis : int or tuple, optional
        同 MATLAB dim；默认 0（按列）
    bias : bool, optional
        False 时总体公式（相当于 MATLAB flag=0）
        True  时样本公式（相当于 MATLAB flag=1）
    nan_policy : {'propagate', 'omit', 'raise'}
        对 NaN 的处理策略
    Returns
    -------
    k : ndarray
        峰度值（已减 3）
    """
    x = np.asarray(x, dtype=float)

    # 处理 NaN
    if nan_policy == "omit":
        m4 = np.nanmean((x - np.nanmean(x, axis=axis, keepdims=True)) ** 4, axis=axis)
        var2 = np.nanvar(x, axis=axis, ddof=0) ** 2
    elif nan_policy == "propagate":
        m4 = np.mean((x - np.mean(x, axis=axis, keepdims=True)) ** 4, axis=axis)
        var2 = np.var(x, axis=axis, ddof=0) ** 2
    else:
        raise ValueError("nan_policy must be 'propagate', 'omit' or 'raise'")

    k = m4 / var2
    if not bias:
        # 总体公式无需修正；MATLAB 默认已减 3
        k = k - 3
    else:
        # 样本无偏修正（MATLAB flag=1）
        n = np.sum(~np.isnan(x), axis=axis) if nan_policy == "omit" else x.shape[axis]
        k = ((n + 1) * k - 3 * (n - 1)) * (n - 1) / ((n - 2) * (n - 3)) - 3

    return k


if __name__ == "__main__":
    from pysdkit.data import test_univariate_signal
    from pysdkit.plot import plot_IMFs
    from matplotlib import pyplot as plt

    time, signal = test_univariate_signal()

    rlmd = RLMD(max_imfs=8, max_iter=100, extd_r=0.1)
    imfs = rlmd.fit_transform(signal)

    plot_IMFs(signal, imfs)
    plt.show()
