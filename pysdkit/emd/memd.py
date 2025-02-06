# -*- coding: utf-8 -*-
"""
Created on 2025/02/04 13:10:33
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
We refactored from https://github.com/mariogrune/MEMD-Python-

待修改
"""
import numpy as np
import warnings

from sys import exit
from scipy.interpolate import interp1d, CubicSpline
from typing import Optional, Tuple


class MEMD(object):
    """
    Multivariate Empirical Mode Decomposition

    Rehman, Naveed, and Danilo P. Mandic. "Multivariate empirical mode decomposition."
    Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 466.2117 (2010): 1291-1302.

    G. Rilling, P. Flandrin and P. Goncalves, "On Empirical Mode Decomposition and its Algorithms", Proc of the IEEE-EURASIP
     Workshop on Nonlinear Signal and Image Processing, NSIP-03, Grado (I), June 2003

    N. E. Huang et al., "A confidence limit for the Empirical Mode Decomposition and Hilbert spectral analysis",
     Proceedings of the Royal Society A, Vol. 459, pp. 2317-2345, 2003

    Python code: https://github.com/mariogrune/MEMD-Python-
    MATLAB code: http://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm
    R code: https://rdrr.io/github/PierreMasselot/Library--emdr/man/memd.html
    """

    def __init__(
        self,
        stop_crit: Optional[str] = "stop",
        max_iter: Optional[int] = 128,
        n_dir: Optional[int] = 64,
        stop_vec: Optional[np.ndarray] = np.array([0.075, 0.75, 0.075]),
        stop_cnt: Optional[int] = 2,
    ) -> None:
        """
        初始化MEMD算法
        注意该算法要求输入
        :param stop_crit: 算法停止迭代的准则，可以选择['stop', 'fix_h']
        :param max_iter: 最大迭代次数
        :param n_dir: 方向向量的投影个数，一般来说该参数无需指定，
                      在输入信号后该参数将设定为输入信号通道数目的两倍
        :param stop_vec: 当`stop_crit`为'stop'时使用的停止参数设置
        :param stop_cnt: 当`stop_crit`为'fix_h'时使用的停驶参数设置
        """

        # 设置算法的停止迭代准则
        if not isinstance(stop_crit, str) or (
            stop_crit != "stop" and stop_crit != "fix_h"
        ):
            exit("invalid stop_criteria. stop_criteria should be either fix_h or stop")
        self.stop_crit = stop_crit
        self.max_iter = max_iter

        # 设置方向投影向量
        if not isinstance(n_dir, int) or n_dir < 6:
            exit(
                "invalid num_dir. num_dir should be an integer greater than or equal to 6."
            )
        self.n_dir = n_dir

        # 使用'stop'作为停止准则
        if not isinstance(stop_vec, (list, tuple, np.ndarray)) or any(
            x for x in stop_vec if not isinstance(x, (int, float, complex))
        ):
            exit(
                "invalid stop_vector. stop_vector should be a list with three elements, default is [0.75,0.75,0.75]"
            )
        self.stop_vec = stop_vec
        self.sd, self.sd2, self.tol = stop_vec[0], stop_vec[1], stop_vec[2]

        # 使用'fix_h'作为停止准则
        if not isinstance(stop_cnt, int) or stop_cnt < 0:
            exit("invalid stop_count. stop_count should be a non-negative integer.")
        self.stop_cnt = stop_cnt

    def __call__(self, *args, **kwargs):
        pass

    def init_hammersley(self, N_dim: int) -> np.ndarray:
        """Initializations for Hammersley function"""
        base = [-self.n_dir]

        # Find the pointset for the given input signal
        if N_dim == 3:
            base.append(2)
            seq = np.zeros((self.n_dir, N_dim - 1))
            for it in range(0, N_dim - 1):
                seq[:, it] = hamm(self.n_dir, base[it])
        else:
            # Prime numbers for Hammersley sequence
            prm = nth_prime(N_dim - 1)
            for itr in range(1, N_dim):
                base.append(prm[itr - 1])
            seq = np.zeros((self.n_dir, N_dim))
            for it in range(0, N_dim):
                seq[:, it] = hamm(self.n_dir, base[it])

        return seq

    def stop_emd(self, signal: np.ndarray, seq: np.ndarray, N_dim: int) -> bool:
        """控制是否停止EMD算法的迭代"""
        ner = np.zeros(shape=(self.n_dir, 1))
        dir_vec = np.zeros(shape=(N_dim, 1))

        for it in range(0, self.n_dir):

            if N_dim != 3:
                # Multivariate signal (for N_dim ~=3) with hammersley sequence
                # Linear normalisation of hammersley sequence in the range of -1.00 - 1.00
                b = 2 * seq[it, :] - 1

                # Find angles corresponding to the normalised sequence
                tht = np.arctan2(
                    np.sqrt(np.flipud(np.cumsum(b[:0:-1] ** 2))), b[: N_dim - 1]
                ).transpose()

                # Find coordinates of unit direction vectors on n-sphere
                dir_vec[:, 0] = np.cumprod(np.concatenate(([1], np.sin(tht))))
                dir_vec[: N_dim - 1, 0] = np.cos(tht) * dir_vec[: N_dim - 1, 0]

            else:
                # Trivariate signal with hammersley sequence
                # Linear normalisation of hammersley sequence in the range of -1.0 - 1.0
                tt = 2 * seq[it, 0] - 1
                if tt > 1:
                    tt = 1
                elif tt < -1:
                    tt = -1

                # Normalize angle from 0 - 2*pi
                phirad = seq[it, 1] * 2 * np.pi
                st = np.sqrt(1.0 - tt * tt)

                dir_vec[0] = st * np.cos(phirad)
                dir_vec[1] = st * np.sin(phirad)
                dir_vec[2] = tt
            # Projection of input signal on nth (out of total ndir) direction vectors
            y = np.dot(signal, dir_vec)

            # Calculates the extrema of the projected signal
            indmin, indmax = local_peaks(y)

            ner[it] = len(indmin) + len(indmax)

        # Stops if the all projected signals have less than 3 extrema
        stop_flag = all(ner < 3)

        return stop_flag

    def stop(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        seq: np.ndarray,
        seq_len: int,
        N_dim: int,
    ) -> Tuple[bool, np.ndarray]:
        """"""
        try:
            env_mean, nem, nzm, amp = envelope_mean(
                signal, time, seq, self.n_dir, seq_len, N_dim
            )
            sx = np.sqrt(np.sum(np.power(env_mean, 2), axis=1))

            if all(amp):
                # something is wrong here
                sx = sx / amp

            if not (
                (np.mean(sx > self.sd) > self.tol or any(sx > self.sd2)) or any(nem > 2)
            ):
                stop_flag = True
            else:
                stop_flag = False
        except Exception as e:
            env_mean = np.zeros(shape=(seq_len, N_dim))
            stop_flag = True

        return stop_flag, env_mean

    def fix(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        seq: np.ndarray,
        seq_len: int,
        N_dim: int,
        counter: int,
    ) -> Tuple[bool, np.ndarray, int]:
        """"""
        try:
            env_mean, nem, nzm, amp = envelope_mean(
                signal, time, seq, self.n_dir, seq_len, N_dim
            )

            if all(np.abs(nzm - nem) > 1):
                stop_flag = False
                counter = 0
            else:
                counter += 1
                stop_flag = counter >= self.stop_cnt

        except Exception as e:
            env_mean = np.zeros(shape=(seq_len, N_dim))
            stop_flag = True

        return stop_flag, env_mean, counter

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Preform the Multivariate Empirical Mode Decomposition method.
        请注意该方法仅适用于输入维数大于等于3的信号（遵循最原始的MEMD的MATLAB代码）
        :param signal:
        :return:
        """
        # 获取输入信号的维度和长度
        N_dim, seq_len = signal.shape

        if N_dim < 3:
            raise ValueError(
                "MEMD仅能处理输入三元及以上的信号，如果待处理的信号不满足要求可以尝试使用MVMD算法。"
            )

        # Initializations for Hammersley function
        seq = self.init_hammersley(N_dim)

        # 通过输入信号的维数初始化投影维数
        self.n_dir = N_dim * 2

        # 生成时间戳序列数组
        time = np.arange(1, seq_len + 1)

        # 对输入信号进行通道变换以进行分解
        signal = signal.transpose((1, 0))

        print(signal.shape)

        # 用于存放分解结果的列表
        imfs = []

        # 记录已分解IMFs的数目
        n_imfs = 1

        # Counter
        nbit = 0

        # 开始进行迭代分解
        while not self.stop_emd(signal=signal, seq=seq, N_dim=N_dim):
            # Get current mode
            m = signal

            # computation of mean and stopping criterion
            if self.stop_crit == "stop":
                stop_flag, env_mean = self.stop(
                    signal=m, time=time, seq=seq, N_dim=N_dim, seq_len=seq_len
                )
            elif self.stop_crit == "fix_h":
                counter = 0
                stop_flag, env_mean, counter = self.fix(
                    signal=m,
                    time=time,
                    seq=seq,
                    seq_len=seq_len,
                    N_dim=N_dim,
                    counter=counter,
                )
            else:
                raise ValueError(
                    "参数`stop_crit`设置错误!请从'stop'或'fix_h'中进行选择。"
                )

            print(stop_flag)

            # In case the current mode is so small that machine precision can cause
            # spurious extrema to appear
            if np.max(np.abs(m)) < 1e-10 * (np.max(np.abs(signal))):
                if not stop_flag:
                    warnings.warn(
                        "emd:warning, forced stop of EMD : too small amplitude"
                    )
                else:
                    print("forced stop of EMD : too small amplitude")
                break

            # sifting loop
            while stop_flag is False and nbit < self.max_iter:
                # sifting
                m = m - env_mean

                # computation of mean and stopping criterion
                if self.stop_crit == "stop":
                    stop_flag, env_mean = self.stop(
                        signal=m, time=time, seq=seq, N_dim=N_dim, seq_len=seq_len
                    )
                else:
                    stop_flag, env_mean, counter = self.fix(
                        signal=m,
                        time=time,
                        seq=seq,
                        seq_len=seq_len,
                        N_dim=N_dim,
                        counter=counter,
                    )

                nbit = nbit + 1

                if nbit == (self.max_iter - 1) and nbit > 100:
                    warnings.warn(
                        "emd:warning, forced stop of sifting : too many iterations"
                    )

            # 记录本次分解的结果
            imfs.append(m)

            n_imfs += 1
            signal = signal - m
            nbit = 0

        # Stores the residue
        imfs.append(signal)
        imfs = np.asarray(imfs)

        return imfs


def local_peaks(x):
    if all(x < 1e-5):
        x = np.zeros((1, len(x)))

    m = len(x) - 1

    # Calculates the extrema of the projected signal
    # Difference between subsequent elements:
    dy = np.diff(x.transpose()).transpose()
    a = np.where(dy != 0)[0]
    lm = np.where(np.diff(a) != 1)[0] + 1
    d = a[lm] - a[lm - 1]
    a[lm] = a[lm] - np.floor(d / 2)
    a = np.insert(a, len(a), m)
    ya = x[a]

    if len(ya) > 1:
        # Maxima
        pks_max, loc_max = peaks(ya)
        # Minima
        pks_min, loc_min = peaks(-ya)

        if len(pks_min) > 0:
            indmin = a[loc_min]
        else:
            indmin = np.asarray([])

        if len(pks_max) > 0:
            indmax = a[loc_max]
        else:
            indmax = np.asarray([])
    else:
        indmin = np.array([])
        indmax = np.array([])

    return indmin, indmax


def peaks(X):
    dX = np.sign(np.diff(X.transpose())).transpose()
    locs_max = np.where(np.logical_and(dX[:-1] > 0, dX[1:] < 0))[0] + 1
    pks_max = X[locs_max]

    return pks_max, locs_max


def hamm(n, base):
    seq = np.zeros((1, n))

    if 1 < base:
        seed = np.arange(1, n + 1)
        base_inv = 1 / base
        while any(x != 0 for x in seed):
            digit = np.remainder(seed[0:n], base)
            seq = seq + digit * base_inv
            base_inv = base_inv / base
            seed = np.floor(seed / base)
    else:
        temp = np.arange(1, n + 1)
        seq = (np.remainder(temp, (-base + 1)) + 0.5) / (-base)

    return seq


def nth_prime(n):
    lst = [2]
    for i in range(3, 104745):
        if is_prime(i):
            lst.append(i)
            if len(lst) == n:
                return lst


def is_prime(x):
    if x == 2:
        return True
    else:
        for number in range(3, x):
            if x % number == 0 or x % 2 == 0:
                return False

        return True


def envelope_mean(m, t, seq, ndir, N, N_dim):  # new

    NBSYM = 2
    count = 0

    env_mean = np.zeros((len(t), N_dim))
    amp = np.zeros((len(t)))
    nem = np.zeros((ndir))
    nzm = np.zeros((ndir))

    dir_vec = np.zeros((N_dim, 1))
    for it in range(0, ndir):
        if N_dim != 3:  # Multivariate signal (for N_dim ~=3) with hammersley sequence
            # Linear normalisation of hammersley sequence in the range of -1.00 - 1.00
            b = 2 * seq[it, :] - 1

            # Find angles corresponding to the normalised sequence
            tht = np.arctan2(
                np.sqrt(np.flipud(np.cumsum(b[:0:-1] ** 2))), b[: N_dim - 1]
            ).transpose()

            # Find coordinates of unit direction vectors on n-sphere
            dir_vec[:, 0] = np.cumprod(np.concatenate(([1], np.sin(tht))))
            dir_vec[: N_dim - 1, 0] = np.cos(tht) * dir_vec[: N_dim - 1, 0]

        else:  # Trivariate signal with hammersley sequence
            # Linear normalisation of hammersley sequence in the range of -1.0 - 1.0
            tt = 2 * seq[it, 0] - 1
            if tt > 1:
                tt = 1
            elif tt < -1:
                tt = -1

                # Normalize angle from 0 - 2*pi
            phirad = seq[it, 1] * 2 * np.pi
            st = np.sqrt(1.0 - tt * tt)

            dir_vec[0] = st * np.cos(phirad)
            dir_vec[1] = st * np.sin(phirad)
            dir_vec[2] = tt

        # Projection of input signal on nth (out of total ndir) direction vectors
        y = np.dot(m, dir_vec)

        # Calculates the extrema of the projected signal
        indmin, indmax = local_peaks(y)

        nem[it] = len(indmin) + len(indmax)
        indzer = zero_crossings(y)
        nzm[it] = len(indzer)

        tmin, tmax, zmin, zmax, mode = boundary_conditions(
            indmin, indmax, t, y, m, NBSYM
        )

        # Calculate multidimensional envelopes using spline interpolation
        # Only done if number of extrema of the projected signal exceed 3
        if mode:
            fmin = CubicSpline(tmin, zmin, bc_type="not-a-knot")
            env_min = fmin(t)
            fmax = CubicSpline(tmax, zmax, bc_type="not-a-knot")
            env_max = fmax(t)
            amp = amp + np.sqrt(np.sum(np.power(env_max - env_min, 2), axis=1)) / 2
            env_mean = env_mean + (env_max + env_min) / 2
        else:  # if the projected signal has inadequate extrema
            count = count + 1

    if ndir > count:
        env_mean = env_mean / (ndir - count)
        amp = amp / (ndir - count)
    else:
        env_mean = np.zeros((N, N_dim))
        amp = np.zeros((N))
        nem = np.zeros((ndir))

    return (env_mean, nem, nzm, amp)


def zero_crossings(x):
    indzer = np.where(x[0:-1] * x[1:] < 0)[0]

    if any(x == 0):
        iz = np.where(x == 0)[0]
        if any(np.diff(iz) == 1):
            zer = x == 0
            dz = np.diff([0, zer, 0])
            debz = np.where(dz == 1)[0]
            finz = np.where(dz == -1)[0] - 1
            indz = np.round((debz + finz) / 2)
        else:
            indz = iz
        indzer = np.sort(np.concatenate((indzer, indz)))

    return indzer


def boundary_conditions(indmin, indmax, t, x, z, nbsym):
    lx = len(x) - 1
    end_max = len(indmax) - 1
    end_min = len(indmin) - 1
    indmin = indmin.astype(int)
    indmax = indmax.astype(int)

    if len(indmin) + len(indmax) < 3:
        mode = 0
        tmin = tmax = zmin = zmax = None
        return (tmin, tmax, zmin, zmax, mode)
    else:
        mode = 1  # the projected signal has inadequate extrema
    # boundary conditions for interpolations :
    if indmax[0] < indmin[0]:
        if x[0] > x[indmin[0]]:
            lmax = np.flipud(indmax[1 : min(end_max + 1, nbsym + 1)])
            lmin = np.flipud(indmin[: min(end_min + 1, nbsym)])
            lsym = indmax[0]

        else:
            lmax = np.flipud(indmax[: min(end_max + 1, nbsym)])
            lmin = np.concatenate(
                (np.flipud(indmin[: min(end_min + 1, nbsym - 1)]), ([0]))
            )
            lsym = 0

    else:
        if x[0] < x[indmax[0]]:
            lmax = np.flipud(indmax[: min(end_max + 1, nbsym)])
            lmin = np.flipud(indmin[1 : min(end_min + 1, nbsym + 1)])
            lsym = indmin[0]

        else:
            lmax = np.concatenate(
                (np.flipud(indmax[: min(end_max + 1, nbsym - 1)]), ([0]))
            )
            lmin = np.flipud(indmin[: min(end_min + 1, nbsym)])
            lsym = 0

    if indmax[-1] < indmin[-1]:
        if x[-1] < x[indmax[-1]]:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0) :])
            rmin = np.flipud(indmin[max(end_min - nbsym, 0) : -1])
            rsym = indmin[-1]

        else:
            rmax = np.concatenate(
                (np.array([lx]), np.flipud(indmax[max(end_max - nbsym + 2, 0) :]))
            )
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0) :])
            rsym = lx

    else:
        if x[-1] > x[indmin[-1]]:
            rmax = np.flipud(indmax[max(end_max - nbsym, 0) : -1])
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0) :])
            rsym = indmax[-1]

        else:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0) :])
            rmin = np.concatenate(
                (np.array([lx]), np.flipud(indmin[max(end_min - nbsym + 2, 0) :]))
            )
            rsym = lx

    tlmin = 2 * t[lsym] - t[lmin]
    tlmax = 2 * t[lsym] - t[lmax]
    trmin = 2 * t[rsym] - t[rmin]
    trmax = 2 * t[rsym] - t[rmax]

    # in case symmetrized parts do not extend enough
    if tlmin[0] > t[0] or tlmax[0] > t[0]:
        if lsym == indmax[0]:
            lmax = np.flipud(indmax[: min(end_max + 1, nbsym)])
        else:
            lmin = np.flipud(indmin[: min(end_min + 1, nbsym)])
        if lsym == 1:
            exit("bug")
        lsym = 0
        tlmin = 2 * t[lsym] - t[lmin]
        tlmax = 2 * t[lsym] - t[lmax]

    if trmin[-1] < t[lx] or trmax[-1] < t[lx]:
        if rsym == indmax[-1]:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0) :])
        else:
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0) :])
        if rsym == lx:
            exit("bug")
        rsym = lx
        trmin = 2 * t[rsym] - t[rmin]
        trmax = 2 * t[rsym] - t[rmax]

    zlmax = z[lmax, :]
    zlmin = z[lmin, :]
    zrmax = z[rmax, :]
    zrmin = z[rmin, :]

    tmin = np.hstack((tlmin, t[indmin], trmin))
    tmax = np.hstack((tlmax, t[indmax], trmax))
    zmin = np.vstack((zlmin, z[indmin, :], zrmin))
    zmax = np.vstack((zlmax, z[indmax, :], zrmax))

    return (tmin, tmax, zmin, zmax, mode)


if __name__ == "__main__":
    memd = MEMD()

    inp = np.random.rand(5, 100)

    imf = memd.fit_transform(inp)
    print(imf.shape)

    imf_x = imf[:, 0, :]
    imf_y = imf[:, 1, :]
    imf_z = imf[:, 2, :]
