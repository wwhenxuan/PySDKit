# -*- coding: utf-8 -*-
"""
Created on 2025/02/02 00:13:28
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
这部分代码等待进行优化
"""
import numpy as np
from scipy.interpolate import pchip_interpolate
from scipy.signal import hilbert

from typing import Tuple


def find_extrema(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """获取输入信号的极值点"""
    length = len(signal)  # 获取信号的长度
    diff = np.diff(signal)  # 获取信号的一阶差分

    # 不考虑两个端点
    d1 = diff[:-1]
    d2 = diff[1:]

    # 通过差分序列进行极值点的初步筛选
    # 在信号变化率从负变正时发生
    indmin = np.where((d1 * d2 < 0.0) & (d1 < 0.0))[0] + 1
    # 在信号变化率从正变负时发生
    indmax = np.where((d1 * d2 < 0.0) & (d1 > 0.0))[0] + 1

    # 进一步处理差分为0的平稳部分
    if len(np.where(diff == 0.0)[0]) > 0:
        # 存放结果的数组
        imax, imin = np.array([], dtype=int), np.array([], dtype=int)

        # 标记diff中差分为0的位置
        bad = diff == 0.0

        # 对数组进行两端拓展使其能够处理开始和结束
        c_bad = np.concatenate([np.array([0]), bad, np.array([0])])

        # 用于找到平稳区间的起始点和结束点
        dd = np.diff(c_bad)
        debs = np.where(dd == 1)[0]
        fins = np.where(dd == -1)[0]

        # 进一步判断
        if len(debs) > 0 and debs[0] == 0:
            if len(debs) > 1:
                debs = debs[1:]
                fins = fins[1:]
            else:
                debs = np.array([], dtype=int)
                fins = np.array([], dtype=int)

        if len(debs) > 0:
            if fins[-1] == length - 1:
                if len(debs) > 1:
                    debs = debs[:-1]
                    fins = fins[:-1]
                else:
                    debs = np.array([], dtype=int)
                    fins = np.array([], dtype=int)

        if len(debs) > 0:
            for k in range(len(debs)):
                if diff[debs[k] - 1] > 0:
                    if diff[fins[k]] < 0:
                        imax = np.concatenate(
                            [imax, [np.round((fins[k] + debs[k]) / 2)]]
                        )
                else:
                    if diff[fins[k]] > 0:
                        imin = np.concatenate(
                            [imin, [np.round((fins[k] + debs[k]) / 2)]]
                        )

        # 对最终的索引进行合并
        if len(imax) > 0:
            indmax = np.sort(np.concatenate([indmax, imax]))
        if len(imin) > 0:
            indmin = np.sort(np.concatenate([indmin, imin]))

    return indmin.astype(int), indmax.astype(int)


def inst_freq_local(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """进行希尔伯特黄变换并获得谱分布"""
    # 输入数据的维度
    dimension = data.shape
    if len(dimension) == 1:
        # 对输入进行维数拓展
        dimension = (1, dimension[0])
        data = np.expand_dims(data, axis=0)
    # 记录变换后的瞬时振幅和频率
    inst_amp = np.zeros(dimension, dtype=float)
    inst_freq = np.zeros(dimension, dtype=float)

    for k in range(dimension[0]):
        # 进行希尔伯特变换
        h = hilbert(data[k, :])

        # 处理功率谱
        inst_amp_temp = np.abs(h)
        inst_amp[k, :] = inst_amp_temp.flatten()

        phi = np.unwrap(np.angle(h))
        inst_freq_temp = (phi[2:] - phi[:-2]) / 2
        inst_freq_temp = np.concatenate(
            [
                np.array([inst_freq_temp[0]]),
                inst_freq_temp,
                np.array([inst_freq_temp[-1]]),
            ]
        )
        inst_freq[k, :] = inst_freq_temp.flatten() / (2 * np.pi)

    # 进一步处理两个端点
    inst_amp[0, 0] = inst_amp[0, 1]
    inst_amp[0, -1] = inst_amp[0, -2]

    inst_freq[np.where(inst_freq <= 0.0)] = 0.0

    return inst_amp[0], inst_freq[0]


def divide2exp(y, inst_amp_0, inst_freq_0):
    # divide y(t) into two sub-signals a1(t)exp(2*pi*f1(t)) and a2(t)exp(2*pi*f2(t)).
    # Parameters
    # ----------
    # y : numpy ndarray
    #     Input signal.
    # inst_amp_0 : numpy ndarray
    #     Instantaneous amplitude of `y`.
    # inst_freq_0 : numpy ndarray
    #     Instantaneous frequency of `y`.
    # Returns
    # -------
    # a1 : numpy ndarray
    #     Instantaneous amplitude of the first sub-signal.
    # f1 : numpy ndarray
    #     Instantaneous frequency of the first sub-signal.
    # a2 : numpy ndarray
    #     Instantaneous amplitude of the first sub-signal.
    # f2 : numpy ndarray
    #     Instantaneous frequency of the second sub-signal.
    # bis_freq : numpy ndarray
    #     Bisecting frequency, (`f1` + `f2`) / 2.
    # ratio_bw : numpy ndarray
    #     Instantaneous bandwidth ratio (stopping criterion).
    # avg_freq : numpy ndarray
    #     Average frequency.
    l_inst = len(inst_amp_0) if len(inst_amp_0.shape) == 1 else len(inst_amp_0[0])
    tt = np.arange(0, l_inst, dtype=int)
    squar_inst_amp_0 = np.power(inst_amp_0, 2.0)

    indmin_y, indmax_y = find_extrema(y)
    indmin_amp_0, indmax_amp_0 = find_extrema(squar_inst_amp_0)

    if len(indmin_amp_0) < 2 or len(indmax_amp_0) < 2:
        a1 = np.zeros(inst_amp_0.shape, dtype=float)
        a2 = np.zeros(inst_amp_0.shape, dtype=float)
        f1 = inst_freq_0.copy()
        f2 = inst_freq_0.copy()
        ratio_bw = a1.copy()
        bis_freq = np.zeros(inst_amp_0.shape, dtype=float)
        avg_freq = np.zeros(inst_amp_0.shape, dtype=float)

        return a1, f1, a2, f2, bis_freq, ratio_bw, avg_freq

    envpmax_inst_amp = pchip_interpolate(indmax_amp_0, inst_amp_0[indmax_amp_0], tt)
    envpmin_inst_amp = pchip_interpolate(indmin_amp_0, inst_amp_0[indmin_amp_0], tt)

    a1 = (envpmax_inst_amp + envpmin_inst_amp) / 2.0
    a2 = (envpmax_inst_amp - envpmin_inst_amp) / 2.0
    indmin_a2, indmax_a2 = find_extrema(a2)

    inst_amp_inst_amp_2 = inst_freq_0 * np.power(inst_amp_0, 2.0)
    inst_amp_tmax = pchip_interpolate(
        indmax_amp_0,
        inst_amp_inst_amp_2[indmax_amp_0],
        np.arange(0, len(inst_amp_inst_amp_2), dtype=int),
    )
    inst_amp_tmin = pchip_interpolate(
        indmin_amp_0,
        inst_amp_inst_amp_2[indmin_amp_0],
        np.arange(0, len(inst_amp_inst_amp_2), dtype=int),
    )
    f1 = np.zeros((len(inst_freq_0),), dtype=float)
    f2 = np.zeros((len(inst_freq_0),), dtype=float)
    for i in range(len(inst_freq_0)):
        a_mtx = np.empty((2, 2), dtype=float)
        a_mtx[0, :] = np.array(
            [np.power(a1[i], 2.0) + a1[i] * a2[i], np.power(a2[i], 2.0) + a1[i] * a2[i]]
        )
        a_mtx[1, :] = np.array(
            [np.power(a1[i], 2.0) - a1[i] * a2[i], np.power(a2[i], 2.0) - a1[i] * a2[i]]
        )
        b_mtx = np.array([inst_amp_tmax[i], inst_amp_tmin[i]])
        c_mtx = np.linalg.solve(a_mtx, b_mtx)
        f1[i] = c_mtx[0]
        f2[i] = c_mtx[1]

    bis_freq = (inst_amp_tmax - inst_amp_tmin) / (4 * a1 * a2)
    if len(indmax_a2) > 3:
        bis_freq = pchip_interpolate(indmax_a2, bis_freq[indmax_a2], tt)

    avg_freq = (inst_amp_tmax + inst_amp_tmin) / (
        2 * (np.power(a1, 2.0) + np.power(a2, 2.0))
    )
    cos_diffphi = (
        np.power(inst_amp_0, 2.0) - np.power(a1, 2.0) - np.power(a2, 2.0)
    ) / (2 * a1 * a2)
    cos_diffphi[cos_diffphi > 1.2] = 1
    cos_diffphi[cos_diffphi < -1.2] = -1
    inst_amp1, inst_freq_diff_phi = inst_freq_local(cos_diffphi)

    diff_a1 = (a1[2:] - a1[:-2]) / 2.0
    diff_a1 = np.concatenate([[diff_a1[0]], diff_a1, [diff_a1[-1]]])
    diff_a2 = (a2[2:] - a2[:-2]) / 2.0
    diff_a2 = np.concatenate([[diff_a2[0]], diff_a2, [diff_a2[-1]]])

    inst_bw = np.power(
        (np.power(diff_a1, 2.0) + np.power(diff_a2, 2.0))
        / (np.power(a1, 2.0) + np.power(a2, 2.0))
        + np.power(a1, 2.0)
        * np.power(a2, 2.0)
        * np.power(inst_freq_diff_phi, 2.0)
        / np.power(np.power(a1, 2.0) + np.power(a2, 2.0), 2.0),
        0.5,
    )
    ratio_bw = np.abs(inst_bw / avg_freq)
    ratio_bw[(a2 / a1) < 5e-3] = 0
    ratio_bw[avg_freq < 1e-7] = 0
    ratio_bw[ratio_bw > 1] = 1

    ff1 = (inst_freq_diff_phi + 2.0 * bis_freq) / 2.0
    ff2 = (2.0 * bis_freq - inst_freq_diff_phi) / 2.0
    f1[np.abs((a1 - a2) / a1) < 0.05] = ff1[np.abs((a1 - a2) / a1) < 0.05]
    f2[np.abs((a1 - a2) / a1) < 0.05] = ff2[np.abs((a1 - a2) / a1) < 0.05]

    temp_inst_amp_0 = inst_amp_0.copy()
    for j in range(len(indmax_y) - 1):
        ind = np.arange(indmax_y[j], indmax_y[j + 1] + 1, dtype=int)
        temp_inst_amp_0[ind] = np.mean(inst_amp_0[ind])

    ratio_bw[np.abs(temp_inst_amp_0) / np.max(np.abs(y)) < 5e-2] = 0
    f1[np.abs(temp_inst_amp_0) / np.max(np.abs(y)) < 4e-2] = 1 / len(y) / 1000
    f2[np.abs(temp_inst_amp_0) / np.max(np.abs(y)) < 4e-2] = 1 / len(y) / 1000
    bis_freq[bis_freq > 0.5] = 0.5
    bis_freq[bis_freq < 0] = 0

    return a1, f1, a2, f2, bis_freq, ratio_bw, avg_freq


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 0, 1, -1, -2, 1, 3])
    from matplotlib import pyplot as plt

    plt.plot(x)
    plt.show()

    print(find_extrema(x))
