# -*- coding: utf-8 -*-
"""
Created on 2025/02/24 13:20:36
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
要不要考虑将这个函数移动到utils中
"""
import numpy as np

from typing import Tuple, Union


def extrema(
    x: np.ndarray,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[None, None, None, None]
]:
    """
    Gets the global extrema points from a time series

    DEFINITION (from https://en.wikipedia.org/wiki/Maxima_and_minima):
    In mathematics, maxima and minima, also known as extrema, are points in the domain of a function at which the
    function takes a largest value (maximum) or smallest value (minimum), either within a given neighbourhood
    (local extrema) or on the function domain in its entirety (global extrema).

    :param x: The input 1D time series or signal.
    :return: xmax - maxima points in descending order.
             imax - indexes of the XMAX.
             xmin - minima points in descending order.
             imin - indexes of the XMIN.
    """

    # 初始化输出变量 将要输出结果的空数组
    xmax, imax, xmin, imin = [], [], [], []

    # 检查输入判断x是否为一位向量
    if np.size(x) != x.shape[0]:
        raise ValueError("输入信号x必须是一维数组，不能是多元向量")

    # 记录信号的长度
    seq_len = x.shape[0]

    # 生成索引向量
    index = np.arange(0, seq_len)

    # 检验输入信号中的缺失值
    nan = np.any(np.isnan(x))
    if nan is True:
        # 输入信号存在缺失值
        raise ValueError("`extrema`函数无法处理带有缺失值的信号")

    # Difference between subsequent elements
    dx = np.diff(x)

    # Is a horizontal line
    if ~np.any(dx) is True:
        return None, None, None, None

    # Flat peaks? Put the middle element:
    a = np.where(dx != 0)[0]  # Indexes where x changes
    lm = np.where(np.diff(a) != 1)[0] + 1  # Indexes where a do not change
    d = a[lm] - a[lm - 1]
    a[lm] = a[lm] - np.floor(d / 2)
    a = np.append(a, seq_len - 1)

    # Peaks ?
    # Series without flat peaks
    xa = x[a]
    # 1  =>  positive slopes (minima begin)
    # 0  =>  negative slopes (maxima begin)
    b = (np.diff(xa) > 0).astype(int)

    # -1 =>  maxima indexes (but one)
    # +1 =>  minima indexes (but one)
    xb = np.diff(b)

    # maxima indexes
    imax = np.where(xb == -1)[0] + 1

    # minima indexes
    imin = np.where(xb == +1)[0] + 1

    imax = a[imax]
    imin = a[imin]

    nmaxi = len(imax)
    nmini = len(imin)

    # Maximum or minumim on a flat peak at the ends?
    if nmaxi == 0 and nmini == 0:
        if x[0] > x[-1]:
            xmax = x[0]
            imax = index[0]
            xmin = x[-1]
            imin = index[-1]
        elif x[1] < x[-1]:
            xmax = x[-1]
            imax = index[-1]
            xmin = x[0]
            imin = index[0]
        return xmax, imax, xmin, imin

    # Maximum or minumim at the ends?
    if nmaxi == 0:
        imax = np.array([0, seq_len - 1])
    elif nmini == 0:
        imin = np.array([0, seq_len - 1])
    else:
        if imax[0] < imin[0]:
            imin = np.append(0, imin)
        else:
            imax = np.append(0, imax)
        if imax[-1] > imin[-1]:
            imin = np.append(seq_len - 1, imin)
        else:
            imax = np.append(seq_len - 1, imax)

    # 通过索引获取具体的数值
    xmax = x[imax]
    xmin = x[imin]

    return xmax, imax, xmin, imin


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    x = 2 * np.pi * np.linspace(-1, 1, 100)
    y = np.cos(x) - 0.5 + 0.5 * np.random.rand(100)
    y[39:45] = 1.85

    xmax, imax, xmin, imin = extrema(y)
    plt.figure(dpi=700)
    plt.plot(y)
    plt.scatter(imax, xmax)
    plt.scatter(imin, xmin)
    plt.show()
