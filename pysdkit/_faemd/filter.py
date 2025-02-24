# -*- coding: utf-8 -*-
"""
Created on 2025/02/24 19:07:29
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from numpy import max, min


def filter_size1D(imax: np.ndarray, imin: np.ndarray) -> int:
    """
    此代码用于计算经验模态分解（EMD）中信号滤波所需的窗口大小。
    其核心逻辑是通过分析信号极值点（极大值、极小值）的间隔分布，
    动态确定合适的滤波窗口尺寸，以确保信号平滑处理的有效性。
    窗口大小的选择基于极值点间隔的统计特征，并强制为奇数以保证对称性。

    Origin Written by Mruthun Thirumalaisamy March 30 2018
    :param imax: 局部极大值点的位置
    :param imin: 局部极小值点的位置
    :return: Calculated value of the window size
    """
    # 通过差分计算极值点之间的间隔长度
    edge_len_max = np.diff(imax)
    edge_len_min = np.diff(imin)

    ##### Window size calculations #####
    # 两类极值间隔的最小值中的较小者
    d1 = min(min(edge_len_max), min(edge_len_min))
    # 两类极值间隔的最小值中的较大者
    d2 = max(min(edge_len_max), min(edge_len_min))
    # 两类极值间隔的最大值中的较小者
    d3 = min(max(edge_len_max), max(edge_len_min))
    # 两类极值间隔的最大值中的较大者
    d4 = max(max(edge_len_max), max(edge_len_min))

    # making sure w_size is an odd integer
    window_size = 2 * (np.floor(d4 / 2) - 1)

    if window_size < 3:
        # WARNING: Calculated Window size less than 3
        # Overriding calculated value and setting window size = 3
        window_size = 3

    return window_size
