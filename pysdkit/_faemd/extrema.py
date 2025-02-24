# -*- coding: utf-8 -*-
"""
Created on 2025/02/24 13:20:36
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from pysdkit.utils import differ


def extrema(x: np.ndarray):
    """"""

    # 初始化输出变量 将要输出结果的空数组
    xmax, imax, xmin, imin = [], [], [], []

    # 检查输入判断x是否为一位向量
    print(np.size(x), x.shape[0])
    if np.size(x) != x.shape[0]:
        raise ValueError("输入信号x必须是一维数组，不能是多元向量")

    # 检验输入信号中的缺失值
    nan = np.any(np.isnan(x))
    print(nan)
    if nan is True:
        # 输入信号存在缺失值
        raise ValueError("`extrema`函数无法处理带有缺失值的信号")

    # Difference between subsequent elements
    dx = differ(x, delta=1)
    print("差分序列", dx, np.any(dx))

    # Is a horizontal line
    if np.any(dx) is not True:
        return None, None, None, None

    # Flat peaks? Put the middle element:
    # a = find()


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    extrema(x)
