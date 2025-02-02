# -*- coding: utf-8 -*-
"""
Created on 2025/02/02 13:01:46
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
生成用于测试二维图像数据的样例
"""
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt

from typing import Optional, Tuple


def test_grayscale() -> np.ndarray:
    """
    加载用于测试的二维灰度图像样例
    This data comes from https://www.mathworks.com/matlabcentral/fileexchange/45918-two-dimensional-variational-mode-decomposition
    Konstantin, Dragomiretskiy, and Dominique Zosso. "Two-dimensional variational mode decomposition."
    Energy Minimization Methods in Computer Vision and Pattern Recognition. Vol. 8932. 2015.
    """
    return np.loadtxt("./texture.txt")
