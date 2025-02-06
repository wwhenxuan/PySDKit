# -*- coding: utf-8 -*-
"""
Created on 2025/02/02 16:47:10
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
对二维灰度图像进行可视化分析
"""
# -*- coding: utf-8 -*-
import numpy as np
from numpy import fft
from matplotlib import pyplot as plt

from typing import Optional, Tuple


def plot_grayscale(
    img: np.ndarray,
    figsize: Optional[Tuple] = (5, 5),
    dpi: Optional[int] = 100,
    cmap: Optional[str] = "gray",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    对二维灰度图像数据进行可视化
    :param img: 输入的numpy中的二维ndarray矩阵
    :param figsize: 绘制图像的大小
    :param dpi: 采用的分辨率，默认为100
    :param cmap: 使用的颜色映射
    :return: Figure and Axes from matplotlib
    """
    # 创建绘图对象
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(img, cmap=cmap)  # 可视化图像
    ax.set_aspect("equal")
    return fig, ax


def plot_grayscale_spectrum(
    img: np.ndarray,
    figsize: Optional[Tuple] = (5, 5),
    dpi: Optional[int] = 100,
    cmap: Optional[str] = "gray",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制二维灰度图像的频谱分布
    :param img: 输入的numpy中的二维ndarray矩阵
    :param figsize: 绘制图像的大小
    :param dpi: 采用的分辨率，默认为100
    :param cmap: 使用的颜色映射
    :return: Figure and Axes from matplotlib
    """
    # 创建绘图对象
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # 对输入图像进行二维快速傅里叶变换
    spectrum = np.abs(fft.fftshift(fft.fft2(img)))  # 获得功率谱
    ax.imshow(spectrum, cmap=cmap)  # 可视化图像
    ax.set_aspect("equal")
    return fig, ax
