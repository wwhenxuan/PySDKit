# -*- coding: utf-8 -*-
"""
Created on Sat Mar 4 21:31:05 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
Some auxiliary function modules for data visualization in the PySDKit library
"""
# 在二维平面上可视化原输入信号和分解得到的本征模态函数
from ._plot_imfs import plot_IMFs

# 单独绘制分解得到的每个本征模态函数的频谱
from ._fourier_spectra import plot_IMFs_amplitude_spectra

# 用于可视化二维灰度图
from ._plot_GrayScale import plot_grayscale_image, plot_grayscale_spectrum
