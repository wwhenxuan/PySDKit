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
from ._plot_images import plot_grayscale_image, plot_grayscale_spectrum

# 通用的图像可视化函数
# 可以选择是否绘制频域图谱以及是否添加颜色条
from ._plot_images import plot_images

# Functions that generate signal visualizations
from ._plot_signal import plot_signal
