# -*- coding: utf-8 -*-
"""
Created on 2024/7/22 22:56
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

# A series of functions for generating 1D NumPy signals
from ._generator import generate_sin_signal, generate_cos_signal
from ._generator import generate_square_wave, generate_triangle_wave, generate_sawtooth_wave
from ._generator import generate_am_signal, generate_exponential_signal

# Generates the main test sample signal
from ._generator import test_emd

# Functions that generate signal visualizations
from ._generator import plot_generate_signal

# add Gaussian noise
from ._add_noise import add_noise

# 加载二维灰度图像的测试案例
from .image import test_grayscale
