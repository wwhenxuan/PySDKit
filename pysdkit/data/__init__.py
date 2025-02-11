# -*- coding: utf-8 -*-
"""
Created on 2024/7/22 22:56
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

# A series of functions for generating 1D NumPy signals
from ._generator import generate_sin_signal, generate_cos_signal
from ._generator import (
    generate_square_wave,
    generate_triangle_wave,
    generate_sawtooth_wave,
)
from ._generator import generate_am_signal, generate_exponential_signal

# Generator for 1D univariate time series data
from ._time_series import generate_time_series

# Generates the main test sample signal
from ._generator import test_emd

# Generator for 1D univariate
from ._generator import test_univariate_signal

# Generator for 1D multivariate
from ._generator import test_multivariate_signal

# Functions that generate signal visualizations
from ._generator import plot_signal

# add Gaussian noise
from ._add_noise import add_noise

# Test case for loading a 2D grayscale image
from ._image import test_grayscale

# Test case for univariate 2D image
from ._image import test_univariate_image

# Test case for multivariate 2D image
from ._image import test_multivariate_image

# Test case for univariate 3D cube
from ._cube import test_univariate_cube

# Test case for multivariate 3D cube
from ._cube import test_multivariate_cube
