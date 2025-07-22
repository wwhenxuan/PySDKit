# -*- coding: utf-8 -*-
"""
Created on 2024/7/22 22:56
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
__all__ = [
    "generate_sin_signal",
    "generate_cos_signal",
    "generate_time_series",
    "generate_square_wave",
    "generate_triangle_wave",
    "generate_sawtooth_wave",
    "generate_am_signal",
    "generate_exponential_signal",
    "test_emd",
    "test_hht",
    "add_noise",
    "test_univariate_signal",
    "test_univariate_nonlinear_chip",
    "test_univariate_gaussamp_quadfm",
    "test_univariate_duffing",
    "test_univariate_logistic_am",
    "test_univariate_cubic_quad",
    "test_multivariate_signal",
    "get_meshgrid_2D",
    "test_grayscale",
    "test_univariate_image",
    "test_multivariate_image",
    "test_univariate_cube",
    "test_multivariate_cube",
    "test_pca",
]

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

# Generate the main test sample signal
from ._generator import test_emd

# Generate the test function for HHT
from ._generator import test_hht

# add Gaussian noise
from ._add_noise import add_noise

# Generator for 1D univariate
from ._test_univariate import test_univariate_signal

# Generator for 1D univariate nonlinear chip signal
from ._test_univariate import test_univariate_nonlinear_chip

# Generator for other univariate with different characterization
from ._test_univariate import (
    test_univariate_gaussamp_quadfm,
    test_univariate_duffing,
    test_univariate_logistic_am,
    test_univariate_cubic_quad,
)

# Generator for 1D multivariate
from ._generator import test_multivariate_signal

# Generate a grid matrix given an input and output range
from ._image import get_meshgrid_2D

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

# Test case for Principal Component Analysis
from ._models import test_pca