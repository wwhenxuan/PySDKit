# -*- coding: utf-8 -*-
"""
Created on 2025/02/06 10:30:01
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

# Fourier Transform
from ._fft import (
    fft,
    ifft,
    fftshift,
    ifftshift,
)

# Fast Fourier transform of two-dimensional image data
from ._fft import fft2d, ifft2d

# Signal mirroring extension
from ._mirror import fmirror

# Various functions for Hilbert Transform
from ._hilbert import (
    hilbert_transform,
    hilbert_real,
    hilbert_imaginary,
    hilbert_spectrum,
)
from ._hilbert import (
    plot_hilbert,
    plot_hilbert_complex_plane,
)

from ._process import normalize_signal
from ._process import common_dtype
from ._process import not_duplicate

# Find the zero crossing points of the signal
from ._process import find_zero_crossings

# Get the time axis of the signal and generate a timestamp array
from ._process import get_timeline

# Algorithm for discrete signal differencing
from ._differ import differ

# Functions for 1D signal smoothing
from ._smooth1d import (
    simple_moving_average,
    weighted_moving_average,
)

# Gaussian smoothing, Savitzky-Golay smoothing, and exponential smoothing
from ._smooth1d import (
    gaussian_smoothing,
    savgol_smoothing,
    exponential_smoothing,
)

# Min-max normalization
from ._function import max_min_normalization

# Z-score standardization
from ._function import z_score_normalization

# Max-absolute normalization
from ._function import max_absolute_normalization

# Logarithmic transformation
from ._function import log_transformation

# Decimal scaling normalization
from ._function import decimal_scaling_normalization

# 寻找一维信号的极值点
from ._instantaneous import find_extrema

# Find the instantaneous amplitude and instantaneous frequency of a one-bit input signal
from ._instantaneous import inst_freq_local

# Decompose the input one-dimensional signal into two eigenmode functions
from ._instantaneous import divide2exp

# Any pair of IMFs is locally orthogonal
from ._function import index_of_orthogonality

# Determine whether an array is one-dimensional
from ._function import is_1d

# Determine whether an array is a complex array
from ._function import is_complex

# Convert all input data into one-dimensional form
from ._function import to_1d

# Add a new axis to the 1D ndarray
from ._function import to_2d

# This function generates the lag matrix of a signal, also known as the data matrix or correlation matrix
from ._function import lags_matrix

# This function calculates the covariance matrix of the input signal's lag matrix
from ._function import covariance_matrix

# This function is used to generate the kernel matrix of the input signal
from ._kernel_matrix import kernel_matrix

# Matrix of euclidian distance I.E. Pairwise distance matrix
from ._kernel_matrix import euclidian_matrix

# Perform Hankel averaging (or diagonal averaging) on the input matrix
from ._diagnalization import diagonal_average

# Extract the specified diagonal from a matrix
from ._diagnalization import get_diagonal
