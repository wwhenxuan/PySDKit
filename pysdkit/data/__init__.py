# -*- coding: utf-8 -*-
"""
Synthetic generators and packaged demo-data loaders.
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
    "test_fmd",
    "test_vmd",
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
    "load_prefixed_filter",
    "load_imd_input_sig",
    "load_imd_gearbox_snippet",
    "load_vme_ecg_055m",
    "load_dual_signal_noise",
    "load_single_nsignal",
    "load_map2",
    "load_wind_demo",
    "load_memd_syn_12channel",
    "load_memd_syn_16channel",
    "load_memd_syn_hex",
    "load_memd_taichi_hex",
    "load_apitmemd_section_2b",
    "load_apitmemd_section_3a",
    "load_apitmemd_section_3b",
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

# Official MATLAB FMD demo vibration record
from ._loaders import test_fmd

# Packaged VMD multi-component demo signal
from ._loaders import test_vmd

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

# Packaged demo arrays used by ALIF, IMD, VME, VTFMTD and ESMD
from ._loaders import (
    load_prefixed_filter,
    load_imd_input_sig,
    load_imd_gearbox_snippet,
    load_vme_ecg_055m,
    load_dual_signal_noise,
    load_single_nsignal,
    load_map2,
    load_wind_demo,
    load_memd_syn_12channel,
    load_memd_syn_16channel,
    load_memd_syn_hex,
    load_memd_taichi_hex,
    load_apitmemd_section_2b,
    load_apitmemd_section_3a,
    load_apitmemd_section_3b,
)
