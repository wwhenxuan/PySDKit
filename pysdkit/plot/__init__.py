# -*- coding: utf-8 -*-
"""
Created on Sat Mar 4 21:31:05 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
Some auxiliary function modules for data visualization in the PySDKit library
"""
# Visualize the original input signal and the decomposed IMFs on a 2D plane
from ._plot_imfs import plot_IMFs

# Plot the spectrum of each decomposed IMF separately
from ._fourier_spectra import plot_IMFs_amplitude_spectra

# Plotting the Hilbert spectrum
from ._fourier_spectra import plot_HilbertSpectrum

# Used to visualize two-dimensional grayscale images
from ._plot_images import plot_grayscale_image, plot_grayscale_spectrum

# General image visualization functions
from ._plot_images import plot_images

# Functions that generate signal visualizations
from ._plot_signal import plot_signal
