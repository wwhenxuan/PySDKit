# -*- coding: utf-8 -*-
"""
Created on 2024/6/3 15:31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def simple_moving_average(signal: np.ndarray, window_size: int = 2) -> np.ndarray:
    """
    Simple Moving Average
    :param signal: Input signal (numpy array)
    :param window_size: Window size for averaging (default is 2)
    :return: Smoothed signal (numpy array)
    """
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')


def weighted_moving_average(signal: np.ndarray, window_size: int = 2) -> np.ndarray:
    """
    Weighted Moving Average
    :param signal: Input signal (numpy array)
    :param window_size: Window size for averaging (default is 2)
    :return: Smoothed signal (numpy array)
    """
    weights = np.arange(1, window_size + 1)
    return np.convolve(signal, weights / weights.sum(), mode='same')


def gaussian_smoothing(signal: np.ndarray, sigma: int = 2) -> np.ndarray:
    """
    Gaussian Filtering Smoothing
    :param signal: Input signal (numpy array)
    :param sigma: Standard deviation for Gaussian kernel (default is 2)
    :return: Smoothed signal (numpy array)
    """
    return gaussian_filter1d(signal, sigma=sigma)


def savgol_smoothing(signal: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """
    Savitzky-Golay Filtering Smoothing
    :param signal: Input signal (numpy array)
    :param window_length: Length of the filter window (default is 11, must be odd)
    :param polyorder: Order of the polynomial used to fit the samples (default is 2)
    :return: Smoothed signal (numpy array)
    """
    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)


def exponential_smoothing(signal: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Exponential Smoothing (Single Exponential Smoothing)
    :param signal: Input signal (numpy array)
    :param alpha: Smoothing factor, range from 0 to 1 (default is 0.4)
    :return: Smoothed signal (numpy array)
    """
    smoothed_signal = np.zeros_like(signal)
    # Initial smoothed value set to the first data point
    smoothed_signal[0] = signal[0]
    for t in range(1, len(signal)):
        smoothed_signal[t] = alpha * signal[t] + (1 - alpha) * smoothed_signal[t - 1]
    return smoothed_signal


def smooth_show_info():
    """
    Visualize different signal smoothing methods on a test signal
    """
    import matplotlib.pyplot as plt
    # Generate test signal
    np.random.seed(0)
    x = np.linspace(0, 2 * np.pi, 100)
    signal = np.sin(x) + np.random.normal(0, 0.1, x.size)

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(12, 5))

    # Apply signal smoothing algorithms and visualize
    ax.plot(x, signal, label='Original Signal', alpha=0.8)
    ax.plot(x, simple_moving_average(signal=signal, window_size=5), label='Simple Moving Average', alpha=0.7)
    ax.plot(x, weighted_moving_average(signal=signal, window_size=5), label='Weighted Moving Average', alpha=0.7)
    ax.plot(x, gaussian_smoothing(signal=signal, sigma=3), label='Gaussian Filtering', alpha=0.7)
    ax.plot(x, savgol_smoothing(signal=signal), label='Savitzky-Golay Filtering', alpha=0.7)
    ax.plot(x, exponential_smoothing(signal=signal, alpha=0.4), label='Exponential Smoothing', alpha=0.7)

    # Add legend
    ax.legend(loc="best", fontsize=11)

    return fig
