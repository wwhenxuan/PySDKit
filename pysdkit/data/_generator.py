# -*- coding: utf-8 -*-
"""
Created on Sat Mar 8 21:45:02 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from scipy.signal import sawtooth
from matplotlib import pyplot as plt

from typing import Tuple


def generate_sin_signal(duration: float = 1.0, sampling_rate: int = 1000, noise_level: float = 0.2,
                        frequency: float = 10.0, rand_state: int = 42) -> Tuple[np.array, np.array]:
    """
    Generate a Cosine signal with Gaussian noise and a sinusoidal component.
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param frequency: Frequency of the sinusoidal component.
    :param rand_state: Random seed for the noise in signal.
    :return: Array of the Cosine signal.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)  # Sinusoidal signal
    np.random.seed(seed=rand_state)
    noise = np.random.normal(0, noise_level, signal.shape)  # Gaussian noise
    emg_signal = signal + noise  # Superimpose signal and noise
    return t, emg_signal


def generate_cos_signal(duration: float = 1.0, sampling_rate: int = 1000, noise_level: float = 0.2,
                        frequency: float = 10.0, rand_state: int = 42) -> Tuple[np.array, np.array]:
    """
    Generate a Cosine signal with Gaussian noise and a sinusoidal component.
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param frequency: Frequency of the sinusoidal component.
    :param rand_state: Random seed for the noise in signal.
    :return: Array of the Cosine signal.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.cos(2 * np.pi * frequency * t)  # Sinusoidal signal
    np.random.seed(seed=rand_state)
    noise = np.random.normal(0, noise_level, signal.shape)  # Gaussian noise
    emg_signal = signal + noise  # Superimpose signal and noise
    return t, emg_signal


def generate_square_wave(duration: float = 1.0, sampling_rate: int = 1000, noise_level: float = 0.2,
                         frequency: float = 10.0, rand_state: int = 42) -> Tuple[np.array, np.array]:
    """
    Generate a square wave signal with Gaussian noise.
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param frequency: Frequency of the square wave component.
    :param rand_state: Random seed for the noise in signal.
    :return: Tuple containing time array and the square wave signal with noise.
    """
    np.random.seed(rand_state)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    square_wave = np.sign(np.sin(2 * np.pi * frequency * t))  # Generate a basic square wave
    noise = np.random.normal(0, noise_level, square_wave.shape)  # Gaussian noise
    noisy_square_wave = square_wave + noise
    return t, noisy_square_wave


def generate_triangle_wave(duration: float = 1.0, sampling_rate: int = 1000, noise_level: float = 0.2,
                           frequency: float = 10.0, rand_state: int = 42) -> Tuple[np.array, np.array]:
    """
    Generate a triangular wave signal with Gaussian noise.
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param frequency: Frequency of the triangular wave component.
    :param rand_state: Random seed for the noise in signal.
    :return: Tuple containing time array and the triangular wave signal with noise.
    """
    np.random.seed(rand_state)  # Set random seed for reproducibility
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # Generate the triangular wave using the sawtooth function and converting it to a triangular shape
    from scipy.signal import sawtooth
    triangle_wave = 2 * np.abs(sawtooth(2 * np.pi * frequency * t, 0.5)) - 1
    noise = np.random.normal(0, noise_level, triangle_wave.shape)  # Add Gaussian noise
    noisy_triangle_wave = triangle_wave + noise  # Combine triangle wave with noise
    return t, noisy_triangle_wave


def generate_sawtooth_wave(duration: float = 1.0, sampling_rate: int = 1000, noise_level: float = 0.2,
                           frequency: float = 10.0, rand_state: int = 42) -> Tuple[np.array, np.array]:
    """
    Generate a sawtooth wave signal with Gaussian noise.
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param frequency: Frequency of the sawtooth wave component.
    :param rand_state: Random seed for the noise in signal.
    :return: Tuple containing time array and the sawtooth wave signal with noise.
    """
    np.random.seed(rand_state)  # Set the random seed for reproducibility
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    saw_wave = sawtooth(2 * np.pi * frequency * t)  # Generate a pure sawtooth wave
    noise = np.random.normal(0, noise_level, saw_wave.shape)  # Add Gaussian noise
    noisy_saw_wave = saw_wave + noise  # Combine sawtooth wave with noise
    return t, noisy_saw_wave


def generate_am_signal(duration: float = 1.0, sampling_rate: int = 1000, noise_level: float = 0.1,
                       carrier_freq: float = 100.0, modulating_freq: float = 5.0, mod_index: float = 1.0,
                       rand_state: int = 42) -> Tuple[np.array, np.array]:
    """
    Generate an Amplitude Modulated (AM) signal with Gaussian noise.
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param carrier_freq: Frequency of the carrier signal.
    :param modulating_freq: Frequency of the modulating signal.
    :param mod_index: Modulation index.
    :param rand_state: Random seed for the noise generation.
    :return: Tuple containing time array and the AM signal with noise.
    """
    np.random.seed(rand_state)  # Set the random seed for reproducibility
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # Carrier signal
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    # Modulating signal
    modulating_signal = np.cos(2 * np.pi * modulating_freq * t)
    # Amplitude Modulated signal
    am_signal = (1 + mod_index * modulating_signal) * carrier
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, am_signal.shape)
    noisy_am_signal = am_signal + noise  # AM signal with noise
    return t, noisy_am_signal


def generate_exponential_signal(duration: float = 1.0, sampling_rate: int = 1000, noise_level: float = 0.1,
                                decay_rate: float = 1.0, initial_amplitude: float = 1.0,
                                rand_state: int = 42) -> Tuple[np.array, np.array]:
    """
    Generate an exponentially decaying signal with Gaussian noise.
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param decay_rate: Exponential decay rate (larger values decay faster).
    :param initial_amplitude: Initial amplitude of the signal.
    :param rand_state: Random seed for the noise generation.
    :return: Tuple containing time array and the exponentially decaying signal with noise.
    """
    np.random.seed(rand_state)  # Set the random seed for reproducibility
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    # Exponentially decaying signal
    exp_signal = initial_amplitude * np.exp(-decay_rate * t)
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, exp_signal.shape)
    noisy_exp_signal = exp_signal + noise  # Exponential signal with noise
    return t, noisy_exp_signal


def base_example(duration: float = 6.0, sampling_rate: int = 128, noise_level: float = 0.0,
                 random_state: int = 42) -> Tuple[np.array, np.array]:
    """
    Generate 5 * sin(2 * pi * t) + 3 * sin(2 * pi * t)
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param random_state: Random seed for the noise generation.
    :return: Tuple containing time array and the 5 * sin(2 * pi * t) + 3 * sin(2 * pi * t) signal with noise.
    """
    np.random.seed(seed=random_state)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = 5 * np.sin(2 * np.pi * t) + 3 * np.sin(2 * np.pi * t)
    noise = np.random.normal(0, noise_level, signal.shape)
    noise_signal = signal + noise
    return t, noise_signal


def fun(duration: float = 10.0, sampling_rate: int = 128) -> Tuple[np.array, np.array]:
    """
    Generate 3 * 2 ^ (-t) * sin(sin(2 * pi * t))
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :return: Tuple containing time array and the 3 * 2 ^ (-t) * sin(sin(2 * pi * t)) signal.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = 3 * 2 ^ (-t) * np.sin(np.sin(2 * np.pi * t))
    return t, signal


def test_emd(duration: float = 1.0, sampling_rate: int = 1000, noise_level: float = 0.1,
             random_state: int = 42) -> Tuple[np.array, np.array]:
    """
    Generate cos(22 * pi * t ^ 2) + 6 * t ^ 2 for emd test.
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param random_state: Random seed for the noise generation.
    :return: Tuple containing time array and the cos(22 * pi * t ^ 2) + 6 * t ^ 2 signal with noise.
    """
    np.random.seed(seed=random_state)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.cos(22 * np.pi * t ** 2) + 2 * t ** 2
    noise = np.random.normal(0, noise_level, signal.shape)
    noise_signal = signal + noise
    return t, noise_signal


def plot_generate_signal(t: np.array, signal: np.array, color: str = 'royalblue', save: bool = False,
                         figsize: Tuple = (12, 4), dpi: int = 600, fontsize: int = 15) -> plt.figure:
    """
    Plot and optionally save an amplitude modulated (AM) signal with time on the x-axis and amplitude on the y-axis.
    :param t: Array of time points corresponding to the signal.
    :param signal: Array containing the signal data to be plotted.
    :param color: Color of the plot line.
    :param save: Boolean flag to indicate whether the plot should be saved to a file.
    :param figsize: Tuple representing the width and height of the figure in inches.
    :param dpi: Dots per inch (resolution) of the figure, if saved.
    :param fontsize: Font size of the axis labels.
    :return: The figure object containing the plot.
    """
    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=figsize)
    # Plot the signal against time with specified line color
    ax.plot(t, signal, color=color)
    # Set the x-axis label with specified font size
    ax.set_xlabel('Time (seconds)', fontsize=fontsize)
    # Set the y-axis label with default font size
    ax.set_ylabel('Amplitude')
    # Enable grid for better readability
    ax.grid(True)
    # Display the plot
    plt.show()
    # Check if the figure needs to be saved
    if save:
        # Save the figure with the specified dpi and bounding box tightly around the figure
        fig.savefig("generate_signal.jpg", dpi=dpi, bbox_inches='tight')
    # Return the figure object
    return fig
