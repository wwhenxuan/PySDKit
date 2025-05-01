# -*- coding: utf-8 -*-
"""
Created on Sat Mar 8 21:45:02 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from scipy.signal import sawtooth, chirp
from matplotlib import pyplot as plt

from typing import Tuple, Optional


def generate_sin_signal(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.2,
    frequency: float = 10.0,
    rand_state: int = 42,
) -> Tuple[np.array, np.array]:
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


def generate_cos_signal(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.2,
    frequency: float = 10.0,
    rand_state: int = 42,
) -> Tuple[np.array, np.array]:
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


def generate_square_wave(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.2,
    frequency: float = 10.0,
    rand_state: int = 42,
) -> Tuple[np.array, np.array]:
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
    square_wave = np.sign(
        np.sin(2 * np.pi * frequency * t)
    )  # Generate a basic square wave
    noise = np.random.normal(0, noise_level, square_wave.shape)  # Gaussian noise
    noisy_square_wave = square_wave + noise
    return t, noisy_square_wave


def generate_triangle_wave(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.2,
    frequency: float = 10.0,
    rand_state: int = 42,
) -> Tuple[np.array, np.array]:
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


def generate_sawtooth_wave(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.2,
    frequency: float = 10.0,
    rand_state: int = 42,
) -> Tuple[np.array, np.array]:
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


def generate_am_signal(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.1,
    carrier_freq: float = 100.0,
    modulating_freq: float = 5.0,
    mod_index: float = 1.0,
    rand_state: int = 42,
) -> Tuple[np.array, np.array]:
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


def generate_exponential_signal(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.1,
    decay_rate: float = 1.0,
    initial_amplitude: float = 1.0,
    rand_state: int = 42,
) -> Tuple[np.array, np.array]:
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


def test_univariate_signal(
    case: int = 1,
    duration: float = 1.0,
    sampling_rate: int = 1000,
) -> Tuple[np.array, np.array]:
    """
    Select a test case for a one-dimensional univariate signal based on the input `case`

    :param case: the test number in [1, 2, 3]
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.

    :return: Tuple containing time array and the generated signal.
    :return: the generated signal for univariate 1D.
    """
    if case == 1:
        return test_1D_1(duration, sampling_rate)
    elif case == 2:
        return test_1D_2(duration, sampling_rate)
    elif case == 3:
        return test_1D_3(duration, sampling_rate)
    else:
        # 当没有这个测试实例是返回`test_emd`这个函数
        print(f"There is no case {case}, so it will return test_emd!")
        return test_emd(duration, sampling_rate)


def test_1D_1(
    duration: float = 1.0, sampling_rate: int = 1000
) -> Tuple[np.array, np.array]:
    """
    4 / np.pi * (np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 30 * t) / 3 + np.sin(2 * np.pi * (50 * t + 20 * t ** 2)) / 2)

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.

    :return: Tuple containing time array and the generated signal.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = (
        4
        / np.pi
        * (
            np.sin(2 * np.pi * 10 * t)
            + np.sin(2 * np.pi * 30 * t) / 3
            + np.sin(2 * np.pi * (50 * t + 20 * t**2)) / 2
        )
    )
    return t, signal


def test_1D_2(
    duration: float = 10.0, sampling_rate: int = 128
) -> Tuple[np.array, np.array]:
    """
    Generate 3 * 2 ^ (-t) * sin(sin(2 * pi * t))

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.

    :return: Tuple containing time array and the 3 * 2 ^ (-t) * sin(sin(2 * pi * t)) signal.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = (
        3 * 2 ** (-t) * np.sin(np.sin(2 * np.pi * t))
        + np.cos(2 * np.pi * t * 5)
        + np.sin(2 * np.pi * (t + 2 * t**2))
        + np.sin(2 * np.pi * t * 30)
    )
    return t, signal


def test_1D_3(
    duration: float = 6.0,
    sampling_rate: int = 128,
    noise_level: float = 0.0,
    random_state: int = 42,
) -> Tuple[np.array, np.array]:
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
    signal = (
        5 * np.sin(2 * np.pi * t)
        + 3 * np.cos(2 * np.pi * t)
        + 2 * np.cos(2 * np.pi * t**2)
        + np.sin(2 * np.pi * t**3)
        + np.sin(2 * np.pi * (t * 30 + t**2 * 2))
    )
    noise = np.random.normal(0, noise_level, signal.shape)
    noise_signal = signal + noise
    return t, noise_signal


def test_emd(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.array, np.array]:
    """
    Generate cos(22 * pi * t ^ 2) + 6 * t ^ 2 for _emd test.

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param random_state: Random seed for the noise generation.

    :return: Tuple containing time array and the cos(22 * pi * t ^ 2) + 6 * t ^ 2 signal with noise.
    """
    np.random.seed(seed=random_state)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = np.cos(22 * np.pi * t**2) + 2 * t**2
    noise = np.random.normal(0, noise_level, signal.shape)
    noise_signal = signal + noise
    return t, noise_signal


def test_hht(duration: float = 2.0, sampling_rate: int = 1000):
    """
    Generate data generation function to verify Hilbert-Huang transform.

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.

    :return: Tuple of time array and signal for hht testing.
    """
    # Generate a sequence of time-stamped samples
    time = np.arange(sampling_rate * duration) / sampling_rate
    # Generate the signal to be decomposed
    signal = chirp(time, 5, 0.8, 10, method="quadratic", phi=100) * np.exp(
        -4 * (time - 1) ** 2
    ) + chirp(time, 40, 1.2, 50, method="linear") * np.exp(-4 * (time - 1) ** 2)
    return time, signal


def test_multivariate_signal(
    case: int = 1,
    duration: float = 1.0,
    sampling_rate: int = 1000,
) -> Tuple[np.array, np.array]:
    """
    Select a test case for a 1D multivariate signal based on the input `case`

    :param case: the test number in [1, 2, 3]
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.

    :return: Tuple containing time array and the generated signal.
    :return: the generated signal for multivariate 1D.
    """
    if case == 1:
        return test_multivariate_1D_1(duration=duration, sampling_rate=sampling_rate)
    elif case == 2:
        return test_multivariate_1D_2(duration=duration, sampling_rate=sampling_rate)
    elif case == 3:
        return test_multivariate_1D_3(duration=duration, sampling_rate=sampling_rate)
    else:
        # When there is no such test instance, the function returns `case==1`
        print(f"There is no case {case}, so it will return case==1!")
        return test_multivariate_1D_1(duration=duration, sampling_rate=sampling_rate)


def test_multivariate_1D_1(
    duration: float = 1.0, sampling_rate: int = 1000
) -> Tuple[np.array, np.array]:
    """
    Generate some simple cosine and sine function for multivariate signal decomposition test

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.

    :return: Tuple containing time array and the multivariate signals of shape [num_vars, seq_len].
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Generate the multi-channels signals
    signal_1 = np.cos(2 * np.pi * 5 * t + np.pi / 3) + 2.5 * np.cos(
        2 * np.pi * 36 * t + np.pi / 2
    )
    signal_2 = 3 * np.cos(2 * np.pi * 24 * t) + 2 * np.cos(2 * np.pi * 36 * t)

    # concat all channels
    signal = np.vstack([signal_1, signal_2])

    return t, signal


def test_multivariate_1D_2(
    duration: float = 1.0, sampling_rate: int = 1000
) -> Tuple[np.array, np.array]:
    """
    Generate four channels simple cosine and sine function for multivariate signal decomposition

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.

    :return: Tuple containing time array and the multivariate signals of shape [num_vars, seq_len].
    """
    np.random.seed(seed=42)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Generate the multi-channels signals
    signal_1 = (
        2 * np.cos(2 * np.pi * 3 * t**2)
        + 1.7 * np.cos(2 * np.pi * 20 * t)
        + np.cos(2 * np.pi * 40 * t)
        + np.random.uniform(0, 0.2, t.shape)
    )

    signal_2 = 1.5 * np.cos(2 * np.pi * 3 * t**2) + np.random.uniform(0, 0.2, t.shape)

    signal_3 = (
        2 * np.cos(2 * np.pi * 20 * t + np.pi / 3)
        + 1.8 * np.cos(2 * np.pi * 40 * t)
        + np.random.uniform(0, 0.2, t.shape)
    )

    signal_4 = (
        1.8 * np.cos(2 * np.pi * 3 * t**2)
        + 2 * np.cos(2 * np.pi * 40 * t)
        + np.random.uniform(0, 0.2, t.shape)
    )

    # Concat all the input channels
    signal = np.vstack([signal_1, signal_2, signal_3, signal_4])

    return t, signal


def test_multivariate_1D_3(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate some simple cosine function for multivariate signal decomposition test

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param random_state: Random seed for the noise generation.

    :return: Tuple containing time array and the multivariate signals of shape [num_vars, seq_len] with noise.
    """
    np.random.seed(seed=random_state)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Generate the multi-channels signals
    f_channel1 = (
        (10 * np.cos(2 * np.pi * 2 * t))
        + (9 * (np.cos(2 * np.pi * 36 * t)))
        + np.random.normal(0, noise_level, t.shape)
    )

    f_channel2 = (
        (9 * (np.cos(2 * np.pi * 24 * t)))
        + (8 * (np.cos(2 * np.pi * 36 * t)))
        + np.random.normal(0, noise_level, t.shape)
    )

    f_channel3 = (
        (8 * (np.cos(2 * np.pi * 28 * t)))
        + (7 * (np.cos(2 * np.pi * 48 * t)))
        + np.random.normal(0, noise_level, t.shape)
    )

    f_channel4 = (
        (7 * (np.cos(2 * np.pi * 32 * t)))
        + (6 * (np.cos(2 * np.pi * 36 * t)))
        + np.random.normal(0, noise_level, t.shape)
    )

    f_channel5 = (
        (6 * (np.cos(2 * np.pi * 19 * t)))
        + (5 * (np.cos(2 * np.pi * 64 * t)))
        + np.random.normal(0, noise_level, t.shape)
    )

    # concat all channels
    f = np.vstack([f_channel1, f_channel2, f_channel3, f_channel4, f_channel5])

    return t, f


if __name__ == "__main__":
    from pysdkit.data import test_emd, test_multivariate_signal
    from pysdkit.plot import plot_signal

    t, signal = test_univariate_signal(case=1)
    print(signal.shape)

    fig = plot_signal(t, signal)

    t, signal = test_multivariate_signal(case=2)
    print(signal.shape)

    fig = plot_signal(t, signal, save=False)
    plt.show()
