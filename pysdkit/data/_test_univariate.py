# -*- coding: utf-8 -*-
"""
Created on 2025/07/18 10:39:38
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Tuple


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
        return test_univariate_1(duration, sampling_rate)
    elif case == 2:
        return test_univariate_2(duration, sampling_rate)
    elif case == 3:
        return test_univariate_3(duration, sampling_rate)
    else:
        # 当没有这个测试实例是返回`test_emd`这个函数
        print(f"There is no case {case}, so it will return test_emd!")
        return test_emd(duration, sampling_rate)


def test_univariate_1(
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


def test_univariate_2(
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


def test_univariate_3(
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


def test_univariate_nonlinear_chip(
    case: int = 1,
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.array, np.array]:
    """
    Select a test case for a one-dimensional nonlinear chip univariate signal based on the input `case`

    :param case: the test number in [1, 2]
    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param random_state: Random seed for the noise generation.
    :return: the generated non-linear chip signal.
    """
    case = int(case)
    if case == 1:
        t, signal = test_nonlinear_chip_1(
            duration=duration,
            sampling_rate=sampling_rate,
            noise_level=noise_level,
            random_state=random_state,
        )
    elif case == 2:
        t, signal = test_nonlinear_chip_2(
            duration=duration,
            sampling_rate=sampling_rate,
            noise_level=noise_level,
            random_state=random_state,
        )
    else:
        print(f"There is no test number {case}, return case=1!")
        t, signal = test_nonlinear_chip_1(
            duration=duration,
            sampling_rate=sampling_rate,
            noise_level=noise_level,
            random_state=random_state,
        )

    return t, signal


def test_nonlinear_chip_1(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.array, np.array]:
    """
    Generate the test example for non-linear chip test.
    a1 = exp(-0.03 * t)
    a2 = exp(-0.06 * t)
    s_1t = a1 * cos(2 * pi * (0.8 + 25 * t + 4 * t ** 2 - 1 * t ** 3 + 0.1 * t ** 4))
    s_2t = a2 * cos(2 * pi * (1 + 40 * t + 8 * t ** 2 - 2 * t ** 3 + 0.1 * t ** 4))
    signal = s_1t + s_2t

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param random_state: Random seed for the noise generation.
    :return: the generated non-linear chip signal.
    """
    np.random.seed(seed=random_state)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Instantaneous amplitudes (IAs) and instantaneous frequencies (IFs)
    a1 = np.exp(-0.03 * t)
    a2 = np.exp(-0.06 * t)

    # A two-component simulated nonlinear chirp signal (NCS)
    s_1t = a1 * np.cos(2 * np.pi * (0.8 + 25 * t + 4 * t**2 - 1 * t**3 + 0.1 * t**4))
    s_2t = a2 * np.cos(2 * np.pi * (1 + 40 * t + 8 * t**2 - 2 * t**3 + 0.1 * t**4))
    signal = s_1t + s_2t

    # Add normal noise to the generated signal
    noise = np.random.normal(0, noise_level, signal.shape)
    noise_signal = signal + noise

    return t, noise_signal


def test_nonlinear_chip_2(
    duration: float = 1.0,
    sampling_rate: int = 1000,
    noise_level: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.array, np.array]:
    """
    Generate the test example for non-linear chip test.
    a1 = 1 + 0.5 * cos(2 * pi * t)
    a2 = 1 - 0.5 * cos(2 * pi * t)
    s_1t = a1 * cos(2 * pi * (100 * t + 150 * t ** 2 + sin(9 * pi * t)))
    s_2t = a2 * cos(2 * pi * (400 * t - 150 * t ** 2 + sin(9 * pi * t)))
    signal = s_1t + s_2t

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param random_state: Random seed for the noise generation.
    :return: the generated non-linear chip signal.
    """
    np.random.seed(seed=random_state)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

    # Instantaneous amplitudes (IAs) and instantaneous frequencies (IFs)
    a1 = 1 + 0.5 * np.cos(2 * np.pi * t)
    a2 = 1 - 0.5 * np.cos(2 * np.pi * t)

    # A two-component simulated nonlinear chirp signal (NCS)
    s_1t = a1 * np.cos(2 * np.pi * (100 * t + 150 * t**2 + np.sin(9 * np.pi * t)))
    s_2t = a2 * np.cos(2 * np.pi * (400 * t - 150 * t**2 + np.sin(9 * np.pi * t)))
    signal = s_1t + s_2t

    # Add normal noise to the generated signal
    noise = np.random.normal(0, noise_level, signal.shape)
    noise_signal = signal + noise

    return t, noise_signal


def test_univariate_gaussamp_quadfm(
    duration: float = 6.0,
    sampling_rate: int = 128,
    noise_level: float = 0.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a Gaussian–modulated quadratic chirp:
        s(t) = 4 * exp(-((t-3)/1.5)**2) * sin(2π*(10*t + 3*t**2))

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param random_state: Random seed for the noise generation.

    :return: Tuple (t, noisy_signal)
    """
    np.random.seed(seed=random_state)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    envelope = 4 * np.exp(-(((t - 3) / 1.5) ** 2))
    signal = envelope * np.sin(2 * np.pi * (10 * t + 3 * t**2))
    noise = np.random.normal(0, noise_level, signal.shape)
    return t, signal + noise


def test_univariate_duffing(
    duration: float = 6.0,
    sampling_rate: int = 128,
    noise_level: float = 0.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a signal governed by a softening Duffing-type ODE:
        x'' + 0.3 x' + 4 x - 0.6 x^3 = 2 cos(2π*1.1 t)
    (integrated via 4th-order Runge–Kutta, initial conditions x(0)=0, x'(0)=0)

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param random_state: Random seed for the noise generation.

    :return: Tuple (t, noisy_signal)
    """
    np.random.seed(seed=random_state)
    dt = 1.0 / sampling_rate
    n = int(duration * sampling_rate)
    t = np.arange(n) * dt
    x = np.zeros(n)
    v = np.zeros(n)

    omega = 2 * np.pi * 1.1
    for k in range(1, n):
        a = (
            -0.3 * v[k - 1]
            - 4 * x[k - 1]
            + 0.6 * x[k - 1] ** 3
            + 2 * np.cos(omega * t[k - 1])
        )
        v1 = v[k - 1]
        x1 = x[k - 1]
        v2 = v[k - 1] + 0.5 * dt * a
        x2 = x[k - 1] + 0.5 * dt * v1
        a2 = (
            -0.3 * v2 - 4 * x2 + 0.6 * x2**3 + 2 * np.cos(omega * (t[k - 1] + 0.5 * dt))
        )

        v3 = v[k - 1] + 0.5 * dt * a2
        x3 = x[k - 1] + 0.5 * dt * v2
        a3 = (
            -0.3 * v3 - 4 * x3 + 0.6 * x3**3 + 2 * np.cos(omega * (t[k - 1] + 0.5 * dt))
        )

        v4 = v[k - 1] + dt * a3
        x4 = x[k - 1] + dt * v3
        a4 = -0.3 * v4 - 4 * x4 + 0.6 * x4**3 + 2 * np.cos(omega * t[k])

        x[k] = x[k - 1] + dt / 6 * (v1 + 2 * v2 + 2 * v3 + v4)
        v[k] = v[k - 1] + dt / 6 * (a + 2 * a2 + 2 * a3 + a4)

    signal = x
    noise = np.random.normal(0, noise_level, signal.shape)
    return t, signal + noise


def test_univariate_logistic_am(
    duration: float = 6.0,
    sampling_rate: int = 128,
    noise_level: float = 0.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a carrier modulated by the chaotic logistic map:
        a_{k+1} = 3.9 * a_k * (1 - a_k),  a_0 = 0.5
        s(t_k) = 3 * a_k * sin(2π * 8 * t_k)

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param random_state: Random seed for the noise generation.

    :return: Tuple (t, noisy_signal)
    """
    np.random.seed(seed=random_state)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    n = len(t)
    a = np.empty(n)
    a[0] = 0.5
    for k in range(1, n):
        a[k] = 3.9 * a[k - 1] * (1 - a[k - 1])
    signal = 3 * a * np.sin(2 * np.pi * 8 * t)
    noise = np.random.normal(0, noise_level, signal.shape)
    return t, signal + noise


def test_univariate_cubic_quad(
    duration: float = 6.0,
    sampling_rate: int = 128,
    noise_level: float = 0.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a signal with quadratic & cubic coupling:
        s(t) = 2 * sin(2π*3*t) + 0.8 * sin(2π*5*t)^2 + 0.5 * sin(2π*7*t)^3

    :param duration: Length of the signal in seconds.
    :param sampling_rate: Number of samples per second.
    :param noise_level: Standard deviation of the Gaussian noise.
    :param random_state: Random seed for the noise generation.

    :return: Tuple (t, noisy_signal)
    """
    np.random.seed(seed=random_state)
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    signal = (
        2 * np.sin(2 * np.pi * 3 * t)
        + 0.8 * np.sin(2 * np.pi * 5 * t) ** 2
        + 0.5 * np.sin(2 * np.pi * 7 * t) ** 3
    )
    noise = np.random.normal(0, noise_level, signal.shape)
    return t, signal + noise
