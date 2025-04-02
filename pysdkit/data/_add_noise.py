# -*- coding: utf-8 -*-
"""
Created on Sat Mar 8 21:45:02 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np


def add_noise(N: int, Mean: float, STD: float) -> np.ndarray:
    """
    Generate Gaussian white noise with mean values Mean and standard deviation STD.

    :param N: The number of samples.
    :param Mean: The mean value of the noise.
    :param STD: The standard deviation of the noise.

    :return: numpy.ndarray: Generated Gaussian white noise.
    """
    # Generate noise using the given mean and standard deviation
    y = np.random.randn(N)
    # Standardize y so that its standard deviation is 1
    y = y / np.std(y)
    # Adjust y so that its mean is 0
    y = y - np.mean(y)
    # Generate noise using the given mean and standard deviation
    y = Mean + STD * y

    return y
