# -*- coding: utf-8 -*-
"""
Created on 2024/6/3 15:31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

This py includes a series of functional modules
"""

import numpy as np


def max_min_normalization(x: np.ndarray) -> np.ndarray:
    """
    Perform min-max normalization on the input signal
    :param x: Input 1D sequence
    :return: Normalized sequence
    """
    return (x - x.min()) / (x.max() - x.min())


def z_score_normalization(x: np.ndarray) -> np.ndarray:
    """
    Perform Z-score normalization on the input signal
    :param x: Input 1D sequence
    :return: Normalized sequence
    """
    return (x - x.mean()) / x.std()


def max_absolute_normalization(x: np.ndarray) -> np.ndarray:
    """
    Perform Max Absolute normalization on the input signal
    :param x: Input 1D sequence
    :return: Normalized sequence
    """
    return x / np.max(np.abs(x))


def log_transformation(x: np.ndarray) -> np.ndarray:
    """
    Perform log transformation on the input signal
    :param x: Input 1D sequence
    :return: Transformed sequence
    """
    return np.log1p(x)


def decimal_scaling_normalization(x: np.ndarray) -> np.ndarray:
    """
    Perform Decimal Scaling normalization on the input signal
    :param x: Input 1D sequence
    :return: Normalized sequence
    """
    max_abs_value = np.max(np.abs(x))
    scaling_factor = 10 ** np.ceil(np.log10(max_abs_value + 1))
    return x / scaling_factor

