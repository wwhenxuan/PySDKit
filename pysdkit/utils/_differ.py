# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 22:05:02 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np


def differ(y: np.ndarray, delta: float, dtype: np.dtype = np.float64) -> np.ndarray:
    """
    Compute the derivative of a discrete time series y.

    :param y: The input time series.
    :param delta: The sampling time interval of y.
    :param dtype: The data type of numpy array
    :return: numpy.ndarray: The derivative of the time series.
    """
    L = len(y)
    ybar = np.zeros(L - 2, dtype=dtype)

    for i in range(1, L - 1):
        ybar[i - 1] = (y[i + 1] - y[i - 1]) / (2 * delta)

    # Prepend and append the boundary differences
    ybar = np.concatenate(
        (
            np.array([(y[1] - y[0]) / delta], dtype=dtype),
            ybar,
            np.array([(y[-1] - y[-2]) / delta], dtype=dtype),
        )
    )

    return ybar
