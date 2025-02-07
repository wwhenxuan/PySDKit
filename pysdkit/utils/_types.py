# -*- coding: utf-8 -*-
"""
Created on 2025/02/06 18:35:08
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com


这个模块要进行修正
"""
import numpy as np

from typing import Optional

__all__ = ["is_1d", "is_complex", "to_1d", "to_2d"]


def is_1d(x: np.ndarray) -> bool:
    """Check if the input sequence is one-dimensional"""
    if np.ndim(x) != 1:
        return False
    else:
        return True


def is_complex(x: np.ndarray) -> bool:
    """Check if the input sequence is a complex sequence"""
    return isinstance(x, (complex, np.complex64, np.complex128))


def to_1d(data: np.ndarray) -> np.ndarray:
    """
    Transform any data to a 1D numpy ndarray

    :param data: None, float, int or ndarray of any data type
    :return: the transformed 1D numpy ndarray
    """
    # Ensure the data type is correct
    data = np.asarray(data)

    # If the data has no dimensions, add one
    if np.ndim(data) == 0:
        return np.asarray([data])

    # If the data is already one-dimensional, return it directly
    elif np.ndim(data) == 1:
        return data

    # Handle more complex multi-dimensional data
    else:
        data = np.squeeze(data)
        if np.ndim(data) == 1:
            return data
        else:
            return np.ravel(data)


def to_2d(data: np.ndarray, column: Optional[bool] = False) -> np.ndarray:
    """
    Transform any data to a 2D numpy ndarray

    :param data: None, float, int or ndarray of any data type
    :param column: Whether to output a row vector or a column vector.
                   Determines where the new dimension is added.
    :return: the transformed 2D numpy ndarray
    """
    # Ensure the data type is correct
    data = np.asarray(data)

    # First, convert the data to a 1D vector for easier processing
    data = to_1d(data)

    if len(data.shape) == 1:
        # Add a new dimension at a specific position
        if column:
            return data[np.newaxis, :]
        else:
            return data[:, np.newaxis]
    else:
        return data
