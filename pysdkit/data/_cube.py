# -*- coding: utf-8 -*-
"""
Created on 2025/02/11 00:19:03
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
Generate 3D cube test sample
"""
import numpy as np

from typing import Optional, Union, Tuple, List


def get_meshgrid_3D(
    low: Union[int, np.ndarray], high: Union[int, np.ndarray], sampling_rate: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a grid 3D cube given an input and output range

    :param low: Minimum value of grid matrix,
        If it is a ndarray array, its length must be 3 to indicate the range of three dimensions.
    :param high: Maximum value of grid matrix,
        If it is a ndarray array, its length must be 3 to indicate the range of three dimensions.
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix

    :return: A grid matrix generated from a range in the form of a tuple
    """
    # First obtain a one-dimensional range array by specifying the ranges `low` and `high`
    if isinstance(low, np.ndarray) and isinstance(high, np.ndarray):
        # Both range parameters are Numpy arrays
        X = np.linspace(low[0], high[0], sampling_rate)
        Y = np.linspace(low[1], high[1], sampling_rate)
        Z = np.linspace(low[2], high[2], sampling_rate)

    elif isinstance(low, np.ndarray) and isinstance(high, int):
        # `low` is a Numpy array `high` is a constant
        X = np.linspace(low[0], high, sampling_rate)
        Y = np.linspace(low[1], high, sampling_rate)
        Z = np.linspace(low[2], high, sampling_rate)

    elif isinstance(low, int) and isinstance(high, np.ndarray):
        # `low` is a constant `high` is a Numpy array
        X = np.linspace(low, high[0], sampling_rate)
        Y = np.linspace(low, high[1], sampling_rate)
        Z = np.linspace(low, high[2], sampling_rate)

    elif isinstance(low, int) and isinstance(high, int):
        # Both range parameters are constants
        X = np.linspace(low, high, sampling_rate)
        Y = np.linspace(low, high, sampling_rate)
        Z = np.linspace(low, high, sampling_rate)

    else:
        raise TypeError("The params `low` and `high` must be int or ndarray of numpy!")

    # Further generate network cube
    x, y, z = np.meshgrid(X, Y, Z)

    return x, y, z


def test_univariate_cube(
    case: int = 1,
    low: Union[int, np.ndarray] = 0,
    high: Union[int, np.ndarray] = 10,
    sampling_rate: int = 30,
) -> np.ndarray:
    """
    Generate a 3D cube data for testing 3D univariate signal decomposition algorithms

    :param case: Test case, range in [1, 2, 3, 4, 5, 6]
    :param low: Minimum value of grid matrix,
        If it is a ndarray array, its length must be 3 to indicate the range of three dimensions.
    :param high: Maximum value of grid matrix,
        If it is a ndarray array, its length must be 3 to indicate the range of three dimensions.
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix
    :return: Get a 3D cube data of a specified range
    """
    # Generate data based on the input test sample number
    if case == 1:
        return test_cube_1(low=low, high=high, sampling_rate=sampling_rate)
    elif case == 2:
        return test_cube_2(low=low, high=high, sampling_rate=sampling_rate)
    elif case == 3:
        return test_cube_3(low=low, high=high, sampling_rate=sampling_rate)
    elif case == 4:
        return test_cube_4(low=low, high=high, sampling_rate=sampling_rate)
    elif case == 5:
        return test_cube_5(low=low, high=high, sampling_rate=sampling_rate)
    elif case == 6:
        return test_cube_6(low=low, high=high, sampling_rate=sampling_rate)
    else:
        raise TypeError("The params `case` must be in the range of [1, 2, 3, 4, 5, 6]")


def test_multivariate_cube(
    case: Union[List[int], Tuple[int], np.ndarray] = None,
    low: Union[int, np.ndarray] = 0,
    high: Union[int, np.ndarray] = 10,
    sampling_rate: int = 30,
) -> np.ndarray:
    """
    Generate 3D cubes data for testing 3D multivariate signal decomposition algorithms

    :param case: The test cases to use are specified by inputting a list or tuple or ndarray in the range [1, 2, 3, 4, 5, 6],
            The default parameter settings are [1, 2, 3].
    :param low: Minimum value of grid matrix,
        If it is a ndarray array, its length must be 3 to indicate the range of three dimensions.
    :param high: Maximum value of grid matrix,
        If it is a ndarray array, its length must be 3 to indicate the range of three dimensions.
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix
    :return: Get 3D cube data of multiple different channels
    """
    # Default Parameters
    if case is None:
        case = [1, 2, 3]
    # Make sure it is the correct data type
    if (
        isinstance(case, tuple)
        or isinstance(case, list)
        or isinstance(case, np.ndarray)
    ):
        # Make sure there is content
        if len(case) >= 1:
            cubes = np.vstack(
                [
                    test_univariate_cube(
                        case=c, low=low, high=high, sampling_rate=sampling_rate
                    )
                    for c in case
                ]
            )
            return cubes
        else:
            raise ValueError("The number in `case` must be greater than 1!")
    else:
        raise TypeError("The params `case` should be a list or tuple or ndarray!")


def test_cube_1(
    low: Union[int, np.ndarray], high: Union[int, np.ndarray], sampling_rate: int = 30
) -> np.ndarray:
    """
    Generate a 3D cube test example 1

    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix

    :return: Cube generated by sine and cosine functions
    """
    # First generate the grid array
    x, y, z = get_meshgrid_3D(low=low, high=high, sampling_rate=sampling_rate)
    # Generate cubes of different frequencies
    uc1 = np.sin(3.5 * x) + np.sin(3.5 * y) + np.sin(3.5 * z)
    uc2 = np.sin(x) + np.sin(y) + np.sin(z)
    ru = np.cos(x / 24) + np.cos(y / 24) + np.cos(z / 24)
    # Fusion of different modalities
    u = uc1 + uc2 + ru
    # Standardize the results of the integration
    u = (u - np.mean(u)) / np.std(u + 0.01)
    return u


def test_cube_2(
    low: Union[int, np.ndarray], high: Union[int, np.ndarray], sampling_rate: int = 30
) -> np.ndarray:
    """
    Generate a 3D cube test example 2

    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix

    :return: Cube generated by sine and cosine functions
    """
    # First generate the grid array
    x, y, z = get_meshgrid_3D(low=low, high=high, sampling_rate=sampling_rate)
    # Generate cubes of different frequencies
    uc1 = np.sin(5 * x) + np.sin(5 * y) + np.sin(5 * z)
    uc2 = np.sin(x) + np.sin(y) + np.sin(z)
    ru = np.cos(x / 16) + np.cos(y / 16) + np.cos(z / 16)
    # Fusion of different modalities
    u = uc1 + uc2 + ru
    # Standardize the results of the integration
    u = (u - np.mean(u)) / np.std(u + 0.001)
    return u


def test_cube_3(
    low: Union[int, np.ndarray], high: Union[int, np.ndarray], sampling_rate: int = 30
) -> np.ndarray:
    """
    Generate a 3D cube test example 3

    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix

    :return: Cube generated by sine and cosine functions
    """
    # First generate the grid array
    x, y, z = get_meshgrid_3D(low=low, high=high, sampling_rate=sampling_rate)
    # Generate cubes of different frequencies
    uc1 = np.sin(2 * x) + np.sin(2 * y) + np.sin(2 * z)
    uc2 = np.sin(4 * x) + np.sin(4 * y) + np.sin(4 * z)
    ru = np.cos(x / 24) + np.cos(y / 24) + np.cos(z / 24)
    # Fusion of different modalities
    u = uc1 + uc2 + ru
    # Standardize the results of the integration
    u = (u - np.mean(u)) / np.std(u + 0.001)
    return u


def test_cube_4(
    low: Union[int, np.ndarray], high: Union[int, np.ndarray], sampling_rate: int = 30
) -> np.ndarray:
    """
    Generate a 3D cube test example 4

    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix

    :return: Cube generated by sine and cosine functions
    """
    # First generate the grid array
    x, y, z = get_meshgrid_3D(low=low, high=high, sampling_rate=sampling_rate)
    # Generate cubes of different frequencies
    uc1 = np.sin(2 * x**1.5) + np.sin(2 * y**1.5) + np.sin(2 * z * 1.5)
    uc2 = np.sin(5 * x) + np.sin(5 * y) + np.sin(5 * z)
    ru = np.cos(x / 24) + np.cos(y / 24) + np.cos(z / 24)
    # Fusion of different modalities
    u = uc1 + uc2 + ru
    # Standardize the results of the integration
    u = (u - np.mean(u)) / np.std(u + 0.001)
    return u


def test_cube_5(
    low: Union[int, np.ndarray], high: Union[int, np.ndarray], sampling_rate: int = 30
) -> np.ndarray:
    """
    Generate a 3D cube test example 5

    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix

    :return: Cube generated by sine and cosine functions
    """
    # First generate the grid array
    x, y, z = get_meshgrid_3D(low=low, high=high, sampling_rate=sampling_rate)
    # Generate cubes of different frequencies
    uc1 = np.cos(x**2) + np.cos(y**2) + np.cos(z * 2)
    uc2 = np.sin(5 * x) + np.sin(5 * y) + np.sin(5 * z)
    ru = np.cos(x / 24) + np.cos(y / 24) + np.cos(z / 24)
    # Fusion of different modalities
    u = uc1 + uc2 + ru
    # Standardize the results of the integration
    u = (u - np.mean(u)) / np.std(u + 0.001)
    return u


def test_cube_6(
    low: Union[int, np.ndarray], high: Union[int, np.ndarray], sampling_rate: int = 30
) -> np.ndarray:
    """
    Generate a 3D cube test example 6

    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix

    :return: Cube generated by sine and cosine functions
    """
    # First generate the grid array
    x, y, z = get_meshgrid_3D(low=low, high=high, sampling_rate=sampling_rate)
    # Generate cubes of different frequencies
    uc1 = np.cos(x**2) + np.cos(y**2) + np.cos(z * 2)
    uc2 = np.sin(5 * x) + np.sin(5 * y) + np.sin(5 * z)
    ru = np.cos(x / 24) + np.cos(y / 24) + np.cos(z / 24)
    # Fusion of different modalities
    u = uc1 + uc2 + ru
    # Standardize the results of the integration
    u = (u - np.mean(u)) / np.std(u + 0.001)
    return u


if __name__ == "__main__":
    print(test_univariate_cube().shape)
    print(test_univariate_cube())
