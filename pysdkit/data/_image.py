# -*- coding: utf-8 -*-
"""
Created on 2025/02/02 13:01:46
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
Generate samples for testing 2D image data
"""
import numpy as np
from os import path
import requests

from typing import Optional, Union, Tuple


def test_grayscale() -> Union[np.ndarray, None]:
    """
    Load a sample 2D grayscale image for testing, the size is [256, 256]

    We download the data from https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/pysdkit/data/texture.txt

    This data comes from https://www.mathworks.com/matlabcentral/fileexchange/45918-two-dimensional-variational-mode-decomposition
    Konstantin, Dragomiretskiy, and Dominique Zosso. "Two-dimensional variational mode decomposition."
    Energy Minimization Methods in Computer Vision and Pattern Recognition. Vol. 8932. 2015.
    """
    # Try accessing data locally first
    current_directory = path.dirname(path.abspath(__file__))
    data_path = path.join(current_directory, "texture.txt")

    if path.exists(data_path):
        # If the data exists, read the data directly
        return np.loadtxt(data_path)

    # URL address for downloading data
    data_url = "https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/pysdkit/data/texture.txt"

    # Downloading files using requests
    try:
        response = requests.get(data_url)
        response.raise_for_status()
        with open(data_path, "wb") as file:
            file.write(response.content)
        print("downloaded successfully!")
        return np.loadtxt(data_path)
    except requests.exceptions.RequestException as error:
        print(error)
        print("something went wrong in downloading data!")

    return None


def get_meshgrid_2D(
    low: float = 0, high: float = 10, sampling_rate: Optional[int] = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a grid matrix given an input and output range

    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix
    :return: A grid matrix generated from a range in the form of a tuple
    """
    # First generate a one-dimensional array
    X = np.linspace(low, high, sampling_rate)
    Y = np.linspace(low, high, sampling_rate)

    # Then construct the grid matrix
    x, y = np.meshgrid(X, Y)

    return x, y


def test_univariate_image(
    case: int = 1, low: int = 0, high: int = 10, sampling_rate: Optional[int] = 256
) -> np.ndarray:
    """
    Generate a single image for 2D decomposition

    please input the case from 1 to 7

    :param case: case number in 1~7
    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix

    :return: the generated image for 2D decomposition
    """
    if case == 1:
        return test_image_1(low, high, sampling_rate)
    elif case == 2:
        return test_image_2(low, high, sampling_rate)
    elif case == 3:
        return test_image_3(low, high, sampling_rate)
    elif case == 4:
        return test_image_4(low, high, sampling_rate)
    elif case == 5:
        return test_image_5(low, high, sampling_rate)
    elif case == 6:
        return test_image_6(low, high, sampling_rate)
    elif case == 7:
        return test_uni_image_7(low, high, sampling_rate)
    else:
        print(
            f"There is no case {case}, so it will return the generated function `test_grayscale`."
        )
        return test_grayscale()


def test_multivariate_image(
    case: Tuple = (1, 2, 3),
    low: int = 0,
    high: int = 10,
    sampling_rate: Optional[int] = 256,
) -> np.ndarray:
    """
    Generate multivariate image for 2D decomposition, the number of image is `len(case)`

    please input the case from 1 to 7

    :param case: case number in 1~7
    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix

    :return: the generated image for 2D decomposition
    """
    # The number of generated images
    number = len(case)

    # Preventing too many samples from being generated
    assert number < 8, "the number of cases should be less than 8, better for 3"

    # Make the input legal
    c_list = []
    for c in case:
        if c in list(range(1, 8)):
            c_list.append(c)

    # Illegal Image Substitute
    sub = number - len(c_list)
    if sub > 0:
        c = 1
        while sub == 0:
            if c not in c_list:
                c_list.append(c)
                sub -= 1
            else:
                c += 1

    # Get multivariate image by sampling univariate function
    images = [
        test_univariate_image(case=c, low=low, high=high, sampling_rate=sampling_rate)[
            np.newaxis, :, :
        ]
        for c in c_list
    ]

    # Merging data types
    images = np.vstack(images)

    return images


def test_image_1(
    low: int = 0, high: int = 10, sampling_rate: Optional[int] = 256
) -> np.ndarray:
    """
    Generate a single image test sample 1
    the symbol function comes from:
    https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd
    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix
    :return: Generate a 2D image using sin(4x), sin(x) and cos(x/24)
    """
    # Construct the two dim grid matrix
    x, y = get_meshgrid_2D(low, high, sampling_rate)

    # Generate a function according to the specified method
    uc1 = np.sin(4 * x) + np.sin(4 * y)
    uc2 = np.sin(x) + np.sin(y)
    ru = np.cos(x / 24) + np.cos(y / 24)

    # add it together
    image = uc1 + uc2 + ru

    # normalize
    image = (image - image.mean()) / image.std()

    return image


def test_image_2(
    low: int = 0, high: int = 10, sampling_rate: Optional[int] = 256
) -> np.ndarray:
    """
    Generate a single image test sample 2
    the symbol function comes from:
    :ref: https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd

    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix
    :return: Generate a mixed image using two-dimensional sine and cosine functions
    """
    # Construct the two dim grid matrix
    x, y = get_meshgrid_2D(low, high, sampling_rate)

    # Generate a function according to the specified method
    uc1 = np.sin(x) + np.sin(y)
    uc2 = np.sin(3.5 * x) + np.sin(3.5 * y)
    ru = np.cos(x / 16) + np.cos(y / 16)

    # add it together
    image = uc1 + uc2 + ru

    # normalize
    image = (image - image.mean()) / image.std()

    return image


def test_image_3(
    low: int = 0, high: int = 10, sampling_rate: Optional[int] = 256
) -> np.ndarray:
    """
    Generate a single image test sample 3
    the symbol function comes from:
    https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd

    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix
    :return: Generate a mixed image using two-dimensional sine and cosine functions
    """
    # Construct the two dim grid matrix
    x, y = get_meshgrid_2D(low, high, sampling_rate)

    # Generate a function according to the specified method
    uc1 = np.sin(x / 24) + np.sin(y / 24)
    uc2 = np.sin(x / 5) + np.sin(y / 5)
    ru = np.sin(x) + np.sin(y)

    # add it together
    image = uc1 + uc2 + ru

    # normalize
    image = (image - image.mean()) / image.std()

    return image


def test_image_4(
    low: int = 0, high: int = 10, sampling_rate: Optional[int] = 256
) -> np.ndarray:
    """
    Generate a single image test sample 4
    the symbol function comes from:
    https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd
    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix
    :return: Generate a mixed image using two-dimensional sine and cosine functions
    """
    # Construct the two dim grid matrix
    x, y = get_meshgrid_2D(low, high, sampling_rate)

    # Generate a function according to the specified method
    uc1 = np.sin(x / 60) + np.sin(y / 60)
    uc2 = np.sin(x / 16) + np.sin(y / 16)
    ru = np.sin(x / 32) + np.sin(y / 32)

    # add it together
    image = uc1 + uc2 + ru

    # normalize
    image = (image - image.mean()) / image.std()

    return image


def test_image_5(
    low: int = 0, high: int = 10, sampling_rate: Optional[int] = 256
) -> np.ndarray:
    """
    Generate a single image test sample 5
    the symbol function comes from:
    https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd
    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix
    :return: Generate a mixed image using two-dimensional sine and cosine functions
    """
    # Construct the two dim grid matrix
    x, y = get_meshgrid_2D(low, high, sampling_rate)

    # Generate a function according to the specified method
    uc1 = np.sin(x * 8) + np.sin(y * 8)
    uc2 = np.sin(x * 16) + np.sin(y * 16)
    ru = np.sin(x / 32) + np.sin(y / 32)

    # add it together
    image = uc1 + uc2 + ru

    # normalize
    image = (image - image.mean()) / image.std()

    return image


def test_image_6(
    low: int = 0, high: int = 10, sampling_rate: Optional[int] = 256
) -> np.ndarray:
    """
    Generate a single image test sample 6
    the symbol function comes from:
    https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd
    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix
    :return: Generate a mixed image using two-dimensional sine and cosine functions
    """
    # Construct the two dim grid matrix
    x, y = get_meshgrid_2D(low, high, sampling_rate)

    # Generate a function according to the specified method
    uc1 = np.cos(x**1.2 * 8) + np.sin(y**1.2 * 8)
    uc2 = np.sin(x**1.3 * 4) + np.cos(y**1.3 * 4)
    ru = np.sin(x / 12) + np.sin(y / 12)

    # add it together
    image = uc1 + uc2 + ru

    # normalize
    image = (image - image.mean()) / image.std()

    return image


def test_uni_image_7(
    low: int = 0, high: int = 10, sampling_rate: Optional[int] = 256
) -> np.ndarray:
    """
    Generate a single image test sample 7
    the symbol function comes from:
    https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd
    :param low: Minimum value of grid matrix
    :param high: Maximum value of grid matrix
    :param sampling_rate: The sampling rate of the grid matrix, which is the number of points in the matrix
    :return: Generate a mixed image using two-dimensional sine and cosine functions
    """
    # Construct the two dim grid matrix
    x, y = get_meshgrid_2D(low, high, sampling_rate)

    # Generate a function according to the specified method
    uc1 = np.cos(x**1.2 * 8) + np.sin(y**1.2 * 8)
    uc2 = np.sin(x**1.3 * 4) + np.cos(y**1.3 * 4)
    ru = np.sin(x**3) + np.sin(y**3)

    # add it together
    image = uc1 + uc2 + ru

    # normalize
    image = (image - image.mean()) / image.std()

    return image


if __name__ == "__main__":
    from pysdkit.plot import plot_grayscale_image
    from matplotlib import pyplot as plt

    plot_grayscale_image(test_uni_image_7())
    plt.show()

    print(test_multivariate_image().shape)
