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

from typing import Optional, Tuple


def test_grayscale() -> np.ndarray | None:
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


def get_meshgrid(low: int = 0, high: int = 10, sampling_rate: Optional[int] = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    给定输入输出范围来生成网格矩阵
    :param low: 网格矩阵的最小值
    :param high: 网格矩阵的最大值
    :param sampling_rate: 网格矩阵的采样率，即为矩阵中的点数
    :return: 以元组的形式范围生成的网格矩阵
    """
    # 首先生成一维数组
    X = np.linspace(low, high, sampling_rate)
    Y = np.linspace(low, high, sampling_rate)

    # 随后构建网格矩阵
    x, y = np.meshgrid(X, Y)

    return x, y


def test_uni_image():
    pass


