# -*- coding: utf-8 -*-
"""
Created on 2025/02/02 13:01:46
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
生成用于测试二维图像数据的样例
"""
import numpy as np
from os import path
import requests


def test_grayscale() -> np.ndarray | None:
    """
    Load a sample 2D grayscale image for testing
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
