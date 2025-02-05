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
    加载用于测试的二维灰度图像样例
    We download the data from https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/pysdkit/data/texture.txt
    This data comes from https://www.mathworks.com/matlabcentral/fileexchange/45918-two-dimensional-variational-mode-decomposition
    Konstantin, Dragomiretskiy, and Dominique Zosso. "Two-dimensional variational mode decomposition."
    Energy Minimization Methods in Computer Vision and Pattern Recognition. Vol. 8932. 2015.
    """
    # 先尝试从本地访问数据
    current_directory = path.dirname(path.abspath(__file__))
    data_path = path.join(current_directory, "texture.txt")

    if path.exists(data_path):
        # 如果数据存在则直接读取数据
        return np.loadtxt(data_path)

    # 下载数据的URL地址
    data_url = "https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/pysdkit/data/texture.txt"

    # 使用requests下载文件
    try:
        response = requests.get(data_url)
        response.raise_for_status()  # 查看请求是否成功
        with open(data_path, "wb") as file:
            file.write(response.content)
        print("downloaded successfully!")
        return np.loadtxt(data_path)
    except requests.exceptions.RequestException as error:
        print(error)
        print("something went wrong in downloading data!")

    return None
