# -*- coding: utf-8 -*-
"""
Created on 2025/02/12 00:14:21
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np


class STL(object):
    """
    Seasonal-Trend decomposition using LOESS (STL)

    STL uses LOESS (locally estimated scatterplot smoothing) to extract smooths estimates of the three components.
    The key inputs into STL are:
    (1) season - The length of the seasonal smoother. Must be odd.
    (2) trend - The length of the trend smoother, usually around 150% of season. Must be odd and larger than season.
    (3) low_pass - The length of the low-pass estimation window, usually the smallest odd number larger than the periodicity of the data.
    """

    def __init__(self):
        pass

    def __call__(self, data):
        pass

    def __str__(self) -> str:
        return "Seasonal-Trend decomposition using LOESS (STL)"

    def fit_transform(self):
        pass
