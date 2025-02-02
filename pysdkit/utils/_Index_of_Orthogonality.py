# -*- coding: utf-8 -*-
"""
Created on 2024/6/2 19:52
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Index of Orthogonality is proposed py Under Barbosa de Souza in
A survey on Hilbert-Huang transform: Evolution, challenges and solutions.
Digital Signal Processing 120 (2022) 103292.
"""
import numpy as np


def index_of_orthogonality(signal: np.ndarray, IMFs: np.ndarray) -> float:
    """
    Any pair of IMFs is locally orthogonal.
    To evaluate EMD performance, an Index of Orthogonality (IO) was proposed,
    so that the closer to zero, the more effective will be the decomposition.
    :param signal: the row input signal.
    :param IMFs: Intrinsic Mode Function after decomposition.
    :return: the value of index of orthogonality
    """
    def cum(x, y):
        # Helper functions
        return np.sum(np.abs(x * y))

    # the value of IO
    index = 0.0

    # Start calculating the evaluation metrics
    K, T = IMFs.shape
    for i in range(K):
        for j in range(K):
            if i != j and i < j:
                index += cum(IMFs[i, :], IMFs[j, :])

    return index / np.sum(signal ** 2)



