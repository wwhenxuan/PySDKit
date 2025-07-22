# -*- coding: utf-8 -*-
"""
Created on 2025/07/22 17:46:28
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Tuple


def test_pca(number: int = 200, dim: int = 2, random_state: Optional[int] = 42) -> np.ndarray:
    """
    Generate synthetic data for testing Principal Component Analysis (PCA).
    The dataset is created by multiplying a set of coordinate points (default 2-D)
    by a random rotation matrix.

    :param number: int, optional
        Number of data points to generate, by default 200.
    :param dim: int, optional
        Dimensionality of the generated data, by default 2.
    :param random_state: int, optional
        Random seed for reproducibility, by default 42.
    :return: X : np.ndarray
        Generated test data of shape [n_samples, n_features].
    """
    # Initialize random number generator
    rng = np.random.RandomState(random_state)

    # Generate a random rotation matrix
    rot = rng.rand(dim, dim)

    return np.dot(rot, rng.randn(dim, number)).T