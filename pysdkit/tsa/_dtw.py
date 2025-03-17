# -*- coding: utf-8 -*-
"""
Created on 2025/02/12 00:11:39
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
from sys import maxsize as MAXSIZE
import numpy as np

from typing import Optional, Callable


def dtw_distance(
    ts_a: np.ndarray,
    ts_b: np.ndarray,
    d: Callable = lambda x, y: abs(x - y),
    max_warping_window: Optional[int] = 10000,
) -> np.ndarray:
    """
    Returns the DTW similarity distance between two 2-D timeseries numpy arrays.

    :param ts_a: array of shape [n_samples, n_timepoints]
    :param ts_b: array of shape [n_samples, n_timepoints]
                 Two arrays containing n_samples of timeseries data
                 whose DTW distance between each sample of A and B
                 will be compared
    :param d: DistanceMetric object (default = abs(x-y)) the distance measure used for A_i - B_j in the
              DTW dynamic programming function
    :param max_warping_window: int, optional (default = infinity)
                               Maximum warping window allowed by the DTW dynamic programming function
    :return: DTW distance between A and B
    """

    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = MAXSIZE * np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - max_warping_window), min(N, i + max_warping_window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = np.min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]
