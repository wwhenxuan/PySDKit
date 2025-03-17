# -*- coding: utf-8 -*-
"""
Created on 2025/03/17 17:53:21
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import sys
import numpy as np
from scipy.stats import mode
from scipy.spatial.distance import squareform

from typing import Optional, Callable, Tuple

from pysdkit.tsa import dtw_distance


class KnnDtw(object):
    """
    K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays
    """

    def __init__(
        self,
        n_neighbors: Optional[int] = 5,
        max_warping_window: int = 10000,
        subsample_step: int = 1,
    ) -> None:
        """
        :param n_neighbors: int, optional (default = 5), Number of neighbors to use by default for KNN
        :param max_warping_window: int, optional (default = infinity)
                                   Maximum warping window allowed by the DTW dynamic programming function
        :param subsample_step: int, optional (default = 1)
                               Step size for the timeseries array. By setting subsample_step = 2,
                               item is skipped. Implemented by x[:, ::subsample_step]
        """
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step

        # Define the sample and label variables of the fitting data in advance
        self.samples, self.labels = None, None

    def fit(self, samples: np.ndarray, labels: np.ndarray) -> None:
        """
        Fit the model using x as training data and l as class labels

        :param samples: array of shape [n_samples, n_timepoints],
                  Training data set for input into KNN classifer
        :param labels: array of shape [n_samples],
                  Training labels for input into KNN classifier
        :return: None
        """
        self.samples = samples
        self.labels = labels

    def _dtw_distance(
        self, ts_a: np.ndarray, ts_b: np.ndarray, d: Callable = lambda x, y: abs(x - y)
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
        :return: DTW distance between A and B
        """
        return dtw_distance(
            ts_a=ts_a, ts_b=ts_b, d=d, max_warping_window=self.max_warping_window
        )

    def _dist_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure

        :param x: array of shape [n_samples, n_timepoints]
        :param y: array of shape [n_samples, n_timepoints]
        :return: Distance matrix between each item of x and y with
                 shape [training_n_samples, testing_n_samples]
        """

        # Compute the distance matrix
        dm_count = 0

        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if np.array_equal(x, y):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)

            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(
                        x[i, :: self.subsample_step], y[j, :: self.subsample_step]
                    )

                    dm_count += 1
            # Convert to squareform
            dm = squareform(dm)
            return dm

        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0]))

            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j] = self._dtw_distance(
                        x[i, :: self.subsample_step], y[j, :: self.subsample_step]
                    )
                    # Update progress bar
                    dm_count += 1

            return dm

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the class labels or probability estimates for the provided data

        :param x: array of shape [n_samples, n_timepoints]
                  Array containing the testing data set to be classified
        :return: (1) the predicted class labels
                 (2) the knn label count probability
        """
        # If no training data is input, prediction is made
        if self.samples is None:
            # No training data has been entered yet
            raise ValueError("Must fit the training data before predicting")

        # Calculate the distance matrix between the input data and the training set samples
        dm = self._dist_matrix(x, self.samples)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, : self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.labels[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1] / self.n_neighbors

        # Returns the classification results and the probability of classification
        return mode_label.ravel(), mode_proba.ravel()
