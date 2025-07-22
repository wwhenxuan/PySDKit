# -*- coding: utf-8 -*-
"""
Created on 2025/07/21 11:53:35
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Tuple
from pysdkit.models._base import normalize, centralize, UnsupervisedModel


def pca(
    X: np.ndarray, n_components: Optional[int] = 2, norm: Optional[str] = "centralize"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Functional interface for Principal Component Analysis (PCA) dimensionality reduction.

    :param X: Input data matrix with shape (n_samples, n_features).
    :param n_components: Number of principal components to retain.
    :param norm: Standardization method, one of ['centralize', 'normalize'].
    :return: - X_reduced : Dimension-reduced data matrix with shape (n_samples, n_components)
             - components : Principal components (eigenvectors) with shape (n_components, n_features)
             - explained_variance_ratio : Variance explained by each principal component
    """
    # 1. Standardize the data
    if norm == "centralize":
        X_norm = centralize(X)

    elif norm == "normalize":
        X_norm = normalize(X)

    else:
        X_norm = X

    # 2. Compute the covariance matrix
    cov_matrix = np.cov(X_norm, rowvar=False)

    # 3. Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. Sort eigenvalues in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_idx]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]

    # 5. Select the top n components
    components = sorted_eigenvectors[:, :n_components].T
    explained_variance = sorted_eigenvalues[:n_components]

    # 6. Compute explained variance ratio
    total_variance = np.sum(sorted_eigenvalues)
    explained_variance_ratio = explained_variance / total_variance

    # 7. Project data onto principal components
    X_reduced = np.dot(X_norm, components.T)

    return X_reduced, components, explained_variance_ratio


class PCA(UnsupervisedModel):
    """Object-oriented interface for Principal Component Analysis (PCA)"""

    def __init__(
        self, n_components: Optional[int] = 2, norm: Optional[str] = "centralize"
    ) -> None:
        """
        Performs dimensionality reduction on an array of shape [n_samples, n_features].

        :param n_components: Number of principal components to retain.
        :param norm: Standardization method, one of ['centralize', 'normalize'].
        """
        self.n_components = n_components
        self.norm = norm

        # Stores the results of the current model fit
        self.X_reduced = None
        self.components = None
        self.explained_variance_ratio = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Allow instances to be called like functions"""
        return self.fit_transform(X=X)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Principal Component Analysis"

    @staticmethod
    def _check_inputs(X: np.ndarray) -> None:
        """
        Check if input data is valid

        :param X: the ndarray with shape (n_samples, n_features).
        :return: None
        """
        if isinstance(X, np.ndarray):
            raise TypeError("Input data must be an ndarray.")
        if len(X.shape) != 2:
            raise ValueError(
                "Input data must be an ndarray with shape (n_samples, n_features)."
            )

    def reset(self) -> None:
        """Clear all stored results from the PCA algorithm."""
        self.X_reduced = None
        self.components = None
        self.explained_variance_ratio = None

    def data_processing(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocesses the data.

        :param X: Data to be preprocessed.
        :return: Preprocessed data.
        """
        if self.norm == "normalize":
            return self.normalize(X)
        elif self.norm == "centralize":
            return self.centralize(X)
        else:
            return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Executes the PCA algorithm.

        :param X: Input data matrix with shape (n_samples, n_features).
        :return: - X_reduced : Dimension-reduced data matrix with shape (n_samples, n_components)
                 - components : Principal components (eigenvectors) with shape (n_components, n_features)
                 - explained_variance_ratio : Variance explained by each principal component
        """
        # Clear previously stored results
        self.reset()

        # Check if the inputs data is valid
        self._check_inputs(X=X)

        # Preprocess the data
        X_norm = self.data_processing(X)

        # Compute the covariance matrix
        cov_matrix = np.cov(X_norm, rowvar=False)

        # Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvalues in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_idx]
        sorted_eigenvectors = eigenvectors[:, sorted_idx]

        # Select the top n components
        self.components = sorted_eigenvectors[:, : self.n_components].T
        explained_variance = sorted_eigenvalues[: self.n_components]

        # Compute explained variance ratio
        total_variance = np.sum(sorted_eigenvalues)
        self.explained_variance_ratio = explained_variance / total_variance

        # Project data onto principal components
        self.X_reduced = np.dot(X_norm, self.components.T)

        return self.X_reduced

    def get_components(self) -> np.ndarray:
        """Returns the top N principal components."""
        if self.components is not None:
            # Return recorded results after PCA has been executed
            return self.components
        else:
            raise ValueError("Please run the PCA algorithm first!")

    def get_explained_variance_ratio(self) -> float:
        """Calculates the explained variance ratio."""
        if self.explained_variance_ratio is not None:
            # Return the explained variance ratio
            return self.explained_variance_ratio
        else:
            raise ValueError("Please run the PCA algorithm first!")


# if __name__ == "__main__":
#     # 创建示例数据（4个样本，3个特征）
#     X = np.array([[2.5, 2.4, 3.1], [0.5, 0.7, 1.2], [2.2, 2.9, 2.5], [1.9, 2.2, 1.8]])
#
#     # 执行PCA降维到2维
#     X_reduced, components, variance_ratio = pca(X, n_components=2)
#
#     print("原始数据形状:", X.shape)
#     print("降维后数据形状:", X_reduced.shape)
#     print("\n主成分（特征向量）:")
#     print(components)
#     print("\n方差解释比例:", variance_ratio)
#     print("\n降维后的数据:")
#     print(X_reduced)
#
#     pca_model = PCA(n_components=2)
#     X_reduced = pca_model.fit_transform(X)
#
#     print(X_reduced)
