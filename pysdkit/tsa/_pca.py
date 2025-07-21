# -*- coding: utf-8 -*-
"""
Created on 2025/07/21 11:53:35
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Tuple


def centralize(X: np.ndarray) -> np.ndarray:
    """
    centralize the input ndarray data.

    :param X: the input ndarray with shape (num_samples, num_features)
    :return: the centralized ndarray with shape (num_samples, num_features)
    """
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    return X_centered


def normalize(X: np.ndarray) -> np.ndarray:
    """
    normalizing the input ndarray data.

    :param X: the input ndarray with shape (num_samples, num_features)
    :return: the normalized ndarray with shape (num_samples, num_features)
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std

    return X_normalized


def pca(X: np.ndarray, n_components: Optional[int] = 2, norm: Optional[str] = "centralize") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA主成分分析降维的函数接口

    :param X: 输入数据矩阵，形状为 (n_samples, n_features)
    :param n_components: 要保留的主成分数量
    :param norm: 使用的标准化方法，可选['centralize', 'normalize']
    :return: - X_reduced : 降维后的数据矩阵，形状为 (n_samples, n_components)
             - components : 主成分（特征向量），形状为 (n_components, n_features)
             - explained_variance_ratio : 各主成分的方差解释比例
    """
    # 1. 数据标准化
    if norm == "centralize":
        # 对输入数据进行
        X_norm = centralize(X)

    elif norm == "normalize":
        # 对输入就进行标准化
        X_norm = normalize(X)

    else:
        X_norm = X

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X_norm, rowvar=False)

    # 3. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. 按特征值降序排序
    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_idx]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]

    # 5. 选择前n个主成分
    components = sorted_eigenvectors[:, :n_components].T
    explained_variance = sorted_eigenvalues[:n_components]

    # 6. 计算方差解释比例
    total_variance = np.sum(sorted_eigenvalues)
    explained_variance_ratio = explained_variance / total_variance

    # 7. 投影到主成分空间
    X_reduced = np.dot(X_norm, components.T)

    return X_reduced, components, explained_variance_ratio

# 示例用法
if __name__ == "__main__":
    # 创建示例数据（4个样本，3个特征）
    X = np.array([
        [2.5, 2.4, 3.1],
        [0.5, 0.7, 1.2],
        [2.2, 2.9, 2.5],
        [1.9, 2.2, 1.8]
    ])

    # 执行PCA降维到2维
    X_reduced, components, variance_ratio = pca(X, n_components=2)

    print("原始数据形状:", X.shape)
    print("降维后数据形状:", X_reduced.shape)
    print("\n主成分（特征向量）:")
    print(components)
    print("\n方差解释比例:", variance_ratio)
    print("\n降维后的数据:")
    print(X_reduced)