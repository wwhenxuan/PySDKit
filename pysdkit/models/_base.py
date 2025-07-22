# -*- coding: utf-8 -*-
"""
Created on 2025/07/21 15:34:55
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from abc import ABC, abstractmethod


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


class SupervisedModel(ABC):
    """Serves as the base class for all algorithm supervised models"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        根据输入的样本和标签数组去拟合有监督模型的参数

        :param X: 形状为[n_samples, num_features]的训练数据样本
        :param y: 形状为[n_samples]的训练数据标签
        :return: None
        """
        pass

    @abstractmethod
    def predict(self, X_pred: np.ndarray):
        """
        根据`fit`方法中拟合获得的参数去执行对应的算法做出模型的预测

        :param X_pred: 形状为[n_samples, num_features]的待预测样本
        :return: The predicted values of `X`
        """
        pass

    @staticmethod
    def centralize(X: np.ndarray) -> np.ndarray:
        """
        centralize the input ndarray data.

        :param X: the input ndarray with shape (num_samples, num_features)
        :return: the centralized ndarray with shape (num_samples, num_features)
        """
        return centralize(X)

    @staticmethod
    def normalize(X: np.ndarray) -> np.ndarray:
        """
        normalizing the input ndarray data.

        :param X: the input ndarray with shape (num_samples, num_features)
        :return: the normalized ndarray with shape (num_samples, num_features)
        """
        return normalize(X)


class UnsupervisedModel(ABC):
    """Serves as the base class for all algorithm unsupervised models"""

    @abstractmethod
    def fit_transform(self, X: np.ndarray):
        """
        执行无监督训练算法的通用接口

        :param X: 待进行无监督学习的数据，形状为[n_samples, num_features]
        :return: the transformed ndarray with shape (num_samples, num_features)
        """
        pass

    @staticmethod
    def centralize(X: np.ndarray) -> np.ndarray:
        """
        centralize the input ndarray data.

        :param X: the input ndarray with shape (num_samples, num_features)
        :return: the centralized ndarray with shape (num_samples, num_features)
        """
        return centralize(X)

    @staticmethod
    def normalize(X: np.ndarray) -> np.ndarray:
        """
        normalizing the input ndarray data.

        :param X: the input ndarray with shape (num_samples, num_features)
        :return: the normalized ndarray with shape (num_samples, num_features)
        """
        return normalize(X)
