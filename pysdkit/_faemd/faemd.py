# -*- coding: utf-8 -*-
"""
Created on 2025/02/01 22:30:40
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np


class FAEMD(object):
    """
    Fast and Adaptive Empirical Mode Decomposition

    Thirumalaisamy, Mruthun R., and Phillip J. Ansell.
    “Fast and Adaptive Empirical Mode Decomposition for Multidimensional, Multivariate Signals.”
    IEEE Signal Processing Letters, vol. 25, no. 10, Institute of Electrical and Electronics Engineers (IEEE),
    Oct. 2018, pp. 1550–54, doi:10.1109/lsp.2018.2867335.

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd
    """

    def __init__(self, max_imfs: int, tol: float = None, window_type: int = 1, ):  # window_type参数暂时待定

        self.max_imfs = max_imfs
        if max_imfs < 1:
            raise ValueError("`max_imfs` must be a positive integer")
        self.tol = tol
        self.window_type = window_type
        # 这里需要对这个参数进行进一步的检验


        pass

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Fast and Adaptive Empirical Mode Decomposition (FAEMD)"

    def _get_tol(self, signal: np.ndarray) -> float:
        """Get the tolerance parameter for FAEMD"""
        # 当`tol`变量默认为None时
        if self.tol is None:
            tol = np.min(np.sqrt(np.mean(signal**2)), np.array([1])) * 0.001
            return tol
        # 返回用户指定的参数
        return self.tol

    def fit_transform(self, signal: np.ndarray):

        # 获取信号的变量数目和长度
        signal = signal.T
        Nx, n = signal.shape

        # Initialisations
        imfs = np.zeros(shape=(Nx, n, self.max_imfs))


        # 获取算法停止的容忍度参数
        tol = self._get_tol(signal)


