# -*- coding: utf-8 -*-
"""
Created on 2025/02/05 20:55:47
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Tuple


class AVNCMD(object):
    """
    Adaptive Variational Nonlinear Chirp Mode Decomposition.

    [1] Liang, Hao and Ding, Xinghao and Jakobsson, Andreas and Tu, Xiaotong and Huang, Yue.
    "Adaptive Variational Nonlinear Chirp Mode Decomposition",
    in 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.
    [2] Chen S, Dong X, Peng Z, et al. Nonlinear chirp mode decomposition: A variational method.
    IEEE Transactions on Signal Processing, 2017.
    """

    def __init__(self,
                 max_imfs: Optional[int] = 3,
                 beta: Optional[float] = 1e-6,
                 tol: Optional[float] = 1e-5,
                 max_iter: Optional[int] = 300,
                 random_state: Optional[int] = 42,
                 dtype: np.dtype = np.float64, ):
        """
        :param max_imfs:
        :param beta: filter parameter
        :param tol: tolerance of convergence criterion
        :param max_iter: the maximum allowable iterations
        :param random_state:
        :param dtype:
        """
        self.beta = beta
        self.tol = tol

        self.DTYPE = dtype

        self.max_imfs = max_imfs

        self.max_iter = max_iter

        # 记录随机种子并创建随机数生成器
        self.random_state = random_state
        self.rng = np.random.RandomState(seed=random_state)

    def _check_IF(self, signal: np.ndarray, iniIF: Optional[np.ndarray] = None):
        """初始化本征函数"""
        length = len(signal)
        if iniIF is None:
            # 如果用户未传入初始化的本征函数
            iniIF = np.ones(shape=(self.max_imfs, length))
            for row in range(length):
                # 遍历其中的每一行信息表示一个本征模态函数
                iniIF[row, :] = self.rng.randint(10, 500)
        elif iniIF is not None and isinstance(iniIF, np.ndarray):
            # 如果用户传入了初始化的本征函数并且不为空
            num_imfs, len_iniIF = iniIF.shape
            if num_imfs != self.max_imfs:
                # 修正要获得的本征模态函数的数目
                self.max_imfs = num_imfs
            if len_iniIF != length:
                # 如果用户传入的初始化本征函数与输入信号的长度不相同
                if len_iniIF > length:
                    iniIF = iniIF[:, :length]
                else:
                    iniIF = np.ones(shape=(num_imfs, length)) * np.array([[value] for value in iniIF[:, 0]])
        else:
            raise ValueError("Wrong Inputs for iniIF!")

        return self.max_imfs, iniIF

    def differ(self, y: np.ndarray, delta: float) -> np.ndarray:
        """
        Compute the derivative of a discrete time series y.

        :param y: The input signal.
        :param delta: The sampling time interval of y.
        :return: numpy.ndarray: The derivative of the time series.
        """
        L = len(y)
        ybar = np.zeros(L - 2, dtype=self.DTYPE)

        for i in range(1, L - 1):
            ybar[i - 1] = (y[i + 1] - y[i - 1]) / (2 * delta)

        # Prepend and append the boundary differences
        ybar = np.concatenate(
            (
                np.array([(y[1] - y[0]) / delta], dtype=self.DTYPE),
                ybar,
                np.array([(y[-1] - y[-2]) / delta], dtype=self.DTYPE),
            )
        )

        return ybar

    def fit_transform(self, signal: np.ndarray, fs: Optional[int] = None, iniIF: Optional[np.ndarray] = None):

        # 获取信号的长度
        length = len(signal)

        # 初始化采样频率信息
        if fs is None:
            fs = length

        # 获取待分解的本征模态函数的数目和初始化的本征函数
        max_imfs, iniIF = self._check_IF(signal=signal, iniIF=iniIF)

        # Initial the time variables
        t = np.arange(0, length) / fs

        # Construct the second-order difference matrix H
        H, HtH = build_second_order_diff(N=length)

        # Form the 2K block second-order difference matrix D
        D = build_block_second_order_diff(H=H, K=max_imfs)

        # Initialization
        sinm, cosm = np.zeros(shape=(max_imfs, length)), np.zeros(shape=(max_imfs, length))

        # the two demodulated quadrature signals
        uk, vk = np.zeros(shape=(max_imfs, length)), np.zeros(shape=(max_imfs, length))

        # IF record
        IFsetiter = np.zeros(shape=(max_imfs, length, self.max_iter + 1))
        IFsetiter[:, :, 0] = iniIF

        # Mode record
        Modeset_iter = np.zeros(shape=(max_imfs, length, self.max_iter + 1))
        Modeset_iter[:, :, 0] = 0


# if __name__ == '__main__':
#     a = np.ones((2, 10))
#     print(np.hstack([a, a]).shape)
#     print(a * np.array([[1], [2]]))
#     print(np.array([[1], [2]]).shape)


import numpy as np
from scipy.sparse import spdiags


def build_second_order_diff(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造与 MATLAB 代码等价的二阶差分矩阵 H 及其转置乘积 HtH。

    :param N: int, the length of inputs signal.
    :return: H : scipy.sparse.csr_matrix
                 大小为 (N-2) × N 的二阶差分矩阵
             HtH : scipy.sparse.csr_matrix
                   H' * H
    """
    # 构造三条对角线的数据
    ones = np.ones(N)
    twos = -2 * ones
    data = np.vstack((ones, twos, ones))  # 3 × N
    diags = np.array([0, 1, 2])  # 对应 0, 1, 2 对角线

    # spdiags 在 SciPy 中的顺序是 (data, diags, m, n)
    H = spdiags(data, diags, N - 2, N, format='csr')

    HtH = H.T @ H
    return H, HtH


#
# # ------- 示例用法 --------
# if __name__ == "__main__":
#     N = 100
#     H, HtH = build_second_order_diff(N)
#     print("H shape:", H.shape)
#     print("HtH shape:", HtH.shape)
#     # 如需转成稠密矩阵查看
#     # print(H.toarray())
#     # print(HtH.toarray())

import numpy as np
from scipy.sparse import block_diag


def build_block_second_order_diff(H, K):
    """
    构造 2K 块二阶差分矩阵 D。

    Parameters
    ----------
    H : scipy.sparse.spmatrix
        单个二阶差分矩阵（来自前面函数 build_second_order_diff 的返回值）
    K : int
        重复次数

    Returns
    -------
    D : scipy.sparse.csr_matrix
        大小为 (K * (N-2)) × (K * N) 的块对角矩阵
    """
    # 将 H 重复 2*K 次，然后沿对角线拼接
    blocks = [H] * (2 * K)
    D = block_diag(blocks, format='csr')
    return D


# ------- 示例用法 --------
if __name__ == "__main__":
    N = 10
    K = 3
    H, _ = build_second_order_diff(N)
    D = build_block_second_order_diff(H, K)
    print("D shape:", D.shape)
    # 如需查看稠密形式
    # print(D.toarray())
