# # -*- coding: utf-8 -*-
# """
# Created on 2025/02/05 20:55:47
# @author: Whenxuan Wang
# @email: wwhenxuan@gmail.com
# """
# import numpy as np
# from scipy.sparse import block_diag
#
# try:
#     from scipy.integrate import cumulative_trapezoid
# except ImportError:
#     from scipy.integrate import cumtrapz as cumulative_trapezoid
#
# from typing import Optional, Tuple
#
#
# class AVNCMD(object):
#     """
#     Adaptive Variational Nonlinear Chirp Mode Decomposition.
#
#     [1] Liang, Hao and Ding, Xinghao and Jakobsson, Andreas and Tu, Xiaotong and Huang, Yue.
#     "Adaptive Variational Nonlinear Chirp Mode Decomposition",
#     in 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.
#     [2] Chen S, Dong X, Peng Z, et al. Nonlinear chirp mode decomposition: A variational method.
#     IEEE Transactions on Signal Processing, 2017.
#     """
#
#     def __init__(self,
#                  max_imfs: Optional[int] = 3,
#                  beta: Optional[float] = 1e-6,
#                  tol: Optional[float] = 1e-5,
#                  max_iter: Optional[int] = 300,
#                  random_state: Optional[int] = 42,
#                  dtype: np.dtype = np.float64, ):
#         """
#         :param max_imfs:
#         :param beta: filter parameter
#         :param tol: tolerance of convergence criterion
#         :param max_iter: the maximum allowable iterations
#         :param random_state:
#         :param dtype:
#         """
#         self.beta = beta
#         self.tol = tol
#
#         self.DTYPE = dtype
#
#         self.max_imfs = max_imfs
#
#         self.max_iter = max_iter
#
#         # 记录随机种子并创建随机数生成器
#         self.random_state = random_state
#         self.rng = np.random.RandomState(seed=random_state)
#
#     def _check_IF(self, signal: np.ndarray, iniIF: Optional[np.ndarray] = None):
#         """初始化本征函数"""
#         length = len(signal)
#         if iniIF is None:
#             # 如果用户未传入初始化的本征函数
#             iniIF = np.ones(shape=(self.max_imfs, length))
#             for row in range(length):
#                 # 遍历其中的每一行信息表示一个本征模态函数
#                 iniIF[row, :] = self.rng.randint(10, 500)
#         elif iniIF is not None and isinstance(iniIF, np.ndarray):
#             # 如果用户传入了初始化的本征函数并且不为空
#             num_imfs, len_iniIF = iniIF.shape
#             if num_imfs != self.max_imfs:
#                 # 修正要获得的本征模态函数的数目
#                 self.max_imfs = num_imfs
#             if len_iniIF != length:
#                 # 如果用户传入的初始化本征函数与输入信号的长度不相同
#                 if len_iniIF > length:
#                     iniIF = iniIF[:, :length]
#                 else:
#                     iniIF = np.ones(shape=(num_imfs, length)) * np.array([[value] for value in iniIF[:, 0]])
#         else:
#             raise ValueError("Wrong Inputs for iniIF!")
#
#         return self.max_imfs, iniIF
#
#     def differ(self, y: np.ndarray, delta: float) -> np.ndarray:
#         """
#         Compute the derivative of a discrete time series y.
#
#         :param y: The input signal.
#         :param delta: The sampling time interval of y.
#         :return: numpy.ndarray: The derivative of the time series.
#         """
#         L = len(y)
#         ybar = np.zeros(L - 2, dtype=self.DTYPE)
#
#         for i in range(1, L - 1):
#             ybar[i - 1] = (y[i + 1] - y[i - 1]) / (2 * delta)
#
#         # Prepend and append the boundary differences
#         ybar = np.concatenate(
#             (
#                 np.array([(y[1] - y[0]) / delta], dtype=self.DTYPE),
#                 ybar,
#                 np.array([(y[-1] - y[-2]) / delta], dtype=self.DTYPE),
#             )
#         )
#
#         return ybar
#
#     def fit_transform(self, signal: np.ndarray, fs: Optional[int] = None, iniIF: Optional[np.ndarray] = None):
#
#         # 获取信号的长度
#         length = len(signal)
#
#         # 初始化采样频率信息
#         if fs is None:
#             fs = length
#
#         # 获取待分解的本征模态函数的数目和初始化的本征函数
#         max_imfs, iniIF = self._check_IF(signal=signal, iniIF=iniIF)
#
#         # Initial the time variables
#         t = np.arange(0, length) / fs
#
#         # Construct the second-order difference matrix H
#         H, HtH = build_second_order_diff(N=length)
#
#         # Form the 2K block second-order difference matrix D
#         D = build_block_second_order_diff(H=H, K=max_imfs)
#
#         # Initialization
#         sinm, cosm = np.zeros(shape=(max_imfs, length)), np.zeros(shape=(max_imfs, length))
#
#         # the two demodulated quadrature signals
#         uk, vk = np.zeros(shape=(max_imfs, length)), np.zeros(shape=(max_imfs, length))
#
#         # IF record
#         IFsetiter = np.zeros(shape=(max_imfs, length, self.max_iter + 1))
#         IFsetiter[:, :, 0] = iniIF
#
#         # Mode record
#         Modeset_iter = np.zeros(shape=(max_imfs, length, self.max_iter + 1))
#         Modeset_iter[:, :, 0] = 0
#
#         # Initialize the dictionary matrix A
#         # TODO: 注意这个函数这样写是否可以，后期纠错要注意
#         A, sinm, cosm = initialize_A(t=t, iniIF=iniIF, sinm=sinm, cosm=cosm, N=length, K=max_imfs)
#
#         # start iterations
#         it = 0  # iteration counter
#         sDif = self.tol + 1  # tolerance counter
#
#         while sDif > self.tol and it < self.max_iter:
#             # 开始主循环进行迭代
#
#             # Gradually increase the filter parameter during the iterations
#             beta_thin = 10 ** (it / 36 - 10)
#             if beta_thin > self.beta:
#                 beta_thin = self.beta
#
#             # Estimating the Nonlinear Chirp Signal
#
#
# # if __name__ == '__main__':
# #     a = np.ones((2, 10))
# #     print(np.hstack([a, a]).shape)
# #     print(a * np.array([[1], [2]]))
# #     print(np.array([[1], [2]]).shape)
#
#
# import numpy as np
# from scipy.sparse import spdiags
#
#
# def build_second_order_diff(N: int) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     构造与 MATLAB 代码等价的二阶差分矩阵 H 及其转置乘积 HtH。
#
#     :param N: int, the length of inputs signal.
#     :return: H : scipy.sparse.csr_matrix
#                  大小为 (N-2) × N 的二阶差分矩阵
#              HtH : scipy.sparse.csr_matrix
#                    H' * H
#     """
#     # 构造三条对角线的数据
#     ones = np.ones(N)
#     twos = -2 * ones
#     data = np.vstack((ones, twos, ones))  # 3 × N
#     diags = np.array([0, 1, 2])  # 对应 0, 1, 2 对角线
#
#     # spdiags 在 SciPy 中的顺序是 (data, diags, m, n)
#     H = spdiags(data, diags, N - 2, N, format='csr')
#
#     HtH = H.T @ H
#     return H, HtH
#
#
# def build_block_second_order_diff(H, K):
#     """
#     构造 2K 块二阶差分矩阵 D。
#
#     Parameters
#     ----------
#     H : scipy.sparse.spmatrix
#         单个二阶差分矩阵（来自前面函数 build_second_order_diff 的返回值）
#     K : int
#         重复次数
#
#     Returns
#     -------
#     D : scipy.sparse.csr_matrix
#         大小为 (K * (N-2)) × (K * N) 的块对角矩阵
#     """
#     # 将 H 重复 2*K 次，然后沿对角线拼接
#     blocks = [H] * (2 * K)
#     D = block_diag(blocks, format='csr')
#     return D
#
#
# from scipy.sparse import diags
#
#
# def initialize_A(t, iniIF, sinm, cosm, N, K):
#     """
#     Initialize the dictionary matrix A.
#     构造与 MATLAB 代码等价的字典矩阵 A。
#
#     Parameters
#     ----------
#     t     : (N,) array_like
#         时间向量
#     iniIF : (K, N) array_like
#         K 条瞬时频率曲线，单位 Hz
#     N     : int
#         信号长度
#     K     : int
#         分量个数
#
#     Returns
#     -------
#     A : ndarray, shape (N, 2*K*N)
#         稠密字典矩阵
#     """
#     A = np.zeros((N, 2 * K * N))
#
#     # 逐分量计算正弦、余弦对角矩阵并填充 A
#     for i in range(K):
#         # 累积积分，得到相位
#         phase = 2 * np.pi * cumulative_trapezoid(iniIF[i, :], t, initial=0)
#         sin_vec = np.sin(phase)
#         cos_vec = np.cos(phase)
#
#         sinm[i, :] = sin_vec
#         cosm[i, :] = cos_vec
#
#         # 构造稀疏对角矩阵（这里也可稠密，但 MATLAB 用 spdiags，保持一致）
#         Sk = diags(sin_vec, 0, shape=(N, N)).toarray()
#         Ck = diags(cos_vec, 0, shape=(N, N)).toarray()
#
#         # 横向拼接 [Ck, Sk]，再填到 A 的对应列
#         Ak = np.hstack((Ck, Sk))
#         start_col = 2 * i * N
#         end_col = 2 * (i + 1) * N
#         A[:, start_col:end_col] = Ak
#
#     return A, sinm, cosm
#
#
# # # ------------------ 示例用法 ------------------
# # if __name__ == "__main__":
# #     # 假设 N=1000, K=3
# #     N = 1000
# #     K = 3
# #     t = np.linspace(0, 1, N)
# #
# #     # 随机生成 3 条瞬时频率曲线作为示例
# #     np.random.seed(0)
# #     iniIF = np.random.rand(K, N) * 10  # 10 Hz 以内
# #
# #     A = initialize_A(t, iniIF, None, None, N, K)
# #     print("A shape:", A.shape)  # (N, 2*K*N)
#
#
# def bayesian_strategy(Phi, y):
#     """
#     The optimization problem may be expressed as
#     minimize   alpha*|| y - Phi w ||_2^2 + || w ||_2^2
#
#     This code is based on the Fast_RVM code available from http://www.miketipping.com/sparsebayes.htm
#     [1] Tipping, Michael E, Sparse Bayesian learning and the relevance vector machine, Journal of machine learning research, 2001.
#     [2] Tipping, Michael E and Faul, Anita C, Fast marginal likelihood maximisation for sparse Bayesian models, International workshop on artificial intelligence and statistics, 2003
#     Please check the accompanying license and the license of [1] and [2] before using.
#
#     :param Phi: transformed dictionary
#     :param y: transformed sampled signal
#     :return: solution of the above optimization problem
#     """
#
#     # parameter setting
#     gamma_0 = np.std(y) ** 2 / 1e2
#     eta = 1e-8
#     maxIter = 1000
#
#     # find initial gamma
#     _, m = Phi.shape
#
#     PHIt = np.matmul(Phi.T, y)
#     PHI2 = np.sum(Phi ** 2).T
#
#     ratio = (PHIt ** 2) / PHI2
#
#     # [maxr,index] = max(ratio);
#     index = np.argmax(ratio)
#     maxr = ratio[index]
#
#     gamma = PHI2[index] / (maxr - gamma_0)
#
#     # Compute initial mu, Sig, S, Q
#     phi = Phi[:, index]  # phi_i
#     Hessian = gamma + np.matmul(phi.T, phi) / gamma_0
#
#     Sig = 1 / Hessian
#
#     mu = Sig * PHIt[index] / gamma_0
#     left = np.matmul(Phi.T, phi) / gamma_0  # TODO: 这个left是矩阵还是常数
#
#     S = PHI2 / gamma_0 - Sig * (left ** 2)
#     Q = PHIt / gamma_0 - Sig * PHIt[index] / gamma_0 * left
#
#     for count in range(maxIter):
#         # Calculate si and qi
#         s, q = S, Q
#
#         s[index] = gamma * S[index] / (gamma - S[index])
#         q[index] = gamma * Q[index] / (gamma - Q[index])
#         theta = q ** 2 - s
#
#         # Choice the next alpha that maximizes marginal likelihood
#         ml = -np.inf * np.ones(m)
#         ig0 = np.where(theta > 0)
#
#         # Index for re-estimate
#
#
#
#
#
#
#
