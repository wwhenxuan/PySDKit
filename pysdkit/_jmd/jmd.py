# -*- coding: utf-8 -*-
"""
Created on 2025/02/05 13:31:52
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from numpy import linalg
import scipy.sparse as sp

from typing import Optional, Tuple

from plotnine.positions.position import position

from pysdkit.utils import fft, fftshift, ifft, ifftshift


class JMD(object):
    """
    Jump Plus AM-FM Mode Decomposition

    Jump Plus AM-FM Mode Decomposition (JMD) is a novel method for decomposing a nonstationary signal into amplitude-
    and frequency-modulated (AM-FM) oscillations and discontinuous (jump) components.
    Current nonstationary signal decomposition methods are designed to either obtain constituent AM-FM oscillatory modes
    or the discontinuous and residual components from the data, separately. Yet, many real-world signals of interest
    simultaneously exhibit both behaviors i.e., jumps and oscillations.
    In JMD method, we design and solve a variational optimization problem to accomplish this task.
    The optimization formulation includes a regularization term to minimize the bandwidth of all signal modes for
    effective oscillation modeling, and a prior for extracting the jump component. JMD addresses the limitations of
    conventional AM-FM signal decomposition methods in extracting jumps, as well as the limitations of existing jump
    extraction methods in decomposing multiscale oscillations.

    Mojtaba Nazari, Anders Rosendal Korshøj and Naveed ur Rehman,
    ''Jump Plus AM-FM Mode Decomposition,'' IEEE TSP (in press),
    available on arXiv: https://doi.org/10.48550/arXiv.2407.07800

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/169388-jump-plus-am-fm-mode-decomposition-jmd?s_tid=prof_contriblnk
    """

    def __init__(
            self,
            K: int,
            alpha: Optional[float] = 5000,
            init: Optional[str] = "zero",
            tol: Optional[float] = 1e-6,
            beta: Optional[float] = 0.03,
            b_bar: Optional[float] = 0.45,
            tau: Optional[float] = 5,
            max_iter: Optional[int] = 2000,
    ) -> None:
        """
        :param K: the number of modes to be recovered
        :param alpha: the balancing parameter of the mode bandwidth
        :param init: - 'zero': all omegas start at 0
                     - 'uniform': all omegas start uniformly distributed
                     - 'random': all omegas initialized randomly
        :param tol: tolerance of convergence criterion; typically around 1e-6
        :param beta: the balancing parameter of the jump constraint (1/expected number of jumps)
        :param b_bar: the balancing parameter related to the β parameter
        :param tau: the dual ascent step (set to 0 for noisy signal)
        :param max_iter: the maximum number of iterations
        """
        self.K = K
        self.alpha = alpha
        self.init = init
        self.tol = tol
        self.beta = beta
        self.b_bar = b_bar
        self.tau = tau
        self.max_iter = max_iter

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Jump Plus AM-FM Mode Decomposition (JMD)"

    def jump_step(self, freqs: np.ndarray, T: int):
        """Init the Jump part"""
        # Define and calculate b using b_bar
        b = 2 / (self.b_bar ** 2)

        # Compute gamma using tau2 and b
        gamma = self.tau * (0.5 * b * self.beta)

        # Initialize vector v with zeros, of length T
        v = np.zeros(T)

        # Create a column vector d filled with ones, of length T
        d = np.ones(shape=[T, 1])

        # Construct a sparse diagonal matrix D with sub-diagonal and super-diagonal
        diagonals = [-d.ravel(), d.ravel()]  # 主对角线和超对角线元素
        D = sp.diags(diagonals, [0, 1], shape=(T, T), format='lil')

        # Apply zero boundary condition to the last row of D
        D[-1, :] = 0

        # Compute the matrix product D' * D
        DTD = (D.T.dot(D)).toarray()

        print(DTD)

        # Initialize vector tt with zeros, of same size as vector f
        x = np.zeros(shape=(T, 1))
        # Initialize rho with the same size as x
        rho = x.copy()

        # Calculate the reciprocal of gamma
        coef1 = 1 / gamma
        # Compute mu using gamma and beta
        mu = 2 * self.beta / gamma

        # Create a sparse diagonal matrix SPDiag with d as the diagonal
        SPDiag = sp.diags(d.ravel(), 0, shape=(T, T)).toarray()

        # matrix keeping track of every iterant // could be discarded for mem
        j_hat_plus = np.zeros(shape=[self.max_iter, len(freqs)])

        return b, v, x, D.toarray(), DTD, SPDiag, j_hat_plus, rho, coef1, mu, gamma

    @staticmethod
    def enc_fmirror(signal: np.ndarray, T: int) -> np.ndarray:
        """进行镜像映射的函数"""
        # 前半部分：前半段逆序，中间原信号，后半段逆序
        f_mirror = np.concatenate((
            signal[:T // 2][::-1],  # 前T/2个元素（逆序）
            signal,  # 原始信号
            signal[T // 2:][::-1]  # 后T/2个元素（逆序）
        ))
        return f_mirror

    @staticmethod
    def dec_fmirror(u: np.ndarray, v: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray]:
        """去除信号的镜像映射部分"""
        pos = T // 4
        return u[:, pos: 3 * pos].copy(), v[:, pos: 3 * pos].copy()

    @staticmethod
    def _max(array: np.ndarray, number: int) -> np.ndarray:
        """逐元素比较寻找最大值并进行替换"""
        # 寻找小于指定值的位置
        index = np.where(array < number)
        # 进行指定值的替换
        array[index] = number

        return array.copy()

    @staticmethod
    def _min(array: np.ndarray, number: int) -> np.ndarray:
        """逐元素比较寻找最小值并进行替换"""
        # 寻找大于指定值的位置
        index = np.where(array > number)
        # 进行指定值的替换
        array[index] = number

        return array.copy()

    def _init_omega(self, fs: float) -> np.ndarray:
        """Initialization of omega_k"""
        # 初始化空数组
        omega_plus = np.zeros(shape=(self.max_iter, self.K))

        if self.init == "zero":
            # 全零初始化
            return omega_plus
        elif self.init == "uniform":
            # 通过均匀分布初始化
            for i in range(1, self.K + 1):
                omega_plus[0, i] = (0.5 / self.K) * (i - 1)
        elif self.init == "random":
            # 随机初始化
            omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5 - np.log(fs)) * np.random.rand(self.K))))
        else:
            raise ValueError("Initialization method not recognized")

        return omega_plus

    def fit_transform(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Signal decomposition using Jump Plus AM-FM Mode Decomposition algorithm

        :param signal: the time domain signal (1D numpy array) to be decomposed
        :return: the decomposed results of IMFs
        """

        # Preparations
        shift = np.mean(signal)
        # Norm
        signal = signal - np.mean(signal)

        # Period and sampling frequency of input signal
        save_T = signal.shape[0]
        fs = 1 / save_T

        # extend the signal by mirroring
        T = save_T
        f = self.enc_fmirror(signal=signal, T=T)

        # Time Domain 0 to T
        T = f.shape[0]
        half_T = T // 2
        t = np.arange(0, T) / T

        # Spectral Domain discretization
        freqs = t - 0.5 - 1 / T

        # Alpha = alpha*ones(1,K);
        a2 = 50
        t2 = np.arange(0.01, np.sqrt(2 / a2) + 0.001, 0.001)

        phi1 = (-a2 / 2) * (t2 ** 2) + (np.sqrt(2 * a2) * t2)
        phi = np.append(phi1, np.ones(self.max_iter - phi1.shape[0]))

        Alpha = self.alpha * phi

        # Construct and center f_hat
        f_hat = fftshift(ts=fft(ts=f))
        f_hat_plus = f_hat.copy()
        f_hat_plus[: T // 2] = 0  # TODO: 这里要不要使用整除

        # matrix keeping track of every iterant // could be discarded for mem
        u_hat_plus = np.zeros(shape=[self.max_iter, freqs.shape[0], self.K])

        # Initialization of omega_k
        omega_plus = self._init_omega(fs=fs)

        # other inits
        # update step
        uDiff = self.tol + np.spacing(1)
        # loop counter
        n = 0
        # accumulator
        sum_uk = 0

        # ----------- Jump part
        b, v, x, D, DTD, SPDiag, j_hat_plus, rho, coef1, mu, gamma = self.jump_step(freqs=freqs, T=T)

        # ----------- Main loop for iterative updates
        while uDiff > self.tol and n < self.max_iter:
            # not converged and below iterations limit

            # update first mode accumulator
            # Initialize sum_uk for k=1
            sum_uk = u_hat_plus[n, :, self.K - 1] + sum_uk - u_hat_plus[n, :, 0]

            # 处理其他分量
            for k in range(0, self.K):
                # Update the accumulator for k > 0
                if k > 0:
                    sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]

                # Update the spectrum of the mode through Wiener filter of residuals
                u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - j_hat_plus[n, :]) / (1 + Alpha[n] * (freqs - omega_plus[n, k] ** 2))

                # Update omega
                omega_plus[n + 1, k] = np.matmul(freqs[half_T: T], (np.abs(u_hat_plus[n + 1, half_T: T, k]) ** 2).T) / np.sum(np.abs(u_hat_plus[n + 1, half_T: T, k]) ** 2)

            # Back to time domain
            u_hat = np.zeros(shape=[T, self.K])

            for k in range(0, self.K):  # TODO: 这部分代码需要注意
                u_hat[half_T: T, k] = np.squeeze(u_hat_plus[n + 1, half_T: T, k])

                conj_values = np.squeeze(np.conj(u_hat_plus[n + 1, half_T : T, k]))
                # print(conj_values.shape, u_hat[1 : half_T, k].shape)
                u_hat[1 : half_T + 1, k] = conj_values[::-1]  # 逆序排列  TODO: 这里是不是要+1

                u_hat[0, k] = np.conj(u_hat[-1, k])

            u = np.zeros(shape=(self.K, len(t)))
            print(u.shape)

            for k in range(0, self.K):
                u[k, :] = np.real(ifft(ts=ifftshift(ts=u_hat[:, k])))

            # Update jump
            v = linalg.solve(a=(1 * SPDiag + gamma * DTD), b=((gamma * np.matmul(D.T, x) - np.matmul(D.T, rho)) + 1 * f.T - 1 * (np.sum(u, axis=0)).T))
            print(v.shape)

            # Update variable x
            Dv = np.matmul(D.T, v[:])

            h = (Dv + coef1 * rho)

            print("h", h.shape)

            # 这里的含义其实应该是逐一元素的比较


            maxi = self._max(((1 / (1 - mu * b)) * np.ones(shape=np.abs(h).shape)) - ((mu * np.sqrt(2 * b) / (1 - mu * b)) * np.ones(shape=np.abs(h).shape)) / np.abs(h), number=0)
            print(np.min(maxi, axis=1))
            # x = np.min(np.max(((1 / (1 - mu * b)) * np.ones(shape=np.abs(h).shape)) - ((mu * np.sqrt(2 * b) / (1 - mu * b)) * np.ones(shape=np.abs(h).shape)) / np.abs(h), axis=0), axis=1) * h




if __name__ == '__main__':
    # a2 = 50
    # t2 = np.arange(0.01, np.sqrt(2 / a2) + 0.001, 0.001)
    # print(t2, len(t2))
    # phi1 = (-a2 / 2) * (t2 ** 2) + (np.sqrt(2 * a2) * t2)
    # phi = np.append(phi1, np.ones(1000 - phi1.shape[0]))
    #
    # print(phi)
    a = np.ones(shape=(10, 15))
    b = np.ones(shape=(15, 10))

    print(np.where(a > 0))

    index = np.where(a > 0)

    a[index] = 2
    print(a)

    # print(np.matmul(a, b).shape)
    #
    # signal = np.random.rand(1024)
    #
    # jmd = JMD(K=3)
    #
    # jmd.fit_transform(signal=signal)
