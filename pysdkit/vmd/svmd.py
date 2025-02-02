# -*- coding: utf-8 -*-
"""
Created on 2024/6/2 17:54
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from numpy.linalg import norm
from scipy.signal import savgol_filter

from .base import Base

from typing import Optional, Tuple


class SVMD(Base):
    """
    Successive variational mode decomposition.
    Mojtaba Nazari, Sayed Mahmoud Sakhaei.
    Signal Processing 174 (2020) 107610.
    """

    def __init__(self,
                 max_alpha: int = 20000,
                 tau: float = 0.0,
                 tol: float = 1e-6,
                 stopc: Optional[int] = 4,
                 init_omega: int = 0,
                 max_iter: int = 300,
                 poly_order: Optional[int] = 8,
                 window_length: Optional[int] = 25,
                 random_seed: int = 42) -> None:
        """
        Input and Parameters:
        :param max_alpha: the balancing parameter of the data-fidelity constraint (compactness of mode)
        :param tau: time-step of the dual ascent. Set it to 0 in presence of high-level noise.
        :param tol: tolerance of convergence criterion; typically around 1e-6.
        :param stopc: the type of stoping criteria:
               1 - In the Presence of Noise (or recommended for the signals with compact spectrums such as EEG);
               2 - For Clean Signal (Exact Reconstruction);
               3 - Bayesian Estimation Method;
               4 - Power of the Last Mode (default).
        :param init_omega: initialization type of center frequency (not necessary to set):
               0- the center frequencies initiate from 0 (for each mode)
               1- the center frequencies initiate randomly with this condition: each new initial value must not be equal to the center frequency of previously extracted modes.
        :param max_iter: number of iterations to obtain each mode.
        :param poly_order: Savitzky-Golay滤波器中多项式的阶数。
        :param window_length: Savitzky-Golay滤波器中的滑动窗口长度（必须是奇数）。
        :param random_seed: the random seed.
        """
        super().__init__()
        # 对于SVMD算法的基本参数
        self.max_alpha = max_alpha
        self.tau = tau
        self.tol = tol
        self.stopc = stopc
        self.init_omega = init_omega
        # 配置Savitzky-Golay滤波器的参数
        self.poly_order = poly_order
        self.window_length = window_length
        # 获取双精度浮点数的相对精度
        self.eps = np.finfo(np.float64).eps
        # 最大迭代次数
        self.max_iter = max_iter
        # 使用的随机数生成器
        self.rng = np.random.RandomState(seed=random_seed)

    # def fmirror
    def sgolayfilt(self, signal: np.ndarray,
                   poly_order: Optional[int] = None,
                   window_length: Optional[int] = None) -> np.ndarray:
        """Filtering the input to estimate the noise"""
        poly_order = self.poly_order if poly_order is None else poly_order
        window_length = self.window_length if window_length is None else window_length
        return savgol_filter(x=signal, polyorder=poly_order, window_length=window_length)

    @staticmethod
    def fmirror_signal_noise(signal: np.ndarray, signal_noise: np.ndarray,
                             seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Mirroring the signal and noise part to extend"""
        f_mir = np.zeros(seq_len // 2)
        f_mir_noise = np.zeros(seq_len // 2)
        f_mir[0: seq_len // 2] = signal[seq_len // 2 - 1::-1]
        f_mir_noise[0:, seq_len / 2] = signal_noise[seq_len / 2 - 1::-1]

        f_mir = np.concatenate((f_mir, signal))
        f_mir_noise = np.concatenate((f_mir_noise, signal_noise))

        f_mir = np.concatenate((f_mir, signal[: seq_len // 2]))
        f_mir_noise = np.concatenate((f_mir_noise, signal_noise[: seq_len // 2]))

        return f_mir, f_mir_noise

    def fit_transform(self, signal: np.ndarray,
                      poly_order: Optional[int] = None,
                      window_length: Optional[int] = None):
        """开始进行信号分解"""
        # TODO: Part1---Start initializing the input signal of size [num_channels, seq_len]
        seq_len = len(signal)
        if seq_len % 2 > 0:
            # Check the length of the signal
            signal = signal[: -1]
            seq_len = seq_len - 1

        # Estimating the noise
        y = self.sgolayfilt(signal=signal, poly_order=poly_order, window_length=window_length)
        signal_noise = signal - y

        fs = 1 / seq_len

        # Mirroring the signal and noise part to expend
        f, fnoise = self.fmirror_signal_noise(signal, signal_noise, seq_len=seq_len)

        # time domain (t -- >> 0: T)
        T = len(f)
        t = np.arange(1, T + 1) / T

        # Set the update step
        udiff = self.tol + self.eps
        # Discretization of spectral domain
        omega_freqs = t - 0.5 - 1 / T

        # FFT of signal(and Hilbert transform concept=making it one-sided)
        f_hat = self.fftshift(ts=self.fft(f))
        f_hat_onesided = f_hat.copy()
        f_hat_onesided[0: T // 2] = 0
        f_hat_n = self.fftshift(ts=self.fft(fnoise))
        f_hat_n_onesided = f_hat_n.copy()
        f_hat_n_onesided[0: T // 2] = 0

        # Noise power estimation
        noisepe = norm(f_hat_n_onesided, ord=2) ** 2

        # Initializing the omega_d
        omega_L = np.zeros(self.max_iter)
        # Choose the initialization methods
        if self.init_omega == 0:
            omega_L[0] = 0
        elif self.init_omega == 1:
            omega_L[0] = np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * self.rng.rand())
        else:
            raise ValueError("the input param 'init_omega' must be 0 or 1!")

        # The initial value of alpha
        min_alpha = 10
        # The initial value of alpha
        Alpha = min_alpha
        alpha = np.zeros(1)

        # Dual variables vector
        lambda_vector = np.zeros([self.max_iter, len(omega_freqs)])

        # Keeping changes of mode spectrum
        u_hat_L = np.zeros([self.max_iter, len(omega_freqs)])

        # Main loop counter
        n = 0  # TODO: 注意这里是从0还是从1开始

        m = 0  # iteration counter for increasing alpha
        SC2 = 0  # main stopping criteria index
        l = 0  # the initial number of modes  TODO: 注意这里应该是0还是1
        bf = 0  # bit flag to increase alpha
        BIC = np.zeros(1)

        # Initialization of filter matrix
        h_hat_Temp = np.zeros([2, len(omega_freqs)])
        u_hat_Temp = np.zeros(len(omega_freqs))  # matrix 1 of modes  TODO:这里要注意
        u_hat_i = np.zeros([1, len(omega_freqs)])  # matrix 2 of modes

        # Counter for initializing omega_L
        n2 = 0

        # Initializing Power of Last Mode index
        polm = np.zeros(2)

        omega_d_Temp = np.zeros(1)  # initialization of center frequencies vector1
        sigerror = np.zeros(1)  # initializing signal error index for stopping
        gamma = np.zeros(1)  # initializing gamma
        normind = np.zeros(1)

        # TODO: Part2---Main loop for iterative updates
        while SC2 != 1:
            while Alpha < self.max_alpha + 1:
                while udiff > self.tol and n < self.max_iter:
                    # update u_L
                    u_hat_L[n + 1, :] = (f_hat_onesided + ((Alpha ** 2) * omega_freqs - omega_L[n] ** 2)
                                         ** u_hat_L[n, :] + lambda_vector[n, :] / 2) / (
                                                1 + (Alpha ** 2) * (omega_freqs - omega_L[n]) ** 4 * (
                                                1 + 2 * Alpha * (omega_freqs - omega_L[n]) ** 2) + np.sum(h_hat_Temp))
                    # update omega_L
                    omega_L[n + 1] = (omega_freqs[T // 2: T] * (np.abs(u_hat_L[n + 1, T // 2: T]) ** 2).T) / np.sum(
                        np.abs(u_hat_L[n + 1, T // 2: T]) ** 2)
                    # update lambda dual ascent
                    lambda_vector[n + 1, :] = lambda_vector[n, :] + self.tau * (f_hat_onesided - (u_hat_L[n + 1, :] + (
                            (Alpha ** 2) * (omega_freqs - omega_L[n]) ** 4 * (
                            f_hat_onesided - u_hat_L[n + 1, :] - np.sum(u_hat_i) + lambda_vector[n,
                                                                                   :] / 2) - np.sum(u_hat_i))))

                    udiff = self.eps

                    # 1st loop criterion
                    udiff = udiff + (1 / T * (u_hat_L[n + 1, :] - u_hat_L[n, :]) * np.conj(
                        (u_hat_L[n + 1, :] - u_hat_L[n, :]).T)) / (1 / T * (u_hat_L[n, :]) * np.conj((u_hat_L[n, :])).T)

                    udiff = np.abs(udiff)
                    n += 1

            # TODO: Part 3---Increasing Alpha to achieve a pure mode
            if np.abs(m - np.log(self.max_alpha)) > 1:
                m = m + 1
            else:
                m = m + 0.05
                bf = bf + 1

            if bf > 2:
                Alpha = Alpha + 1

            if Alpha <= (self.max_alpha - 1):  # exp(SC1) <= (max_alpha)
                if bf == 1:
                    Alpha = self.max_alpha - 1
                else:
                    Alpha = np.exp(m)

                omega_L = omega_L[n]

                # Initializing
                udiff = self.tol + self.eps  # update step
                temp_ud = u_hat_L[n, :]  # keeping the last update of the obtained mode

                n = 0  # loop counter  TODO: 注意这里是1还是0

                lambda_vector = np.zeros([self.max_iter, len(omega_freqs)])
                u_hat_L = np.zeros([self.max_iter, len(omega_freqs)])
                u_hat_L[n, :] = temp_ud

        # TODO: Part 4---Saving the Modes and Center Frequencies
        omega_L = omega_L[omega_L > 0]
        u_hat_Temp = u_hat_L[n, :]
        omega_d_Temp[l] = omega_L[n - 1, 0]

        alpha[l] = Alpha
        Alpha = min_alpha
        bf = 0

        # Initializing omega_L
        if self.init_omega > 0:
            ii = 0
            while ii < 1 and n2 < 300:

                omega_L = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * self.rng.rand()))

                checkp = np.abs(omega_d_Temp - omega_L)

                if len(np.where(checkp < 0.02)[1]) <= 0:
                    # It will continue if difference between previous vector of omega_d and the current random omega_plus is about 2Hz
                    ii = 1
                n2 += 1
        else:
            omega_L = 0

        # Update step
        udiff = self.tol + self.eps

        lambda_vector = np.zeros([self.max_iter, len(omega_freqs)])

        gamma[l] = 1

        h_hat_Temp[l, :] = gamma[l] / ((alpha[l] ** 2) * (omega_freqs - omega_d_Temp[l]) ** 4)

        # Keeping the last desired mode as one of the extracted modes
        u_hat_i[l, :] = u_hat_Temp

        # TODO: Part 5---Stopping Criteria
        if self.stopc is not None:
            # Checking the stopping criteria

            if self.stopc == 1:
                # In the presence of noise
                if u_hat_i.shape[0] == 1:
                    sigerror[l] = norm((f_hat_onesided - u_hat_L), ord=2) ** 2
                else:
                    sigerror[l] = norm((f_hat_onesided - np.sum(u_hat_i)), ord=2) ** 2

                if n2 >= 300 or sigerror[l] <= np.round(noisepe):
                    SC2 = 1

            elif self.stopc == 2:
                # Exact Reconstruction
                sum_u = np.sum(u_hat_Temp[0, :])  # sum of current obtained modes
                normind[l] = (1 / T) * (norm(sum_u - f_hat_onesided, ord=2) ** 2) / ((1 / T) * norm(f_hat_n_onesided, ord=2) ** 2)
                if n2 >= 300 or normind[l] < 0.005:
                    SC2 = 1

            elif self.stopc == 3:
                # Bayesian Method
                if u_hat_i.shape[0] == 1:
                    sigerror[l] = norm((f_hat_onesided - u_hat_i), ord=2) ** 2
                else:
                    sigerror[l] = norm((f_hat_onesided - np.sum(u_hat_i)), ord=2) ** 2

                BIC[l] = 2 * T * np.log(sigerror[l]) + (3 * (l + 1)) * np.log(2 * T)

                if l > 0:
                    if BIC[l] > BIC[l - 1]:
                        SC2 = 1

            elif self.stopc == 4:
                # Power of the last mode
                if l < 1:
                    polm[l] = norm((4 * Alpha * u_hat_i[l, :] / (1 + 2 * Alpha * (omega_freqs - omega_d_Temp[l, :]) ** 2)) * u_hat_i[l, :].T, ord=2)
                    polm_temp = polm[l]
                    polm[l] = polm[l] / np.max(polm_temp)
                else:
                    polm[l] = norm((4 * Alpha * u_hat_i[l, :] / (1 + 2 * Alpha * ())))










