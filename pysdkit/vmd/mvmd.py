# -*- coding: utf-8 -*-
"""
Created on 2024/6/1 18:44
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from typing import Optional, Tuple
from .base import Base


class MVMD(Base):
    """
    Multivariate Variational mode decomposition, object-oriented interface.
    ur Rehman, Naveed and Aftab, Hania (2019) 'Multivariate Variational Mode Decomposition',
    IEEE Transactions on Signal Processing, 67(23), pp. 6039–6052.
    Python code: https://github.com/yunyueye/MVMD
    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/72814-multivariate-variational-mode-decomposition-mvmd
    """

    def __init__(self, alpha: float, K: int, tau: float, init: str = "zero", DC: bool = False,
                 tol: float = 1e-7, max_iter: int = 100) -> None:
        """
        Multivariate Variational Mode Decomposition (MVMD) algorithm.
        :param alpha: float
            The balancing parameter of the data-fidelity constraint, controlling the trade-off
            between the smoothness of the modes and the accuracy of the reconstruction.
        :param K: int
            The number of modes to be recovered, determining how many intrinsic mode functions
            are decomposed from the input signal.
        :param tau: float
            The time-step of the dual ascent optimization algorithm. Setting tau to 0 applies a
            noise-slack variable that aids in noise robustness.
        :param init: str, optional
            Initialization method for the center frequencies of the modes. Can be 'zero' (all omegas
            start at 0 frequency), 'uniform' (uniformly distributed), or 'random' (random values).
            Defaults to 'zero'.
        :param DC: bool, optional
            If True, the first mode is constrained to zero frequency, extracting the mean (DC component)
            of the signal. Defaults to False.
        :param tol: float, optional
            Tolerance for convergence. The algorithm stops when the difference between spectra
            across iterations is below this threshold. Typically around 1e-6 to 1e-7. Defaults to 1e-7.
        :param max_iter: int, optional
            Maximum number of iterations if convergence is not reached. Defaults to 100.
        Attributes:
        -----------
        DTYPE : numpy.dtype
            The data type for internal computations, set to numpy.complex64 to optimize
            performance by reducing memory and computation requirements.
        """
        super().__init__()
        self.alpha = alpha
        self.K = K
        self.tau = tau
        self.init = init.lower()
        self.DC = DC
        self.tol = tol
        self.max_iter = max_iter
        self.DTYPE = np.complex64

    def __call__(self, signal: np.ndarray, return_all: bool = False) -> Optional[np.ndarray]:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal, return_all=return_all)

    def __init_omega(self, fs: float) -> np.ndarray:
        """Initialization of omega_k"""
        omega_plus = np.zeros(shape=(self.max_iter, self.K), dtype=self.DTYPE)
        if self.init == "uniform":
            for i in range(1, self.K + 1):
                omega_plus[0, i - 1] = (0.5 / self.K) * (i - 1)
        elif self.init == "random":
            omega_plus[0, :] = np.sort(np.exp(np.log(fs)) +
                                       (np.log(0.5) - np.log(fs)) * np.random.rand(1, self.K))
        else:
            omega_plus[0, :] = 0
        # Processing the DC component of the signal
        if self.DC is True:
            omega_plus[0, 0] = 0
        return omega_plus

    def fit_transform(self, signal: np.ndarray,
                      return_all: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """
        Multivariate signal decomposition using MVMD algorithm
        :param signal: the time domain signal (ndarray) to be decomposed
        :param return_all: Whether to return all results of the algorithm, False only return the collection of decomposed modes
        :return:  u       - the collection of decomposed modes, shape: [K, length, num_channels]
                  u_hat   - spectra of the modes,
                  omega   - estimated mode center-frequencies
        """

        # Get the number of channels and the length of the input signal
        C, T = signal.shape

        # Get the sampling frequency of the signal
        fs = 1.0 / float(T)

        # Perform signal mirroring expansion
        fMirr = self.multi_fmirror(ts=signal, C=C, T=T)

        # Get the new length of the signal
        T = int(fMirr.shape[1])
        t = np.linspace(1.0 / float(T), 1, int(T))

        # Discretization in the frequency domain
        freqs = t - 0.5 - 1 / T

        # Each mode has its own alpha
        Alpha = self.alpha * np.ones(self.K, dtype=self.DTYPE)

        # Construct and center f_hat
        f_hat = self.fftshift(ts=self.fft(ts=fMirr))
        f_hat_plus = f_hat
        f_hat_plus[:, 0: int(T / 2)] = 0

        # matrix keeping track of every iterant // could be discarded for mem
        u_hat_plus = np.zeros(shape=(self.max_iter, len(freqs), self.K, C), dtype=self.DTYPE)

        omega_plus = self.__init_omega(fs=fs)

        # Start with empty dual variables
        lambda_hat = np.zeros(shape=(self.max_iter, len(freqs), C), dtype=self.DTYPE)

        # Initialization of omega_k
        uDiff = self.tol + np.spacing(1)
        # Loop counter
        n = 1
        # Accumulator
        sum_uk = np.zeros(shape=(len(freqs), C))

        """Start main loop"""
        while uDiff > self.tol and n < self.max_iter - 1:
            # update first mode accumulator
            k = 1
            sum_uk = u_hat_plus[n - 1, :, self.K - 1, :] + sum_uk - u_hat_plus[n - 1, :, 0, :]

            # update spectrum of first mode through Wiener filter of residuals
            for c in range(C):
                u_hat_plus[n, :, k - 1, c] = (f_hat_plus[c, :] - sum_uk[:, c] - lambda_hat[n - 1, :, c] / 2) \
                                             / (1 + Alpha[k - 1] * np.square(freqs - omega_plus[n - 1, k - 1]))

            # update first omega if not held at 0
            if self.DC is False:
                omega_plus[n, k - 1] = np.sum(np.matmul(np.expand_dims(freqs[T // 2: T], axis=0),
                                                        np.square(np.abs(u_hat_plus[n, T // 2: T, k - 1, :])))) \
                                       / np.sum(np.square(np.abs(u_hat_plus[n, T // 2: T, k - 1, :])))

            for k in range(2, self.K + 1):

                # update mode accumulator
                print(n)
                sum_uk = u_hat_plus[n, :, k - 2, :] + sum_uk - u_hat_plus[n - 1, :, k - 1, :]

                # update mode spectrum
                for c in range(C):
                    u_hat_plus[n, :, k - 1, c] = (f_hat_plus[c, :] - sum_uk[:, c] -
                                                  lambda_hat[n - 1, :, c] / 2) \
                                                 / (1 + Alpha[k - 1] * np.square(freqs - omega_plus[n - 1, k - 1]))

                # center frequencies
                omega_plus[n, k - 1] = np.sum(np.matmul(np.expand_dims(freqs[T // 2: T], axis=0),
                                                        np.square(np.abs(u_hat_plus[n, T // 2: T, k - 1, :])))) \
                                       / np.sum(np.square(np.abs(u_hat_plus[n, T // 2: T, k - 1, :])))

            # Dual ascent
            lambda_hat[n, :, :] = lambda_hat[n - 1, :, :] + self.tau * np.sum(u_hat_plus[n, :, :, :], axis=1)

            # loop counter
            n = n + 1

            # Determine whether the algorithm converges
            uDiff = np.spacing(1)
            for i in range(1, self.K + 1):
                uDiff = uDiff + 1 / float(T) * np.dot(
                    u_hat_plus[n - 1, :, i - 1, :] - u_hat_plus[n - 2, :, i - 1, :],
                    np.conj(u_hat_plus[n - 1, :, i - 1, :] - u_hat_plus[n - 2, :, i - 1, :]).T
                )

            uDiff = np.sum(np.abs(uDiff))

        N = min(n, self.max_iter)
        omega = omega_plus[0: N, :]

        # Signal reconstruction
        u_hat = np.zeros(shape=(T, self.K, C), dtype=self.DTYPE)
        for c in range(C):
            u_hat[T // 2: T, :, c] = np.squeeze(u_hat_plus[N - 1, T // 2: T, :, c])
            second_index = list(range(1, T // 2 + 1))
            second_index.reverse()
            u_hat[second_index, :, c] = np.squeeze(np.conj(u_hat_plus[N - 1, T // 2:T, :, c]))
            u_hat[0, :, c] = np.conj(u_hat[-1, :, c])

        u = np.zeros(shape=(self.K, len(t), C), dtype=self.DTYPE)

        for k in range(1, self.K + 1):
            for c in range(C):
                u[k - 1, :, c] = (self.ifft(self.ifftshift(u_hat[:, k - 1, c]))).real

        # remove mirror part
        u = u[:, T // 4: 3 * T // 4, :]

        # recompute spectrum
        u_hat = np.zeros(shape=(T // 2, self.K, C), dtype=self.DTYPE)

        for k in range(1, self.K + 1):
            for c in range(C):
                u_hat[:, k - 1, c] = self.fftshift(ts=self.fft(ts=u[k - 1, :, c])).conj()

        # ifftshift
        u = np.fft.ifftshift(u, axes=-1)

        if return_all is True:
            return u.real, u_hat, omega
        else:
            return u.real
