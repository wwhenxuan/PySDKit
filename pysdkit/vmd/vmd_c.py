# -*- coding: utf-8 -*-
"""
Created on Sat Mar 4 11:59:21 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from typing import Optional, List, Tuple

from .base import Base
from pysdkit.plot import plot_IMFs


class VMD(Base):
    """
    Variational mode decomposition, object-oriented interface.
    Original paper: Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’,
    IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.
    Python code: https://github.com/vrcarva/vmdpy
    MATLAB code: https://www.mathworks.com/help/wavelet/ref/vmd.html
    """

    def __init__(self, alpha: int, K: int, tau: float, init: str = 'uniform', DC: bool = False,
                 max_iter: int = 500, tol: float = 1e-6) -> None:
        """
        :param alpha: the balancing parameter of the data-fidelity constraint
        :param K: the number of modes to be recovered
        :param tau: time-step of the dual ascent ( pick 0 for noise-slack )
        :param init: uniform = all omegas start uniformly distributed
                     zero = all omegas initialized randomly
                     random = all omegas start at 0
        :param DC: true if the first mode is put and kept at DC (0-freq)
        :param max_iter: Maximum number of iterations
        :param tol: tolerance of convergence criterion; typically around 1e-6
        """
        super().__init__()
        # parameters of VMD signal decomposition algorithm
        self.alpha = alpha
        self.K = K
        self.tau = tau
        self.init = init
        self.DC = DC
        self.max_iter = max_iter
        self.tol = tol

        # The last input original signal
        self.signal = None

        # the result of signal decomposition
        self.u, self.u_hat, self.omega = None, None, None

    def __call__(self, signal: np.ndarray, return_all: bool = False) -> Optional[np.ndarray]:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal, return_all=return_all)

    def __init_omega(self, fs: float) -> np.ndarray:
        """Initialization of omega_k"""
        omega_plus = np.zeros([self.max_iter, self.K])
        if self.init.lower() == 'uniform':
            for i in range(self.K):
                omega_plus[0, i] = (0.5 / self.K) * i
        elif self.init.lower() == 'random':
            omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1, self.K)))
        elif self.init.lower() == 'zero':
            omega_plus[0, :] = 0.
        else:
            raise ValueError
        return omega_plus

    def plot_IMFs(self, max_imf: int = -1, colors: Optional[List] = None, save_figure: bool = False,
                  return_figure: bool = False, dpi: int = 500, fontsize: float = 14,
                  spine_width: float = 2, labelpad: float = 10, save_name: Optional[str] = None) -> None:
        """
        An easy way to visualize signal decomposition results
        :param max_imf: The number of decomposition modes to be plotted
        :param colors: List of color strings for plotting
        :param save_figure: Whether to save the figure as an image
        :param return_figure: Whether to return the figure object
        :param dpi: The resolution of the saved image
        :param fontsize: The font size of the axis labels
        :param spine_width: The width of the visible axes spines
        :param labelpad: Controls the filling distance of the y-axis coordinate
        :param save_name: The name of the saved image file
        :return: The figure object for the plot
        """
        if self.u is not None and self.signal is not None:
            plot_IMFs(signal=self.signal, IMFs=self.u, max_imf=max_imf, colors=colors, save_figure=save_figure,
                      return_figure=return_figure, dpi=dpi, fontsize=fontsize,
                      spine_width=spine_width, labelpad=labelpad, save_name=save_name)
        else:
            raise ValueError

    def fit_transform(self, signal: np.ndarray,
                      return_all: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """
        Signal decomposition using VMD algorithm
        :param signal: the time domain signal (1D numpy array)  to be decomposed
        :param return_all: Whether to return all results of the algorithm, False only return the collection of decomposed modes,
                           True plus the spectra of the modes and the estimated mode center-frequencies
        :return:  u       - the collection of decomposed modes,
                  u_hat   - spectra of the modes,
                  omega   - estimated mode center-frequencies
        """
        if len(signal) % 2 == 1:
            signal = signal[:-1]

        # Period and sampling frequency of input signal
        fs = 1. / len(signal)

        # Mirror expansion of signals
        sym = len(signal) // 2
        fMirr = self.fmirror(ts=signal, sym=sym)
        # Time Domain 0 to T (of mirrored signal)
        T = len(fMirr)
        t = np.arange(1, T + 1) / T

        # Spectral Domain
        freqs = t - 0.5 - (1 / T)

        # Construct and center f_hat
        f_hat = self.fftshift(ts=self.fft(ts=fMirr))
        f_hat_plus = np.copy(f_hat)
        f_hat_plus[: T // 2] = 0

        # For future generalizations: individual alpha for each mode
        alpha = np.ones(self.K) * self.alpha
        # matrix keeping track of every iterant // could be discarded for mem
        u_hat_plus = np.zeros([self.max_iter, len(freqs), self.K], dtype=complex)
        # Initialization of omega_k
        omega_plus = self.__init_omega(fs=fs)
        if self.DC:
            omega_plus[0, 0] = 0
        # start with empty dual variables
        lambda_hat = np.zeros(shape=[self.max_iter, len(freqs)], dtype=complex)

        sum_uk = 0  # accumulator
        convergence = np.spacing(1) + self.tol  # Determine whether the algorithm converges

        # Main loop for iterative updates
        for n in range(0, self.max_iter - 1):
            # update spectrum of first mode through Wiener filter of residuals
            sum_uk = u_hat_plus[n, :, self.K - 1] + sum_uk - u_hat_plus[n, :, 0]
            u_hat_plus[n + 1, :, 0] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                    1. + alpha[0] * (freqs - omega_plus[n, 0]) ** 2)

            # update first omega if not held at 0
            if not self.DC:
                omega_plus[n + 1, 0] = np.dot(freqs[T // 2:T], (abs(u_hat_plus[n + 1, T // 2:T, 0]) ** 2)) / np.sum(
                    abs(u_hat_plus[n + 1, T // 2:T, 0]) ** 2)

            # update of any other mode
            for k in range(1, self.K):
                # mode spectrum
                sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
                u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                        1 + alpha[k] * (freqs - omega_plus[n, k]) ** 2)
                # center frequencies
                omega_plus[n + 1, k] = np.dot(freqs[T // 2:T], (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / np.sum(
                    abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)

            # Update Lagrange multipliers
            lambda_hat[n + 1, :] = lambda_hat[n, :] + self.tau * (np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus)

            # Determine whether the algorithm has converged
            for i in range(self.K):
                convergence = convergence + (1 / T) * np.dot((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]),
                                                             np.conj((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i])))
            convergence = np.abs(convergence)
            if convergence <= self.tol:
                break

        # discard empty space if converged early
        niter = np.min([self.max_iter, n])
        omega = omega_plus[:niter, :]
        idxs = np.flip(np.arange(1, T // 2 + 1), axis=0)

        # signal reconstruction
        u_hat = np.zeros([T, self.K], dtype=complex)
        u_hat[T // 2:T, :] = u_hat_plus[niter - 1, T // 2:T, :]
        u_hat[idxs, :] = np.conj(u_hat_plus[niter - 1, T // 2:T, :])
        u_hat[0, :] = np.conj(u_hat[-1, :])

        u = np.zeros([self.K, len(t)])
        for k in range(self.K):
            u[k, :] = np.real(self.ifft(ts=self.ifftshift(ts=u_hat[:, k])))

        # remove mirror part
        u = u[:, T // 4:3 * T // 4]

        # recompute spectrum
        u_hat = np.zeros([u.shape[1], self.K], dtype=complex)
        for k in range(self.K):
            u_hat[:, k] = self.fftshift(ts=self.fft(ts=u[k, :]))

        # record the result of this operation
        self.u = u
        self.u_hat = u_hat
        self.omega = omega
        self.signal = signal

        if return_all is True:
            return u, u_hat, omega
        else:
            return u
