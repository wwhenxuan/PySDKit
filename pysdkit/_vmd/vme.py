# -*- coding: utf-8 -*-
"""
Created on 2025/04/02 22:44:18
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Tuple

from pysdkit.utils import fft, fftshift, ifft, ifftshift


class VME(object):
    """
    Variational Mode Extraction, a useful decomposition algorithm to extract a specific mode from the signal.

    The VME is a robust method when there is no need to decompose the whole signal.
    Indeed, if the aim is to achieve a particular mode from the signal VME is the best choice
    (just by knowing an approximation of the frequency band of the specific mode of interest).
    Indeed, VME assumes that signal is composed of two parts: F(t)=Ud(t)+Fr(t); in which F(t) refers to input signal,
    Ud(t) is the desired mode, and Fr(t) indicates the residual signal.

    Nazari, Mojtaba, and Sayed Mahmoud Sakhaei.
    “Variational Mode Extraction: A New Efficient Method to Derive Respiratory Signals from ECG.”
    IEEE Journal of Biomedical and Health Informatics, vol. 22, no. 4,
    Institute of Electrical and Electronics Engineers (IEEE), July 2018, pp. 1059–67, doi:10.1109/jbhi.2017.2734074.

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/76003-variational-mode-extraction-vme-m?s_tid=srchtitle
    """

    def __init__(self, alpha: Optional[float] = 20000, omega_init: Optional[float] = 0.0, fs: Optional[int] = None, tau: Optional[float] = 0, tol: Optional[float] = 1e-7,
                 max_iter: Optional[int] = 300) -> None:
        """
        :param alpha: compactness of mode constraint
        :param omega_init: initial guess of mode center-frequency (Hz)
        :param fs: the sampling frequency of input signal
        :param tau: time-step of the dual ascent. set it to 0 in the presence of high level of noise.
        :param tol: tolerance of convergence criterion; typically around 1e-6
        :param max_iter: maximum number of iterations
        """
        self.alpha = alpha
        self.omega_init = omega_init
        self.fs = fs
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter

    def __call__(self, *args, **kwargs):
        """allow instances to be called like functions"""
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Variational Mode Extraction (VME)"

    def fit_transform(self, signal: np.ndarray, return_all: Optional[bool] = False) -> np.ndarray:
        """
        Signal decomposition using VME algorithm

        :param signal: the time domain signal (1D numpy array) to be decomposed
        :param return_all: whether to only return the IMFs results
        :return: the decomposed results of IMFs
        """

        # ----- Part 1: Start initializing
        save_T = signal.shape[0]
        fs = save_T if self.fs is None else self.fs

        # Mirroring the signal to extend
        T = save_T

        # First half: first half reverse order, original signal in the middle, second half reverse order
        f_mirror = np.concatenate(
            (
                signal[: T // 2][::-1],  # First T/2 elements (in reverse order)
                signal,  # original signal
                signal[T // 2 :][::-1],  # The last T/2 elements (in reverse order)
            )
        )
        f = f_mirror.copy()

        # Time Domain (t -->> 0 to T)
        T = f.shape[0]
        t = np.arange(1, T + 1) / T

        half_T = T // 2

        # update step
        uDiff = self.tol + np.spacing(1)

        # Discretization of spectral domain
        omega_axis = t - 0.5 - 1 / T

        # FFT of signal(and Hilbert transform concept=making it one-sided)
        f_hat = fftshift(ts=fft(ts=f))
        f_hat_onesided = f_hat
        f_hat_onesided[: half_T] = 0

        # Initializing omega_d
        omega_d = np.zeros(self.max_iter)
        omega_d[0] = self.omega_init / fs

        # dual variables vector
        Lambda = np.zeros(shape=[self.max_iter, len(omega_axis)])

        # keeping changes of mode spectrum
        u_hat_d = np.zeros(shape=[self.max_iter, len(omega_axis)])

        # keeping changes of residual spectrum
        # F_r = np.zeros(shape=[self.max_iter, len(omega_axis)])

        # main loop counter
        n = 0

        # ----- Part 2: Main loop for iterative updates
        while uDiff > self.tol and n < self.max_iter - 1:
            # update u_d
            u_hat_d[n + 1, :] = (f_hat_onesided + (u_hat_d[n, :] * (self.alpha ** 2) * (omega_axis - omega_d[n]) ** 4) + Lambda[n, :] / 2) / ((1 + (self.alpha ** 2) * (omega_axis - omega_d[n]) ** 4) * (1 + 2 * self.alpha ** 2 * (omega_axis - omega_d[n]) ** 4))

            # update omega_d
            omega_d[n + 1] = (omega_axis[half_T : T] * (np.abs(u_hat_d[n + 1, half_T : T]) ** 2).T) / np.sum(np.abs(u_hat_d[n + 1, half_T : T]) ** 2, axis=0)

            # update lambda (dual ascent) ===> lambda = lambda + tau*(f(t)-(Ud+Fr))
            Lambda[n + 1, :] = Lambda[n, :] + (self.tau * (f_hat_onesided - (u_hat_d[n + 1, :] + ((self.alpha ** 2 * (omega_axis - omega_d[n + 1]) ** 4) * (f_hat_onesided - (u_hat_d[n + 1, :]))) / (1 + 2 * self.alpha ** 2 * (omega_axis - omega_d[n + 1]) ** 4))))

            # add the main loop counter
            n += 1

            uDiff = np.spacing(1)

            # 1st loop criterion
            uDiff = uDiff + 1 / T * np.matmul((u_hat_d[n, :] - u_hat_d[n - 1, :]), np.conj((u_hat_d[n, :] - u_hat_d[n - 1, :])).T)
            uDiff = np.abs(uDiff)

        # ----- Part 3: Signal Reconstruction
        N = min(self.max_iter, n)
        omega = omega_d[T]

        u_hat = np.zeros(T)
        u_hat[half_T : T] = np.squeeze(u_hat_d[N, half_T : T])
        conj_values = np.squeeze(np.conj(u_hat_d[N, half_T : T]))
        u_hat[1 : half_T + 1] = conj_values[::-1]
        u_hat[0, :] = np.conj(u_hat[-1, :])

        u_d = np.zeros(T)
        u_d[:] = np.real(ifftshift(ts=ifft(ts=u_hat[:, 0])))

        # Remove mirror part
        u_d = u_d[:, T // 4: 3 * T // 4]

        u_hat = fftshift(ts=fft(u_d)).T




if __name__ == '__main__':
    a = np.array([1, 2, 3, 4, 5])

    print(a[::-1])

