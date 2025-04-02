# -*- coding: utf-8 -*-
"""
Created on 2025/04/02 22:44:18
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Tuple


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

    def __init__(self, alpha: Optional[float] = 20000, omega_init: Optional[float] = 0.0, fs: Optional[int] = None, tau: Optional[float] = 0, tol: Optional[float] = 1e-7) -> None:
        """
        :param alpha: compactness of mode constraint
        :param omega_init: initial guess of mode center-frequency (Hz)
        :param fs: the sampling frequency of input signal
        :param tau: time-step of the dual ascent. set it to 0 in the presence of high level of noise.
        :param tol: tolerance of convergence criterion; typically around 1e-6
        """
        self.alpha = alpha
        self.omega_init = omega_init
        self.fs = fs
        self.tau = tau
        self.tol = tol

    def __call__(self, *args, **kwargs):
        """allow instances to be called like functions"""
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Variational Mode Extraction (VME)"

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Signal decomposition using VME algorithm

        :param signal: the time domain signal (1D numpy array) to be decomposed
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

        # update step
        uDiff = self.tol + np.spacing(1)

        # Discretization of spectral domain








if __name__ == '__main__':
    a = np.array([1, 2, 3, 4, 5])

    print(a[::-1])

