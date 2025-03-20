# -*- coding: utf-8 -*-
"""
Created on Sat Mar 4 11:59:21 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from typing import Tuple
from pysdkit.utils import fft, ifft, fftshift, ifftshift, fmirror


def vmd(
    signal: np.array,
    alpha: int,
    K: int,
    tau: float,
    init: str = "uniform",
    DC: bool = False,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Variational mode decomposition, object-oriented interface.
    Original paper: Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’,
    IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.

    :param signal: the time domain signal (1D numpy array)  to be decomposed
    :param alpha: the balancing parameter of the data-fidelity constraint
    :param K: the number of modes to be recovered
    :param tau: time-step of the dual ascent ( pick 0 for noise-slack )
    :param init: uniform = all omegas start uniformly distributed
                 zero = all omegas initialized randomly
                 random = all omegas start at 0
    :param DC: true if the first mode is put and kept at DC (0-freq)
    :param max_iter: Maximum number of iterations
    :param tol: tolerance of convergence criterion; typically around 1e-6

    :return: - u - the collection of decomposed modes,
             - u_hat   - spectra of the modes,
             - omega   - estimated mode center-frequencies
    """

    if len(signal) % 2:
        signal = signal[:-1]

    # Period and sampling frequency of input signal
    fs = 1.0 / len(signal)

    # Mirror expansion of signals
    sym = len(signal) // 2
    fMirr = fmirror(ts=signal, sym=sym)
    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1, T + 1) / T

    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1, T + 1) / T

    # Spectral Domain
    freqs = t - 0.5 - (1 / T)

    # Construct and center f_hat
    f_hat = fftshift(ts=fft(ts=fMirr))
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[: T // 2] = 0

    # For future generalizations: individual alpha for each mode
    alpha = np.ones(K) * alpha
    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros([max_iter, len(freqs), K], dtype=complex)
    # Initialization of omega_k
    omega_plus = np.zeros([max_iter, K])
    if init.lower() == "uniform":
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * i
    elif init.lower() == "random":
        omega_plus[0, :] = np.sort(
            np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1, K))
        )
    elif init.lower() == "zero":
        omega_plus[0, :] = 0.0
    else:
        raise ValueError
    if DC:
        omega_plus[0, 0] = 0
    # start with empty dual variables
    lambda_hat = np.zeros(shape=[max_iter, len(freqs)], dtype=complex)

    sum_uk = 0  # accumulator
    convergence = np.spacing(1) + tol  # Determine whether the algorithm converges

    # Main loop for iterative updates
    for n in range(0, max_iter - 1):
        # update spectrum of first mode through Wiener filter of residuals
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]
        u_hat_plus[n + 1, :, 0] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
            1.0 + alpha[0] * (freqs - omega_plus[n, 0]) ** 2
        )

        # update first omega if not held at 0
        if not DC:
            omega_plus[n + 1, 0] = np.dot(
                freqs[T // 2 : T], (abs(u_hat_plus[n + 1, T // 2 : T, 0]) ** 2)
            ) / np.sum(abs(u_hat_plus[n + 1, T // 2 : T, 0]) ** 2)

        # update of any other mode
        for k in range(1, K):
            # mode spectrum
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
            u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                1 + alpha[k] * (freqs - omega_plus[n, k]) ** 2
            )
            # center frequencies
            omega_plus[n + 1, k] = np.dot(
                freqs[T // 2 : T], (abs(u_hat_plus[n + 1, T // 2 : T, k]) ** 2)
            ) / np.sum(abs(u_hat_plus[n + 1, T // 2 : T, k]) ** 2)

        # Update Lagrange multipliers
        lambda_hat[n + 1, :] = lambda_hat[n, :] + tau * (
            np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus
        )

        # Determine whether the algorithm has converged
        for i in range(K):
            convergence = convergence + (1 / T) * np.dot(
                (u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]),
                np.conj((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i])),
            )
        convergence = np.abs(convergence)
        if convergence <= tol:
            break

    # discard empty space if converged early
    niter = np.min([max_iter, n])
    omega = omega_plus[:niter, :]
    idxs = np.flip(np.arange(1, T // 2 + 1), axis=0)

    # signal reconstruction
    u_hat = np.zeros([T, K], dtype=complex)
    u_hat[T // 2 : T, :] = u_hat_plus[niter - 1, T // 2 : T, :]
    u_hat[idxs, :] = np.conj(u_hat_plus[niter - 1, T // 2 : T, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])

    u = np.zeros([K, len(t)])
    for k in range(K):
        u[k, :] = np.real(ifft(ts=ifftshift(ts=u_hat[:, k])))

    # remove mirror part
    u = u[:, T // 4 : 3 * T // 4]

    # recompute spectrum
    u_hat = np.zeros([u.shape[1], K], dtype=complex)
    for k in range(K):
        u_hat[:, k] = fftshift(ts=fft(ts=u[k, :]))

    return u, u_hat, omega
