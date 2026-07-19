# -*- coding: utf-8 -*-
"""
Created on 2025/02/05 13:31:52
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Jump Plus AM-FM Mode Decomposition (JMD).

Faithful port of the authors' MATLAB ``JMD.m``.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy import linalg
import scipy.sparse as sp

from pysdkit.utils import fft, fftshift, ifft, ifftshift


class JMD(object):
    """
    Jump Plus AM-FM Mode Decomposition.

    Decomposes a nonstationary signal into AM–FM oscillatory modes and a
    discontinuous jump component by jointly solving a VMD-like bandwidth
    minimization and a jump prior (JOT-style ADMM).

    Mojtaba Nazari, Anders Rosendal Korshøj and Naveed ur Rehman,
    "Jump Plus AM-FM Mode Decomposition," IEEE TSP,
    https://doi.org/10.48550/arXiv.2407.07800
    """

    def __init__(
        self,
        K: int,
        alpha: float = 5000.0,
        init: str = "zero",
        tol: float = 1e-6,
        beta: float = 0.03,
        b_bar: float = 0.45,
        tau: float = 5.0,
        max_iter: int = 2000,
    ) -> None:
        """
        :param K: number of AM–FM modes to recover
        :param alpha: balancing parameter of the mode bandwidth
        :param init: ``'zero'`` / ``'uniform'`` / ``'random'`` omega initialization
        :param tol: convergence tolerance
        :param beta: jump-constraint weight (≈ 1 / expected number of jumps)
        :param b_bar: parameter related to the jump prior
        :param tau: jump ADMM step (used to form ``gamma``; set carefully for noise)
        :param max_iter: maximum number of iterations
        """
        self.K = int(K)
        self.alpha = float(alpha)
        self.init = str(init).lower()
        self.tol = float(tol)
        self.beta = float(beta)
        self.b_bar = float(b_bar)
        self.tau = float(tau)
        self.max_iter = int(max_iter)

        self.u: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None

    def __call__(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        return self.fit_transform(signal=signal, return_all=return_all)

    def __str__(self) -> str:
        return "Jump Plus AM-FM Mode Decomposition (JMD)"

    def jump_step(self, freqs: np.ndarray, T: int) -> Tuple[
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        float,
        float,
    ]:
        """Initialize jump-ADMM variables (matches MATLAB jump block)."""
        b = 2.0 / (self.b_bar**2)
        gamma = self.tau * (0.5 * b * self.beta)
        v = np.zeros(T, dtype=float)
        d = np.ones(T)
        D = sp.diags([-d, d], [0, 1], shape=(T, T), format="csr").tolil()
        D[-1, :] = 0
        D = D.tocsr()
        DTD = (D.T @ D).toarray()
        x = np.zeros(T, dtype=float)
        rho = np.zeros(T, dtype=float)
        coef1 = 1.0 / gamma
        mu = 2.0 * self.beta / gamma
        SPDiag = sp.eye(T, format="csr")
        j_hat_plus = np.zeros((self.max_iter, len(freqs)), dtype=complex)
        return (
            b,
            v,
            x,
            D.toarray(),
            DTD,
            SPDiag.toarray(),
            j_hat_plus,
            rho,
            coef1,
            mu,
            gamma,
        )

    @staticmethod
    def enc_fmirror(signal: np.ndarray) -> np.ndarray:
        """Mirror extension matching ``JMD.m`` / VMD style."""
        signal = np.asarray(signal, dtype=float).ravel()
        t = signal.size
        half = t // 2
        return np.concatenate([signal[half - 1 :: -1], signal, signal[: half - 1 : -1]])

    @staticmethod
    def dec_fmirror(u: np.ndarray, T: int) -> np.ndarray:
        """Remove mirrored flanks."""
        return u[:, T // 4 : 3 * T // 4].copy()

    @staticmethod
    def _soft_jump(h: np.ndarray, mu: float, b: float) -> np.ndarray:
        """Element-wise jump proximal map from MATLAB ``min(max(...),1).*h``."""
        abs_h = np.abs(h) + np.finfo(float).eps
        scale = (1.0 / (1.0 - mu * b)) - (
            mu * np.sqrt(2.0 * b) / (1.0 - mu * b)
        ) / abs_h
        scale = np.clip(scale, 0.0, 1.0)
        return scale * h

    def _init_omega(self, fs: float) -> np.ndarray:
        """Initialization of omega_k (MATLAB ``init`` cases 0/1/2)."""
        omega_plus = np.zeros((self.max_iter, self.K), dtype=float)
        if self.init == "zero":
            return omega_plus
        if self.init == "uniform":
            for i in range(self.K):
                omega_plus[0, i] = (0.5 / self.K) * i
            return omega_plus
        if self.init == "random":
            omega_plus[0, :] = np.sort(
                np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(self.K))
            )
            return omega_plus
        raise ValueError("init must be 'zero', 'uniform', or 'random'")

    def fit_transform(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Run JMD on a univariate signal.

        :param signal: 1-D time-domain signal
        :param return_all: if True, return ``(modes, jump, omega)``
        :return: modes ``(K, N)`` or ``(modes, jump, omega)``
        """
        signal = np.asarray(signal, dtype=float).ravel()
        if signal.size % 2 == 1:
            signal = signal[:-1]
        if signal.size < 4:
            raise ValueError("signal length must be at least 4 samples")
        if self.K < 1:
            raise ValueError("K must be >= 1")

        # MATLAB: demean, restore mean on the jump at the end
        shift = float(np.mean(signal))
        signal = signal - shift

        f = self.enc_fmirror(signal)
        T = f.size
        half_T = T // 2
        t = np.arange(1, T + 1, dtype=float) / T
        freqs = t - 0.5 - 1.0 / T
        fs = 1.0 / signal.size

        a2 = 50.0
        t2 = np.arange(0.01, np.sqrt(2.0 / a2) + 1e-12, 0.001)
        phi1 = (-a2 / 2.0) * (t2**2) + (np.sqrt(2.0 * a2) * t2)
        if self.max_iter > phi1.size:
            phi = np.concatenate([phi1, np.ones(self.max_iter - phi1.size)])
        else:
            phi = phi1[: self.max_iter].copy()
        Alpha = self.alpha * phi

        f_hat = fftshift(fft(f))
        f_hat_plus = f_hat.copy()
        f_hat_plus[:half_T] = 0.0

        u_hat_plus = np.zeros((self.max_iter, freqs.size, self.K), dtype=complex)
        omega_plus = self._init_omega(fs=fs)

        u_diff = self.tol + np.spacing(1)
        n = 0
        sum_uk = 0.0

        b, v, x, D, DTD, SPDiag, j_hat_plus, rho, coef1, mu, gamma = self.jump_step(
            freqs=freqs, T=T
        )
        A_jump = SPDiag + gamma * DTD

        while u_diff > self.tol and n < self.max_iter - 1:
            sum_uk = u_hat_plus[n, :, self.K - 1] + sum_uk - u_hat_plus[n, :, 0]

            for k in range(self.K):
                if k > 0:
                    sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
                u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - j_hat_plus[n, :]) / (
                    1.0 + Alpha[n] * (freqs - omega_plus[n, k]) ** 2
                )

                num = np.dot(
                    freqs[half_T:T], np.abs(u_hat_plus[n + 1, half_T:T, k]) ** 2
                )
                den = np.sum(np.abs(u_hat_plus[n + 1, half_T:T, k]) ** 2)
                omega_plus[n + 1, k] = num / (den + np.finfo(float).eps)

            # Back to time domain for jump update
            u_hat = np.zeros((T, self.K), dtype=complex)
            for k in range(self.K):
                u_hat[half_T:T, k] = u_hat_plus[n + 1, half_T:T, k]
                u_hat[half_T:0:-1, k] = np.conj(u_hat_plus[n + 1, half_T:T, k])
                u_hat[0, k] = np.conj(u_hat[-1, k])

            u = np.zeros((self.K, T), dtype=float)
            for k in range(self.K):
                u[k, :] = np.real(ifft(ifftshift(u_hat[:, k])))

            rhs = gamma * (D.T @ x) - (D.T @ rho) + f - np.sum(u, axis=0)
            v = linalg.solve(A_jump, rhs)

            dv = D @ v
            h = dv + coef1 * rho
            x = self._soft_jump(h, mu=mu, b=b)
            rho = rho - gamma * (x - dv)

            v = v - (np.mean(v) - np.mean(f))

            j_hat_plus[n + 1, :] = fftshift(fft(v))
            j_hat_plus[n + 1, :half_T] = 0.0

            n += 1
            u_diff = np.spacing(1)
            for i in range(self.K):
                diff = u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]
                u_diff += (1.0 / T) * np.vdot(diff, diff).real
            jdiff = j_hat_plus[n, :] - j_hat_plus[n - 1, :]
            u_diff += (1.0 / T) * np.vdot(jdiff, jdiff).real
            u_diff = abs(u_diff)

        n_keep = min(n, self.max_iter - 1)
        omega_hist = omega_plus[: n_keep + 1, :]

        u_hat = np.zeros((T, self.K), dtype=complex)
        for k in range(self.K):
            u_hat[half_T:T, k] = u_hat_plus[n_keep, half_T:T, k]
            u_hat[half_T:0:-1, k] = np.conj(u_hat_plus[n_keep, half_T:T, k])
            u_hat[0, k] = np.conj(u_hat[-1, k])

        u = np.zeros((self.K, T), dtype=float)
        for k in range(self.K):
            u[k, :] = np.real(ifft(ifftshift(u_hat[:, k])))

        u = self.dec_fmirror(u, T=T)
        v = v[T // 4 : 3 * T // 4] + shift

        # Sort modes by ascending center frequency (MATLAB cleanup)
        order = np.argsort(omega_hist[-1, :])
        u = u[order, :]
        omega = omega_hist[:, order]

        self.u = u
        self.v = v
        self.omega = omega

        if return_all:
            return u, v, omega[-1]
        return u
