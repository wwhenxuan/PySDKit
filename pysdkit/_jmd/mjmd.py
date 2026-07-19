# -*- coding: utf-8 -*-
"""
Created on 2025/02/05 13:32:06
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Multivariate Jump Plus AM-FM Mode Decomposition (MJMD).

Faithful port of the authors' MATLAB ``MJMD.m``.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy import linalg
import scipy.sparse as sp


class MJMD(object):
    """
    Multivariate Jump Plus AM-FM Mode Decomposition.

    Extends JMD to multichannel data: oscillatory modes share common
    center frequencies across channels (MVMD-style), while each channel
    has its own jump component estimated by a JOT-like ADMM prior.

    Mojtaba Nazari, Anders Rosendal Korshøj and Naveed ur Rehman,
    "Jump Plus AM-FM Mode Decomposition," IEEE TSP,
    https://doi.org/10.48550/arXiv.2407.07800
    """

    def __init__(
        self,
        K: int,
        alpha: float = 5000.0,
        init: str = "zero",
        tol: float = 1e-5,
        beta: float = 0.05,
        b_bar: float = 0.45,
        tau: float = 5.0,
        DC: bool = False,
        max_iter: int = 2000,
    ) -> None:
        """
        :param K: number of multivariate AM–FM modes
        :param alpha: bandwidth balancing parameter
        :param init: ``'zero'`` / ``'uniform'`` / ``'random'``
        :param tol: convergence tolerance
        :param beta: jump-constraint weight
        :param b_bar: jump-prior parameter
        :param tau: jump ADMM step used to form ``gamma``
        :param DC: if True, keep the first mode at zero frequency
        :param max_iter: maximum iterations
        """
        self.K = int(K)
        self.alpha = float(alpha)
        self.init = str(init).lower()
        self.tol = float(tol)
        self.beta = float(beta)
        self.b_bar = float(b_bar)
        self.tau = float(tau)
        self.DC = bool(DC)
        self.max_iter = int(max_iter)

        self.u: Optional[np.ndarray] = None
        self.jump: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None

    def __call__(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        return self.fit_transform(signal=signal, return_all=return_all)

    def __str__(self) -> str:
        return "Multivariate Jump Plus AM-FM Mode Decomposition (MJMD)"

    @staticmethod
    def _resolve_shape(signal: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Accept ``(C, N)`` or ``(N, C)`` (rows > cols ⇒ channels along columns,
        matching MATLAB ``MJMD.m``).
        Returns demeaned data as ``(C, N)``.
        """
        arr = np.asarray(signal, dtype=float)
        if arr.ndim != 2:
            raise ValueError("MJMD expects a 2-D multivariate signal")
        s1, s2 = arr.shape
        if s1 > s2:
            # MATLAB: rows = length, cols = channels
            data = arr.T.copy()
        else:
            data = arr.copy()
        c, n = data.shape
        if n % 2 == 1:
            data = data[:, :-1]
            n = data.shape[1]
        if n < 4:
            raise ValueError("signal length must be at least 4 samples")
        return data, c, n

    @staticmethod
    def _mirror_multi(signal: np.ndarray) -> np.ndarray:
        """Channel-wise mirror extension, shape ``(C, 2N)``."""
        c, n = signal.shape
        half = n // 2
        out = np.zeros((c, 2 * n), dtype=float)
        out[:, :half] = signal[:, half - 1 :: -1]
        out[:, half : half + n] = signal
        out[:, half + n :] = signal[:, : half - 1 : -1]
        return out

    @staticmethod
    def _soft_jump(h: np.ndarray, mu: float, b: float) -> np.ndarray:
        abs_h = np.abs(h) + np.finfo(float).eps
        scale = (1.0 / (1.0 - mu * b)) - (
            mu * np.sqrt(2.0 * b) / (1.0 - mu * b)
        ) / abs_h
        scale = np.clip(scale, 0.0, 1.0)
        return scale * h

    def _init_omega(self, fs: float) -> np.ndarray:
        omega_plus = np.zeros((self.max_iter, self.K), dtype=float)
        if self.init == "uniform":
            omega_plus[0, :] = (0.5 / self.K) * np.arange(self.K)
        elif self.init == "random":
            omega_plus[0, :] = np.sort(
                np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(self.K))
            )
        else:
            omega_plus[0, :] = 0.0
        if self.DC:
            omega_plus[0, 0] = 0.0
        return omega_plus

    def fit_transform(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Run MJMD on a multivariate signal.

        :param signal: ``(C, N)`` preferred, or ``(N, C)`` if ``N > C``
        :param return_all: if True, return ``(modes, jump, omega)``
        :return: modes ``(K, N, C)``, optionally jump ``(C, N)`` and omega ``(K,)``
        """
        data, n_ch, n_samp = self._resolve_shape(signal)
        if self.K < 1:
            raise ValueError("K must be >= 1")

        shift = data.mean(axis=1, keepdims=True)
        data = data - shift

        f1 = self._mirror_multi(data)  # (C, 2N)
        t_len = f1.shape[1]
        half = t_len // 2
        t = np.arange(1, t_len + 1, dtype=float) / t_len
        freqs = t - 0.5 - 1.0 / t_len
        fs = 1.0 / n_samp

        a2 = 50.0
        t2 = np.arange(0.01, np.sqrt(2.0 / a2) + 1e-12, 0.001)
        phi1 = (-a2 / 2.0) * (t2**2) + (np.sqrt(2.0 * a2) * t2)
        if self.max_iter > phi1.size:
            phi = np.concatenate([phi1, np.ones(self.max_iter - phi1.size)])
        else:
            phi = phi1[: self.max_iter].copy()
        Alpha = self.alpha * phi

        # FFT along time for each channel → (C, T)
        f_hat = np.fft.fftshift(np.fft.fft(f1, axis=1), axes=1)
        f_hat_plus = f_hat.copy()
        f_hat_plus[:, :half] = 0.0

        # Mode spectra: (T, C, K)
        u_hat_plus_00 = np.zeros((t_len, n_ch, self.K), dtype=complex)
        u_hat_plus = np.zeros((t_len, n_ch, self.K), dtype=complex)
        omega_plus = self._init_omega(fs=fs)

        u_diff = self.tol + np.spacing(1)
        n = 0
        sum_uk = np.zeros((t_len, n_ch), dtype=complex)

        # Jump ADMM setup
        b = 2.0 / (self.b_bar**2)
        gamma = self.tau * (0.5 * b * self.beta)
        d = np.ones(t_len)
        D = sp.diags([-d, d], [0, 1], shape=(t_len, t_len), format="csr").tolil()
        D[-1, :] = 0
        D = D.tocsr()
        D_arr = D.toarray()
        DTD = (D.T @ D).toarray()
        x = np.zeros((t_len, n_ch), dtype=float)
        rho = np.zeros((t_len, n_ch), dtype=float)
        coef1 = 1.0 / gamma
        mu = 2.0 * self.beta / gamma
        SPDiag = np.eye(t_len)
        A_jump = SPDiag + gamma * DTD
        j_hat_plus = np.zeros((t_len, n_ch, self.max_iter), dtype=complex)
        jump = np.zeros((n_ch, t_len), dtype=float)

        while u_diff > self.tol and n < self.max_iter - 1:
            for k in range(self.K):
                if k > 0:
                    sum_uk = u_hat_plus[:, :, k - 1] + sum_uk - u_hat_plus_00[:, :, k]
                else:
                    sum_uk = (
                        u_hat_plus_00[:, :, self.K - 1]
                        + sum_uk
                        - u_hat_plus_00[:, :, k]
                    )

                for c in range(n_ch):
                    u_hat_plus[:, c, k] = (
                        f_hat_plus[c, :].T - sum_uk[:, c] - j_hat_plus[:, c, n]
                    ) / (1.0 + Alpha[n] * (freqs - omega_plus[n, k]) ** 2)

                # Shared center frequency across channels
                num = freqs[half:t_len] @ (np.abs(u_hat_plus[half:t_len, :, k]) ** 2)
                den = np.sum(np.abs(u_hat_plus[half:t_len, :, k]) ** 2)
                omega_plus[n + 1, k] = np.sum(num) / (np.sum(den) + np.finfo(float).eps)

                # Encourage separation of neighboring omegas (MATLAB)
                while (
                    k > 0
                    and abs(omega_plus[n + 1, k] - omega_plus[n + 1, k - 1]) < 0.001
                ):
                    omega_plus[n + 1, k] = float(
                        np.exp(
                            np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand()
                        )
                    )

                if self.DC and k == 0:
                    omega_plus[n + 1, 0] = 0.0

            # Time-domain modes for jump update: u (T, C, K)
            u_hat = np.zeros((t_len, n_ch, self.K), dtype=complex)
            for k in range(self.K):
                u_hat[half:t_len, :, k] = u_hat_plus[half:t_len, :, k]
                u_hat[half:0:-1, :, k] = np.conj(u_hat_plus[half:t_len, :, k])
                u_hat[0, :, k] = np.conj(u_hat[-1, :, k])

            u_time = np.zeros((t_len, n_ch, self.K), dtype=float)
            for k in range(self.K):
                for c in range(n_ch):
                    u_time[:, c, k] = np.real(
                        np.fft.ifft(np.fft.ifftshift(u_hat[:, c, k]))
                    )

            for c in range(n_ch):
                temp = np.sum(u_time[:, c, :], axis=1)
                f_c = f1[c, :]
                rhs = gamma * (D_arr.T @ x[:, c]) - (D_arr.T @ rho[:, c]) + f_c - temp
                v = linalg.solve(A_jump, rhs)
                dv = D_arr @ v
                h = dv + coef1 * rho[:, c]
                x[:, c] = self._soft_jump(h, mu=mu, b=b)
                rho[:, c] = rho[:, c] - gamma * (x[:, c] - dv)
                v = v - (np.mean(v) - np.mean(f_c))
                jump[c, :] = v
                j_hat_plus[:, c, n + 1] = np.fft.fftshift(np.fft.fft(v))
                j_hat_plus[:half, c, n + 1] = 0.0

            n += 1
            u_hat_plus_m1 = u_hat_plus_00
            u_hat_plus_00 = u_hat_plus.copy()

            ud = u_hat_plus_00 - u_hat_plus_m1
            mode_term = np.sum((1.0 / t_len) * ud * np.conj(ud))
            jd = j_hat_plus[:, :, n] - j_hat_plus[:, :, n - 1]
            jump_term = np.sum((1.0 / t_len) * jd * np.conj(jd))
            u_diff = float(np.spacing(1) + np.abs(mode_term + jump_term))

        # Final reconstruction
        u_hat = np.zeros((t_len, n_ch, self.K), dtype=complex)
        u_hat[half:t_len, :, :] = u_hat_plus[half:t_len, :, :]
        u_hat[half:0:-1, :, :] = np.conj(u_hat_plus[half:t_len, :, :])
        u_hat[0, :, :] = np.conj(u_hat[-1, :, :])

        u_time = np.zeros((t_len, n_ch, self.K), dtype=float)
        for c in range(n_ch):
            for k in range(self.K):
                u_time[:, c, k] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, c, k])))

        omega = omega_plus[n, :].copy()
        order = np.argsort(omega)
        u_time = u_time[:, :, order]
        omega = omega[order]

        # Remove mirror: (N, C, K) → modes (K, N, C)
        u_crop = u_time[t_len // 4 : 3 * t_len // 4, :, :]
        jump_crop = jump[:, t_len // 4 : 3 * t_len // 4] + shift

        modes = np.transpose(u_crop, (2, 0, 1))  # (K, N, C)

        self.u = modes
        self.jump = jump_crop
        self.omega = omega

        if return_all:
            return modes, jump_crop, omega
        return modes
