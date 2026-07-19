# -*- coding: utf-8 -*-
"""
Created on 2025/07/20
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Orthogonalized Variational Mode Decomposition (OVMD).

Marbona H., Rodríguez D., Martínez-Cava A., Valero E.,
Orthogonalized Variational Mode Decomposition,
Signal Processing, 239:110251, 2026.
https://doi.org/10.1016/j.sigpro.2025.110251
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

from .base import Base


class OVMD(Base):
    """
    Orthogonalized Variational Mode Decomposition.

    OVMD extends VMD by adding weak orthogonality objectives that transfer
    correlated spectral content from weaker modes to dominant modes when
    their frequency bands overlap. Combined with a proportional filter
    bandwidth ``alpha_k = alpha / omega_k``, this reduces mode duplication
    under over-segmentation.

    Marbona et al., Signal Processing, 239:110251, 2026.
    """

    def __init__(
        self,
        alpha: float = 100.0,
        K: int = 5,
        tau: float = 0.0,
        tol: float = 1e-5,
        max_iter: int = 1000,
    ) -> None:
        """
        :param alpha: balancing / bandwidth parameter (``2F`` in the paper)
        :param K: number of modes to recover
        :param tau: dual-ascent step size (0 for noise-slack)
        :param tol: relative spectral update tolerance
        :param max_iter: maximum number of ADMM iterations
        """
        super().__init__()
        self.alpha = float(alpha)
        self.K = int(K)
        self.tau = float(tau)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

        self.signal: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.u_hat: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None
        self.n_iter: Optional[int] = None

    def __call__(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self.fit_transform(signal=signal, return_all=return_all)

    def __str__(self) -> str:
        return "Orthogonalized Variational Mode Decomposition (OVMD)"

    @staticmethod
    def _mirror_extend(signal: np.ndarray) -> np.ndarray:
        """Mirror extension matching the authors' ``OVMD.m``."""
        signal = np.asarray(signal, dtype=float).ravel()
        t = signal.size
        t2 = t // 2
        return np.concatenate([signal[t2 - 1 :: -1], signal, signal[: t2 - 1 : -1]])

    def fit_transform(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Decompose a 1-D signal with OVMD.

        :param signal: time-domain signal
        :param return_all: if True, also return spectra and center frequencies
        :return: modes ``(K, N)``, optionally ``(u, u_hat, omega)``
        """
        signal = np.asarray(signal, dtype=float).ravel()
        if signal.size < 4:
            raise ValueError("signal length must be at least 4 samples")
        if self.K < 1:
            raise ValueError("K must be >= 1")

        # Keep an even length for clean FFT / mirror bookkeeping.
        if signal.size % 2 == 1:
            signal = signal[:-1]

        f = self._mirror_extend(signal)
        t_len = f.size
        t = np.arange(1, t_len + 1, dtype=float) / t_len
        freqs = t - 0.5 - 1.0 / t_len

        alpha_vec = self.alpha * np.ones(self.K, dtype=float)

        f_hat = self.fftshift(self.fft(f))
        f_hat_plus = f_hat.copy()
        f_hat_plus[: t_len // 2] = 0.0

        # Two-row ping-pong buffers (previous / current), as in OVMD.m
        u_hat_plus = np.full((2, t_len, self.K), np.finfo(float).eps, dtype=complex)
        max_uhat = np.full(self.K, np.finfo(float).eps, dtype=float)
        norm_uhat = np.full(self.K, np.finfo(float).eps, dtype=float)

        omega_hist = np.zeros((self.max_iter, self.K), dtype=float)
        omega_filter = np.ones((self.K, t_len), dtype=float)
        for k in range(self.K):
            norm_uhat[k] = np.linalg.norm(u_hat_plus[1, :, k])
            max_uhat[k] = np.max(np.abs(u_hat_plus[1, :, k]))
            omega_filter[k, :] = 1.0 / (
                1.0 + alpha_vec[k] * (np.abs(freqs) - omega_hist[0, k]) ** 2
            )

        lambda_hat = np.zeros(t_len, dtype=complex)
        sum_uk = np.zeros(t_len, dtype=complex)
        u_diff = 1.0
        m = 0  # 0-based iteration counter (MATLAB m starts at 1)

        half = t_len // 2

        while u_diff > self.tol and m < self.max_iter - 1:
            for k in range(self.K):
                receive_cont = np.zeros(t_len, dtype=complex)
                send_cont = np.zeros(t_len, dtype=float)
                o_add = np.zeros(t_len, dtype=complex)
                o_subs = np.zeros(t_len, dtype=float)

                uk = u_hat_plus[1, :, k]
                denom_kk = norm_uhat[k] * norm_uhat[k] + np.finfo(float).eps
                proj_kk = np.sqrt(np.abs(uk * np.conj(uk)) / denom_kk)

                for i in range(self.K):
                    if i == k:
                        continue
                    ui = u_hat_plus[1, :, i]
                    denom_ik = norm_uhat[i] * norm_uhat[k] + np.finfo(float).eps
                    proj_ik = np.sqrt(np.abs(ui * np.conj(uk)) / denom_ik)
                    if max_uhat[k] > max_uhat[i]:
                        # mode k dominant: receive correlated content from i
                        o_add = o_add + proj_ik * ui
                        o_subs = o_subs + (proj_ik**2) * (np.abs(ui) ** 2)
                        receive_cont = receive_cont + (proj_ik * proj_kk) * ui
                    else:
                        # mode k weaker: send correlated content toward i
                        send_cont = send_cont + omega_filter[i, :] * (proj_ik**2)

                receive_cont = omega_filter[k, :] * receive_cont
                o_kk = proj_kk * uk
                o_subs = (omega_filter[k, :] ** 2) * (
                    np.abs(o_add + o_kk) ** 2
                    - (o_subs + (proj_kk**2) * (np.abs(uk) ** 2))
                )

                # residual accumulator (classic VMD ADMM ordering)
                k_minus = self.K - 1 if k == 0 else k - 1
                sum_uk = u_hat_plus[1, :, k_minus] + sum_uk - u_hat_plus[0, :, k]

                denom = 1.0 + alpha_vec[k] * (freqs - omega_hist[m, k]) ** 2 + send_cont
                u_hat_plus[1, :, k] = (
                    f_hat_plus - sum_uk - lambda_hat / 2.0 + receive_cont
                ) / denom

                norm_uhat[k] = np.linalg.norm(u_hat_plus[1, :, k])
                max_uhat[k] = float(np.max(np.abs(u_hat_plus[1, :, k])))

                # center-frequency update with orthogonality correction
                pos = slice(half, t_len)
                num = np.dot(
                    freqs[pos],
                    np.abs(u_hat_plus[1, pos, k]) ** 2 + o_subs[pos],
                )
                den = np.sum(np.abs(u_hat_plus[1, pos, k]) ** 2 + o_subs[pos])
                omega_hist[m + 1, k] = num / (den + np.finfo(float).eps)
                omega_filter[k, :] = 1.0 / (
                    1.0 + alpha_vec[k] * (np.abs(freqs) - omega_hist[m + 1, k]) ** 2
                )

            # proportional bandwidth + convergence measure
            u_diff = np.finfo(float).eps
            for i in range(self.K):
                alpha_vec[i] = self.alpha / max(
                    omega_hist[m + 1, i], np.finfo(float).eps
                )
                omega_filter[i, :] = 1.0 / (
                    1.0 + alpha_vec[i] * (np.abs(freqs) - omega_hist[m + 1, i]) ** 2
                )
                cur = u_hat_plus[1, :, i]
                prev = u_hat_plus[0, :, i]
                u_diff += (np.linalg.norm(cur - prev) ** 2) / (
                    np.linalg.norm(cur) ** 2 + np.finfo(float).eps
                )

            lambda_hat = lambda_hat + self.tau * (
                np.sum(u_hat_plus[1, :, :], axis=1) - f_hat_plus
            )
            u_hat_plus[0, :, :] = u_hat_plus[1, :, :]
            m += 1
            u_diff = abs(u_diff)

        n_iter = min(self.max_iter, m + 1)
        omega = omega_hist[:n_iter, :]

        # reconstruct two-sided spectra
        u_hat = np.zeros((t_len, self.K), dtype=complex)
        u_hat[half:t_len, :] = u_hat_plus[1, half:t_len, :]
        u_hat[half:0:-1, :] = np.conj(u_hat_plus[1, half:t_len, :])
        u_hat[0, :] = np.conj(u_hat[-1, :])

        # sort modes by spectral energy (high -> low)
        order = np.argsort(np.sum(np.abs(u_hat) ** 2, axis=0))[::-1]
        omega = omega[:, order]
        u_hat = u_hat[:, order]

        u = np.zeros((self.K, t_len), dtype=float)
        for k in range(self.K):
            u[k, :] = np.real(self.ifft(self.ifftshift(u_hat[:, k])))

        # remove mirrored flanks
        u = u[:, t_len // 4 : 3 * t_len // 4]

        u_hat_out = np.zeros((u.shape[1], self.K), dtype=complex)
        for k in range(self.K):
            u_hat_out[:, k] = self.fftshift(self.fft(u[k, :]))

        self.signal = signal
        self.u = u
        self.u_hat = u_hat_out
        self.omega = omega
        self.n_iter = n_iter

        if return_all:
            return u, u_hat_out, omega
        return u
