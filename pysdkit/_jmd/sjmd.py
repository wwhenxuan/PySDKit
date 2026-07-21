# -*- coding: utf-8 -*-
"""
Created on 2025/07/20
@author: Rongkun Zhu
@email: 25481568@life.hkbu.edu.hk

Successive Jump and Mode Decomposition (SJMD / SMJMD).

Faithful port of the authors' MATLAB ``SJMD.m``.
One implementation covers both univariate (SJMD) and multivariate (SMJMD).
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp


class SJMD(object):
    """
    Successive Jump and Mode Decomposition (SJMD / SMJMD).

    Successively extracts AM–FM modes and a jump component without a
    predefined mode count ``K``. The same routine accepts:

    - univariate data ``(N,)`` or ``(1, N)`` → SJMD
    - multivariate data ``(C, N)`` (or ``(N, C)`` if ``N > C``) → SMJMD

    Nazari, Korshøj, ur Rehman, "Successive Jump and Mode Decomposition,"
    arXiv:2504.08453.
    """

    def __init__(
        self,
        max_alpha: float = 80000.0,
        tau: float = 50.0,
        beta: float = 0.5,
        b_bar: float = 0.9,
        stopc: int = 4,
        tol: float = 1e-5,
        kdm: Optional[int] = None,
        max_inner: int = 100,
        min_alpha: float = 10.0,
    ) -> None:
        """
        :param max_alpha: upper bound on the bandwidth parameter α
        :param tau: jump-ADMM step used to form ``gamma``
        :param beta: jump-constraint weight
        :param b_bar: expected minimal jump height
        :param stopc: stopping criterion
            1 exact reconstruction, 2 Bayesian, 3 k most dominant modes,
            4 power of the last mode (recommended / default)
        :param tol: inner-loop convergence tolerance
        :param kdm: number of dominant modes when ``stopc == 3``
        :param max_inner: max ADMM iterations per mode / alpha stage
        :param min_alpha: initial alpha for each successive mode
        """
        self.max_alpha = float(max_alpha)
        self.tau = float(tau)
        self.beta = float(beta)
        self.b_bar = float(b_bar)
        self.stopc = int(stopc)
        self.tol = float(tol)
        self.kdm = kdm
        self.max_inner = int(max_inner)
        self.min_alpha = float(min_alpha)

        self.u: Optional[np.ndarray] = None
        self.j: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None
        self.n_channels: Optional[int] = None

    def __call__(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return self.fit_transform(signal=signal, return_all=return_all)

    def __str__(self) -> str:
        return "Successive Jump and Mode Decomposition (SJMD / SMJMD)"

    @staticmethod
    def _resolve_input(signal: np.ndarray) -> Tuple[np.ndarray, int, int, bool]:
        """Return ``(data CxN, C, N, univariate)`` matching MATLAB layout."""
        arr = np.asarray(signal, dtype=float)
        if arr.ndim == 1:
            data = arr.reshape(1, -1)
            uni = True
        elif arr.ndim == 2:
            x, y = arr.shape
            if x > y:
                data = arr.T.copy()
                uni = data.shape[0] == 1
            else:
                data = arr.copy()
                uni = data.shape[0] == 1
        else:
            raise ValueError("SJMD expects 1-D or 2-D input")

        # MATLAB: if mod(2*save_T,2)~=0 → trim (always false for integer);
        # keep even-length guard used elsewhere in the toolbox.
        if data.shape[1] % 2 == 1:
            data = data[:, :-1]
        c, n = data.shape
        if n < 8:
            raise ValueError("signal length must be at least 8 samples")
        return data, c, n, uni

    @staticmethod
    def _mirror(signal: np.ndarray) -> np.ndarray:
        """Mirror extension as in MATLAB ``SJMD.m`` (overlap + append last)."""
        t = signal.shape[1]
        half = t // 2
        f_mir = np.zeros((signal.shape[0], 2 * t - 1), dtype=float)
        # f_mir(:,1:T/2) = signal(:,T/2:-1:1)
        f_mir[:, 0:half] = signal[:, half - 1 :: -1]
        # f_mir(:,T/2:3*T/2-1) = signal  (overwrites column T/2)
        f_mir[:, half - 1 : half - 1 + t] = signal
        # f_mir(:,3*T/2:2*T-1) = signal(:,T:-1:T/2+1)
        f_mir[:, 3 * half - 1 : 2 * t - 1] = signal[:, t - 1 : half - 1 : -1]
        # f_mir = [f_mir f_mir(:,end)]
        return np.concatenate([f_mir, f_mir[:, -1:]], axis=1)

    @staticmethod
    def _soft_jump(h: np.ndarray, mu: float, b: float) -> np.ndarray:
        abs_h = np.abs(h) + np.finfo(float).eps
        scale = (1.0 / (1.0 - mu * b)) - (
            mu * np.sqrt(2.0 * b) / (1.0 - mu * b)
        ) / abs_h
        return np.clip(scale, 0.0, 1.0) * h

    @staticmethod
    def _hermitian_from_onesided(
        onesided: np.ndarray, half: int, t_len: int
    ) -> np.ndarray:
        """Build full Hermitian spectrum from one-sided FFT bins."""
        out = np.zeros((t_len,) + onesided.shape[1:], dtype=complex)
        out[half:t_len, ...] = onesided[half:t_len, ...]
        out[half:0:-1, ...] = np.conj(onesided[half:t_len, ...])
        out[0, ...] = np.conj(out[-1, ...])
        return out

    def _check_stop(
        self,
        k: int,
        f_mir: np.ndarray,
        sum_u: np.ndarray,
        u_hat_i: np.ndarray,
        f_hat_temp: np.ndarray,
        polm: List[float],
        polm_temp: Optional[float],
        bic: List[float],
        t_len: int,
    ) -> Tuple[bool, List[float], Optional[float], List[float]]:
        """MATLAB Part 5 stopping criteria."""
        stop = False
        stopc = self.stopc

        if stopc == 1:
            numer = (1.0 / t_len) * (np.linalg.norm(f_mir - sum_u) ** 2)
            denom = (1.0 / t_len) * (np.linalg.norm(f_mir) ** 2) + np.finfo(float).eps
            if numer / denom <= 0.02:
                stop = True
        elif stopc == 2:
            if u_hat_i.shape[2] == 1:
                err = np.linalg.norm(f_hat_temp - u_hat_i[:, :, 0].T) ** 2
            else:
                err = np.linalg.norm(f_hat_temp - np.sum(u_hat_i, axis=2).T) ** 2
            bic.append(
                2 * t_len * np.log(err + np.finfo(float).eps)
                + (3 * k) * np.log(2 * t_len)
            )
            if k > 1 and bic[-1] > bic[-2]:
                stop = True
        elif stopc == 3:
            if self.kdm is None:
                raise ValueError("kdm must be set when stopc == 3")
            if k == self.kdm:
                stop = True
        else:
            # case 4 / default: power of the last mode
            power = float(
                np.mean(
                    np.abs(
                        np.sum(
                            (1.0 / t_len)
                            * u_hat_i[:, :, k - 1]
                            * np.conj(u_hat_i[:, :, k - 1]),
                            axis=0,
                        )
                    )
                )
            )
            if k < 2:
                polm_temp = power
                polm.append(1.0)
            else:
                polm.append(power / (polm_temp + np.finfo(float).eps))
                if abs(polm[-1] - polm[-2]) < 0.01:
                    stop = True

        return stop, polm, polm_temp, bic

    def fit_transform(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Run SJMD / SMJMD.

        :param signal: ``(N,)``, ``(1, N)``, or ``(C, N)``
        :param return_all: if True, return ``(modes, jump)``
        :return:
            - univariate: modes ``(K, N)``, jump ``(N,)``
            - multivariate: modes ``(K, N, C)``, jump ``(C, N)``
        """
        data, n_ch, save_t, univariate = self._resolve_input(signal)
        self.n_channels = n_ch

        shift = data.mean(axis=1, keepdims=True)
        data = data - shift

        f_mir = self._mirror(data)
        f = f_mir.copy()
        t_len = f.shape[1]
        half = t_len // 2
        t = np.arange(1, t_len + 1, dtype=float) / t_len
        omega_freqs = t - 0.5 - 1.0 / t_len

        f_hat = np.fft.fftshift(np.fft.fft(f, axis=1), axes=1)
        f_hat_onesided = f_hat.copy()
        f_hat_onesided[:, :half] = 0.0
        f_hat_temp = f_hat_onesided.copy()

        n_max = self.max_inner
        b = 2.0 / (self.b_bar**2)
        gamma = self.tau * (0.5 * b * self.beta)
        d = np.ones(t_len)
        D = sp.diags([-d, d], [0, 1], shape=(t_len, t_len), format="lil")
        D[-1, :] = 0
        D = D.tocsr()
        D_arr = D.toarray()
        DTD = (D.T @ D).tocsc()
        coef1 = 1.0 / gamma
        mu = 2.0 * self.beta / gamma
        A_jump = sp.eye(t_len, format="csc") + gamma * DTD
        solve_jump = sp.linalg.factorized(A_jump)

        x = np.zeros((t_len, n_ch), dtype=float)
        rho = np.zeros((t_len, n_ch), dtype=float)
        j_comp = np.zeros((n_ch, t_len), dtype=float)
        j_hat_plus = np.zeros((t_len, n_ch, n_max), dtype=complex)
        g_hat_plus = np.zeros((t_len, n_ch, n_max), dtype=complex)

        u_hat_temp_list: List[np.ndarray] = []
        omega_d_temp: List[float] = []
        u_hat_i = np.zeros((t_len, n_ch, 0), dtype=complex)

        sum_u = np.zeros((n_ch, t_len), dtype=float)
        sc2 = False
        k = 1
        polm: List[float] = []
        polm_temp: Optional[float] = None
        bic: List[float] = []

        while not sc2:
            # ---- Part 2/3: extract one mode while increasing alpha ----
            omega_l = np.zeros(n_max, dtype=float)
            u_hat_l = np.zeros((t_len, n_ch, n_max), dtype=complex)
            udiff = self.tol + np.spacing(1)
            n = 0  # 0-based; MATLAB starts at 1 and writes to n+1
            m = 0.0
            bf = 0
            alpha_val = self.min_alpha

            while alpha_val < (self.max_alpha + 1):
                while udiff > self.tol and n < n_max - 1:
                    for c in range(n_ch):
                        denom = 1.0 + alpha_val * (omega_freqs - omega_l[n]) ** 2
                        u_hat_l[:, c, n + 1] = (
                            f_hat_onesided[c, :]
                            - j_hat_plus[:, c, n]
                            - g_hat_plus[:, c, n]
                        ) / denom

                        num_g = (
                            (alpha_val**2)
                            * (omega_freqs - omega_l[n]) ** 4
                            * (
                                f_hat_onesided[c, :]
                                - j_hat_plus[:, c, n]
                                - u_hat_l[:, c, n + 1]
                            )
                        )
                        den_g = (
                            1.0 + 2.0 * (alpha_val**2) * (omega_freqs - omega_l[n]) ** 4
                        )
                        g_hat_plus[:, c, n + 1] = num_g / den_g

                    # shared / single center frequency
                    abs2 = np.abs(u_hat_l[half:t_len, :, n + 1]) ** 2
                    num_w = omega_freqs[half:t_len] @ abs2
                    omega_l[n + 1] = float(
                        np.sum(num_w) / (np.sum(abs2) + np.finfo(float).eps)
                    )

                    # back to time domain
                    u_hat = self._hermitian_from_onesided(
                        u_hat_l[:, :, n + 1], half=half, t_len=t_len
                    )
                    g_hat = self._hermitian_from_onesided(
                        g_hat_plus[:, :, n + 1], half=half, t_len=t_len
                    )

                    u_td = np.zeros((n_ch, t_len), dtype=float)
                    fr_td = np.zeros((n_ch, t_len), dtype=float)
                    for c in range(n_ch):
                        u_td[c, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, c])))
                        fr_td[c, :] = np.real(
                            np.fft.ifft(np.fft.ifftshift(g_hat[:, c]))
                        )

                    for c in range(n_ch):
                        temp = u_td[c, :] + fr_td[c, :]
                        f1 = f[c, :]
                        rhs = (
                            gamma * (D_arr.T @ x[:, c])
                            - (D_arr.T @ rho[:, c])
                            + f1
                            - temp
                        )
                        v = solve_jump(rhs)
                        dv = D_arr @ v
                        h = dv + coef1 * rho[:, c]
                        x[:, c] = self._soft_jump(h, mu=mu, b=b)
                        rho[:, c] = rho[:, c] - gamma * (x[:, c] - dv)
                        v = v - (np.mean(v) - np.mean(f1))
                        j_comp[c, :] = v
                        j_hat_plus[:, c, n + 1] = np.fft.fftshift(np.fft.fft(v))
                        j_hat_plus[:half, c, n + 1] = 0.0

                    n += 1
                    ud = u_hat_l[:, :, n] - u_hat_l[:, :, n - 1]
                    udiff_m = (1.0 / t_len) * (ud.conj().T @ ud)
                    jd = j_hat_plus[:, :, n] - j_hat_plus[:, :, n - 1]
                    jdiff_m = (1.0 / t_len) * (jd.conj().T @ jd)
                    udiff = float(
                        np.spacing(1)
                        + np.abs(np.sum(udiff_m))
                        + np.abs(np.sum(jdiff_m))
                    )

                # ---- Part 3: increase alpha ----
                if abs(m - np.log(self.max_alpha)) > 1:
                    m = m + 1.0
                else:
                    m = m + 0.05
                    bf += 1
                if bf >= 2:
                    alpha_val = alpha_val + 1.0

                if alpha_val <= (self.max_alpha - 1):
                    if bf == 1:
                        alpha_val = self.max_alpha - 1
                    else:
                        alpha_val = float(np.exp(m))

                    # MATLAB: omega_L = omega_L(n) → keep current omega as scalar start
                    omega_keep = float(omega_l[n])
                    udiff = self.tol + np.spacing(1)
                    temp_ud = u_hat_l[:, :, n].copy()
                    n = 0
                    u_hat_l = np.zeros((t_len, n_ch, n_max), dtype=complex)
                    u_hat_l[:, :, 0] = temp_ud
                    omega_l = np.zeros(n_max, dtype=float)
                    omega_l[0] = omega_keep

            # ---- Part 4: save mode ----
            # MATLAB: omega_L=omega_L(omega_L>0); omega_d_Temp(k)=omega_L(n-1)
            # With a leading zero at omega_l[0], filtering + (n-1) selects the last
            # positive center frequency written during ADMM.
            omega_pos = omega_l[omega_l > 0]
            omega_save = float(omega_pos[-1]) if omega_pos.size else 0.0

            mode_spec = u_hat_l[:, :, n].copy()
            u_hat_temp_list.append(mode_spec)
            omega_d_temp.append(omega_save)

            u_hat_i = np.concatenate([u_hat_i, mode_spec[:, :, None]], axis=2)

            u_hat = self._hermitian_from_onesided(mode_spec, half=half, t_len=t_len)
            u_td = np.zeros((n_ch, t_len), dtype=float)
            for c in range(n_ch):
                u_td[c, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, c])))

            f = f - u_td
            sum_u = sum_u + u_td

            f_hat = np.fft.fftshift(np.fft.fft(f, axis=1), axes=1)
            f_hat_onesided = f_hat.copy()
            f_hat_onesided[:, :half] = 0.0

            x = np.zeros((t_len, n_ch), dtype=float)
            rho = np.zeros((t_len, n_ch), dtype=float)

            sc2, polm, polm_temp, bic = self._check_stop(
                k=k,
                f_mir=f_mir,
                sum_u=sum_u,
                u_hat_i=u_hat_i,
                f_hat_temp=f_hat_temp,
                polm=polm,
                polm_temp=polm_temp,
                bic=bic,
                t_len=t_len,
            )

            # Part 6: reset for next mode
            k += 1
            if k > 50:
                sc2 = True

        # ---- Part 7: reconstruct ----
        n_modes = len(omega_d_temp)
        if n_modes == 0:
            raise RuntimeError("SJMD failed to extract any mode")

        u_hat_temp = np.stack(u_hat_temp_list, axis=2)  # (T, C, K)
        omega = np.asarray(omega_d_temp, dtype=float)

        u_hat = self._hermitian_from_onesided(u_hat_temp, half=half, t_len=t_len)

        u_time = np.zeros((t_len, n_ch, n_modes), dtype=float)
        for c in range(n_ch):
            for ki in range(n_modes):
                u_time[:, c, ki] = np.real(
                    np.fft.ifft(np.fft.ifftshift(u_hat[:, c, ki]))
                )

        order = np.argsort(omega)
        u_time = u_time[:, :, order]
        omega = omega[order]

        # remove mirror: MATLAB T/4+1:3*T/4
        u_crop = u_time[t_len // 4 : 3 * t_len // 4, :, :]
        j_crop = j_comp[:, t_len // 4 : 3 * t_len // 4] + shift

        if univariate:
            modes = np.transpose(u_crop[:, 0, :])  # (K, N)
            jump = j_crop[0, :]
        else:
            modes = np.transpose(u_crop, (2, 0, 1))  # (K, N, C)
            jump = j_crop

        self.u = modes
        self.j = jump
        self.omega = omega

        if return_all:
            return modes, jump
        return modes


# Paper name for the multivariate case (same implementation)
SMJMD = SJMD
