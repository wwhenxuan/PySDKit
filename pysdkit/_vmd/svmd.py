# -*- coding: utf-8 -*-
"""
Created on 2024/6/2 17:54
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Successive Variational Mode Decomposition (SVMD).

Unlike classical VMD, which extracts K modes concurrently and therefore
requires a known mode count, SVMD extracts compact-spectrum modes
**one by one**.  At each stage the residual / previously extracted modes
are discouraged from overlapping the mode of interest, and a stopping
criterion decides when to halt.

Mojtaba Nazari, Sayed Mahmoud Sakhaei.
Successive Variational Mode Decomposition,
Signal Processing 174 (2020) 107610.
https://doi.org/10.1016/j.sigpro.2020.107610
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm
from scipy.signal import savgol_filter

from .base import Base


class SVMD(Base):
    """
    Successive Variational Mode Decomposition.

    SVMD extends the variational framework of VMD by extracting modes
    successively.  Each new mode ``u_L`` is obtained by solving a constrained
    optimization problem that enforces:

    1. a compact spectrum around a center frequency ``ω_L`` (as in VMD);
    2. little spectral overlap with previously extracted modes;
    3. little spectral overlap with the residual signal.

    Because modes are obtained one after another, the algorithm does **not**
    require the number of modes ``K`` a priori.  Several stopping criteria
    (noise power, exact reconstruction, BIC, power of the last mode) are
    available.

    Nazari & Sakhaei, Signal Processing, 174:107610, 2020.
    """

    def __init__(
        self,
        max_alpha: float = 20000,
        tau: float = 0.0,
        tol: float = 1e-6,
        stopc: int = 4,
        init_omega: int = 0,
        max_iter: int = 300,
        max_modes: int = 30,
        poly_order: int = 8,
        window_length: int = 25,
        random_seed: int = 42,
    ) -> None:
        """
        :param max_alpha: balancing parameter of the data-fidelity constraint
            (compactness of each mode).  Typical values: 1e3–2e4.
        :param tau: dual-ascent time step.  Set to 0 under high-level noise.
        :param tol: relative convergence tolerance for each mode (≈ 1e-6).
        :param stopc: stopping-criterion type
            1 – residual power vs. estimated noise (compact spectra / EEG);
            2 – exact reconstruction (clean signals);
            3 – Bayesian information criterion (BIC);
            4 – power of the last mode (default, as in the MATLAB toolbox).
        :param init_omega: center-frequency initialisation
            0 – start each mode at DC (default, usually sufficient);
            1 – random initialisation avoiding previously found frequencies.
        :param max_iter: maximum ADMM iterations used to refine each mode.
        :param max_modes: hard upper bound on the number of extracted modes
            (safety guard against non-termination).
        :param poly_order: Savitzky–Golay polynomial order used for noise
            estimation (stopc = 1).
        :param window_length: Savitzky–Golay window length (must be odd).
        :param random_seed: RNG seed for ``init_omega = 1``.
        """
        super().__init__()
        self.max_alpha = float(max_alpha)
        self.tau = float(tau)
        self.tol = float(tol)
        self.stopc = int(stopc)
        self.init_omega = int(init_omega)
        self.max_iter = int(max_iter)
        self.max_modes = int(max_modes)
        self.poly_order = int(poly_order)
        self.window_length = int(window_length)
        self.eps = np.finfo(np.float64).eps
        self.rng = np.random.RandomState(seed=random_seed)

        if self.stopc not in (1, 2, 3, 4):
            raise ValueError("stopc must be one of {1, 2, 3, 4}")
        if self.init_omega not in (0, 1):
            raise ValueError("init_omega must be 0 or 1")

        self.signal: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.u_hat: Optional[np.ndarray] = None
        self.omega: Optional[np.ndarray] = None

    def __call__(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Allow instances to be called like functions."""
        return self.fit_transform(signal=signal, return_all=return_all)

    def __str__(self) -> str:
        return "Successive Variational Mode Decomposition (SVMD)"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _sgolay(
        self,
        signal: np.ndarray,
        poly_order: Optional[int] = None,
        window_length: Optional[int] = None,
    ) -> np.ndarray:
        """Savitzky–Golay smoother used to estimate the noise floor."""
        poly_order = self.poly_order if poly_order is None else int(poly_order)
        window_length = (
            self.window_length if window_length is None else int(window_length)
        )
        if window_length % 2 == 0:
            window_length += 1
        if window_length > signal.size:
            window_length = signal.size if signal.size % 2 == 1 else signal.size - 1
        if window_length < poly_order + 2:
            poly_order = max(1, window_length - 2)
        return savgol_filter(signal, window_length=window_length, polyorder=poly_order)

    def _init_omega_scalar(self, fs: float) -> float:
        """Return the initial center frequency for a new mode."""
        if self.init_omega == 0:
            return 0.0
        return float(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * self.rng.rand()))

    def _reinit_omega(self, fs: float, omega_d: List[float]) -> Tuple[float, int]:
        """
        Draw a random ω that is at least ≈ 0.02 (normalised Hz) away from
        previously extracted center frequencies.  Returns ``(omega, n2)``.
        """
        n2 = 0
        omega = 0.0
        while n2 < 300:
            omega = float(
                np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * self.rng.rand())
            )
            if len(omega_d) == 0 or np.min(np.abs(np.asarray(omega_d) - omega)) >= 0.02:
                return omega, n2 + 1
            n2 += 1
        return omega, n2

    # ------------------------------------------------------------------ #
    # Main API
    # ------------------------------------------------------------------ #
    def fit_transform(
        self,
        signal: np.ndarray,
        return_all: bool = False,
        poly_order: Optional[int] = None,
        window_length: Optional[int] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Decompose a 1-D signal with SVMD.

        :param signal: real-valued 1-D array (even length preferred; odd
            length is truncated by one sample, matching the MATLAB code).
        :param return_all: if True, also return spectra and center frequencies.
        :param poly_order: optional override of the Savitzky–Golay order.
        :param window_length: optional override of the Savitzky–Golay window.
        :return: modes ``(L, N)``, optionally ``(u, u_hat, omega)``.
        """
        signal = np.asarray(signal, dtype=float).ravel()
        if signal.size < 4:
            raise ValueError("signal length must be at least 4 samples")

        # MATLAB: discard the last sample if the length is odd
        if signal.size % 2 == 1:
            signal = signal[:-1]
        save_T = signal.size
        self.signal = signal

        # ---- Part 1: initialisation -----------------------------------
        y = self._sgolay(signal, poly_order=poly_order, window_length=window_length)
        signoise = signal - y
        fs = 1.0 / save_T

        # Mirror extension of the signal and of the estimated noise
        f = self.fmirror(signal, save_T // 2)
        fnoise = self.fmirror(signoise, save_T // 2)

        T = f.size
        t = np.arange(1, T + 1, dtype=float) / T
        omega_freqs = t - 0.5 - 1.0 / T
        half = T // 2

        # One-sided FFT (Hilbert concept used by VMD / SVMD)
        f_hat = self.fftshift(self.fft(f))
        f_hat_onesided = f_hat.copy()
        f_hat_onesided[:half] = 0.0

        f_hat_n = self.fftshift(self.fft(fnoise))
        f_hat_n_onesided = f_hat_n.copy()
        f_hat_n_onesided[:half] = 0.0
        noisepe = float(norm(f_hat_n_onesided, ord=2) ** 2)

        N = self.max_iter
        min_alpha = 10.0

        # Accumulated modes / filters / frequencies
        modes_hat: List[np.ndarray] = []  # each: (T,) complex
        omega_d: List[float] = []
        alpha_hist: List[float] = []
        h_rows: List[np.ndarray] = []  # filter rows for previously extracted modes

        # Stopping-criterion auxiliaries
        polm_list: List[float] = []
        polm_temp = 1.0
        bic_list: List[float] = []
        sigerror_list: List[float] = []
        normind_list: List[float] = []

        SC2 = 0
        mode_idx = 0  # 0-based mode counter (= MATLAB l-1)

        # ---- Part 2–6: successive extraction --------------------------
        while SC2 != 1 and mode_idx < self.max_modes:
            # Fresh ADMM state for the new mode
            Alpha = min_alpha
            omega_L = np.zeros(N, dtype=float)
            n2 = 0
            if self.init_omega == 0 or mode_idx == 0:
                omega_L[0] = self._init_omega_scalar(fs)
            else:
                omega_L[0], n2 = self._reinit_omega(fs, omega_d)

            lambda_hat = np.zeros((N, T), dtype=complex)
            u_hat_L = np.zeros((N, T), dtype=complex)

            udiff = self.tol + self.eps
            n = 0  # 0-based iteration index (MATLAB n starts at 1)
            m = 0.0
            bf = 0

            h_sum = (
                np.sum(np.vstack(h_rows), axis=0)
                if len(h_rows) > 0
                else np.zeros(T, dtype=float)
            )
            u_sum = (
                np.sum(np.vstack(modes_hat), axis=0)
                if len(modes_hat) > 0
                else np.zeros(T, dtype=complex)
            )

            # Outer α-growth loop --------------------------------------
            while Alpha < self.max_alpha + 1:
                # Inner ADMM loop --------------------------------------
                while udiff > self.tol and n < N - 1:
                    dw = omega_freqs - omega_L[n]
                    dw2 = dw**2
                    dw4 = dw2**2
                    A2 = Alpha**2

                    # Update mode spectrum û_L  (Eq. derived via ADMM)
                    numer = (
                        f_hat_onesided + (A2 * dw4) * u_hat_L[n] + lambda_hat[n] / 2.0
                    )
                    denom = 1.0 + (A2 * dw4) * (1.0 + 2.0 * Alpha * dw2) + h_sum
                    u_hat_L[n + 1] = numer / denom

                    # Update center frequency ω_L (positive frequencies only)
                    power = np.abs(u_hat_L[n + 1, half:]) ** 2
                    power_sum = np.sum(power)
                    if power_sum > 0:
                        omega_L[n + 1] = float(
                            np.dot(omega_freqs[half:], power) / power_sum
                        )
                    else:
                        omega_L[n + 1] = omega_L[n]

                    # Dual ascent for λ
                    if self.tau != 0.0:
                        filter_num = (
                            A2
                            * dw4
                            * (
                                f_hat_onesided
                                - u_hat_L[n + 1]
                                - u_sum
                                + lambda_hat[n] / 2.0
                            )
                            - u_sum
                        )
                        filter_den = 1.0 + A2 * dw4
                        residual = f_hat_onesided - (
                            u_hat_L[n + 1] + filter_num / filter_den + u_sum
                        )
                        lambda_hat[n + 1] = lambda_hat[n] + self.tau * residual
                    else:
                        lambda_hat[n + 1] = lambda_hat[n]

                    # Relative spectral update
                    diff = u_hat_L[n + 1] - u_hat_L[n]
                    num = np.vdot(diff, diff).real / T
                    den = np.vdot(u_hat_L[n], u_hat_L[n]).real / T + self.eps
                    udiff = abs(num / den)
                    n += 1

                # ---- Part 3: increase α toward max_alpha -------------
                if abs(m - np.log(self.max_alpha)) > 1.0:
                    m = m + 1.0
                else:
                    m = m + 0.05
                    bf = bf + 1

                if bf >= 2:
                    Alpha = Alpha + 1.0

                if Alpha <= (self.max_alpha - 1.0):
                    if bf == 1:
                        Alpha = self.max_alpha - 1.0
                    else:
                        Alpha = float(np.exp(m))

                    # Keep the current ω / spectrum and restart ADMM
                    omega_keep = float(omega_L[n])
                    temp_ud = u_hat_L[n].copy()

                    udiff = self.tol + self.eps
                    n = 0
                    lambda_hat = np.zeros((N, T), dtype=complex)
                    u_hat_L = np.zeros((N, T), dtype=complex)
                    u_hat_L[0] = temp_ud
                    omega_L = np.zeros(N, dtype=float)
                    omega_L[0] = omega_keep

            # ---- Part 4: store the converged mode --------------------
            # Last written iterate is index n (MATLAB uses u_hat_L(n,:)
            # after the while, with ω from the previous step).
            mode_spectrum = u_hat_L[n].copy()
            omega_val = float(omega_L[max(n - 1, 0)])
            if omega_val <= 0.0 and n > 0:
                # Fall back to the last strictly positive ω, matching
                # MATLAB ``omega_L = omega_L(omega_L > 0)``.
                pos = omega_L[omega_L > 0]
                omega_val = float(pos[-1]) if pos.size > 0 else float(omega_L[n])

            modes_hat.append(mode_spectrum)
            omega_d.append(omega_val)
            alpha_hist.append(float(Alpha))

            # Filter that suppresses this mode in subsequent extractions
            gamma = 1.0
            h_row = gamma / (
                (alpha_hist[-1] ** 2) * (omega_freqs - omega_d[-1]) ** 4 + self.eps
            )
            h_rows.append(h_row)

            # ---- Part 5: stopping criteria ---------------------------
            u_stack = np.vstack(modes_hat)
            residual = f_hat_onesided - np.sum(u_stack, axis=0)

            if self.stopc == 1:
                # Residual power vs. estimated noise power
                sigerror = float(norm(residual, ord=2) ** 2)
                sigerror_list.append(sigerror)
                if n2 >= 300 or sigerror <= round(noisepe):
                    SC2 = 1

            elif self.stopc == 2:
                # Exact reconstruction (relative residual energy)
                sum_u = np.sum(u_stack, axis=0)
                ni = (norm(sum_u - f_hat_onesided) ** 2 / T) / (
                    norm(f_hat_onesided) ** 2 / T + self.eps
                )
                normind_list.append(float(ni))
                if n2 >= 300 or ni < 0.005:
                    SC2 = 1

            elif self.stopc == 3:
                # Bayesian information criterion
                sigerror = float(norm(residual, ord=2) ** 2)
                sigerror_list.append(sigerror)
                # MATLAB: BIC(l) = 2*T*log(sigerror) + (3*l)*log(2*T)
                # with 1-based l = mode_idx + 1
                bic = 2 * T * np.log(max(sigerror, self.eps)) + (
                    3 * (mode_idx + 1)
                ) * np.log(2 * T)
                bic_list.append(float(bic))
                if mode_idx >= 1 and bic_list[-1] > bic_list[-2]:
                    SC2 = 1

            else:
                # stopc == 4: power of the last mode (default)
                mode_l = modes_hat[-1]
                omega_l = omega_d[-1]
                # After α-growth, Alpha sits near / above max_alpha
                A = float(Alpha)
                filt = (4.0 * A * mode_l) / (
                    1.0 + 2.0 * A * (omega_freqs - omega_l) ** 2
                )
                polm_val = float(norm(filt * np.conj(mode_l), ord=2))
                if mode_idx < 1:
                    polm_temp = polm_val if polm_val != 0 else 1.0
                    polm_list.append(1.0)
                else:
                    polm_list.append(polm_val / (polm_temp + self.eps))
                    if abs(polm_list[-1] - polm_list[-2]) < 0.001:
                        SC2 = 1

            # ---- Part 6: prepare the next mode -----------------------
            mode_idx += 1

        if len(modes_hat) == 0:
            raise RuntimeError("SVMD failed to extract any mode")

        # ---- Part 7: signal reconstruction ---------------------------
        L = len(modes_hat)
        omega = np.asarray(omega_d, dtype=float)

        u_hat_full = np.zeros((T, L), dtype=complex)
        stacked = np.vstack(modes_hat).T  # (T, L)
        # Hermitian completion identical to VMD / MATLAB SVMD
        idxs = np.flip(np.arange(1, half + 1), axis=0)
        u_hat_full[half:T, :] = stacked[half:T, :]
        u_hat_full[idxs, :] = np.conj(stacked[half:T, :])
        u_hat_full[0, :] = np.conj(u_hat_full[-1, :])

        u = np.zeros((L, T), dtype=float)
        for ell in range(L):
            u[ell, :] = np.real(self.ifft(self.ifftshift(u_hat_full[:, ell])))

        # Sort by ascending center frequency
        order = np.argsort(omega)
        omega = omega[order]
        u = u[order, :]

        # Remove the mirrored borders
        u = u[:, T // 4 : 3 * T // 4]

        # Recompute spectra on the cropped modes
        u_hat = np.zeros((save_T, L), dtype=complex)
        for ell in range(L):
            u_hat[:, ell] = self.fftshift(self.fft(u[ell, :]))

        self.u = u
        self.u_hat = u_hat
        self.omega = omega

        if return_all:
            return u, u_hat, omega
        return u


def svmd(
    signal: np.ndarray,
    max_alpha: float = 20000,
    tau: float = 0.0,
    tol: float = 1e-6,
    stopc: int = 4,
    init_omega: int = 0,
    max_iter: int = 300,
    return_all: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Functional interface to Successive Variational Mode Decomposition.

    See :class:`SVMD` for parameter descriptions.
    """
    return SVMD(
        max_alpha=max_alpha,
        tau=tau,
        tol=tol,
        stopc=stopc,
        init_omega=init_omega,
        max_iter=max_iter,
    ).fit_transform(signal=signal, return_all=return_all)
