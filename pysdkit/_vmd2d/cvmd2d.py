# -*- coding: utf-8 -*-
"""
Compact / Two-Dimensional TV Variational Mode Decomposition (CVMD2D / 2D-TV-VMD).

Zosso D., Dragomiretskiy K., Bertozzi A.L., Weiss P.S.
Two-Dimensional Compact Variational Mode Decomposition.
Journal of Mathematical Imaging and Vision, 58(2):294–320, 2017.
https://doi.org/10.1007/s10851-017-0710-z

MATLAB reference:
https://www.mathworks.com/matlabcentral/fileexchange/67285-two-dimensional-compact-variational-mode-decomposition-2d-tv-vmd
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm

from pysdkit.utils import fft2d, ifft2d, fftshift, ifftshift


class CVMD2D(object):
    """
    Compact Variational Mode Decomposition for 2D Images (2D-TV-VMD).

    Decomposes an image into spatially compact, spectrally sparse modes with
    optional support segmentation (MBO / winner-takes-all) and artifact maps.

    The optimisation proceeds in three scheduled phases (``A_phase = [a, b]``):

    - iterations ``1 … a-1``: classical 2D VMD (no spatial support evolution)
    - iterations ``a … b-1``: 2D-TV-VMD (individual MBO on supports ``A_k``)
    - iterations ``b … end``: segmented 2D-TV-VMD (joint winner-takes-all)

    MATLAB code: ``VMD_2D_TV.m`` (Zosso & Dragomiretskiy).
    """

    def __init__(
        self,
        K: int = 5,
        alpha: float = 1000,
        beta: float = 0.5,
        gamma: float = 500,
        delta: float = np.inf,
        rho: float = 10,
        rho_k: float = 10,
        tau: float = 0.0,
        tau_k: float = 2.5,
        t: float = 1.5,
        DC: bool = False,
        init: Union[str, int, np.ndarray] = "radially",
        u_tol: float = 1e-10,
        A_tol: float = 1e-4,
        omega_tol: float = 1e-10,
        max_iter: int = 130,
        M: int = 1,
        A_phase: Optional[np.ndarray] = None,
        random_seed: int = 42,
    ) -> None:
        """
        :param K: number of modes
        :param alpha: spectral narrow-band / bandwidth penalty
        :param beta: L1 area penalty on spatial supports ``A_k``
        :param gamma: heat-diffusion weight for TV (MBO) propagation of ``A_k``
        :param delta: artifact threshold on residual energy (``inf`` → disabled)
        :param rho: data-fidelity weight
        :param rho_k: u–v splitting weight
        :param tau: dual step for data fidelity (0 → noise-slack)
        :param tau_k: dual step for u–v splitting
        :param t: ODE / PDE step for support updates
        :param DC: keep the first mode at the DC frequency ``(0, 0)``
        :param init: ``"radially"`` / ``"uniform"`` / ``0``, ``"random"`` / ``1``,
                     or an array of shape ``(2, K, M)`` with custom centre frequencies
        :param u_tol: relative tolerance on modes ``u``
        :param A_tol: absolute tolerance on supports ``A``
        :param omega_tol: tolerance on centre frequencies
        :param max_iter: maximum ADMM iterations ``N``
        :param M: number of spectral sub-modes per spatial mode
        :param A_phase: ``[a, b]`` phase schedule (default ``[100, 150]``)
        :param random_seed: RNG seed for random omega initialisation
        """
        self.K = int(K)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.rho = float(rho)
        self.rho_k = float(rho_k)
        self.tau = float(tau)
        self.tau_k = float(tau_k)
        self.t = float(t)
        self.DC = bool(DC)
        self.init = init
        self.u_tol = float(u_tol)
        self.A_tol = float(A_tol)
        self.omega_tol = float(omega_tol)
        self.max_iter = int(max_iter)
        self.M = int(M)
        if A_phase is None:
            self.A_phase = np.array([100.0, 150.0], dtype=float)
        else:
            self.A_phase = np.asarray(A_phase, dtype=float).ravel()
            if self.A_phase.size != 2:
                raise ValueError("A_phase must be a length-2 array [a, b]")

        self.rng = np.random.default_rng(seed=random_seed)

    def __call__(
        self, image: np.ndarray, return_all: bool = False
    ) -> Union[
        np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        return self.fit_transform(image=image, return_all=return_all)

    def __str__(self) -> str:
        return "Compact Variational Mode Decomposition for 2D Images (CVMD2D)"

    def _init_omega(self) -> np.ndarray:
        """Initialise centre frequencies ``omega`` (MATLAB ``init`` cases 0/1 / custom)."""
        # (N+1) × 2 × K × M — extra slot so omega[n+1] is always valid
        omega = np.zeros((self.max_iter + 1, 2, self.K, self.M), dtype=float)

        init = self.init
        # Map friendly aliases / MATLAB numeric codes
        if isinstance(init, str):
            key = init.lower()
            if key in ("radially", "uniform", "radial"):
                init = 0
            elif key == "random":
                init = 1
            else:
                raise ValueError(
                    "init string must be one of "
                    "['radially', 'uniform', 'random']; got {!r}".format(init)
                )

        if isinstance(init, (int, np.integer)):
            if init == 0:
                # Radially uniform on the half-plane
                max_k = self.K - 1 if self.DC else self.K
                radius = 0.3
                # MATLAB: for k = DC+(1:maxK) with angle (k-1+(m-1)*maxK)
                # 0-based equivalent: angle factor = (k + m*max_k)
                for k in range(int(self.DC), int(self.DC) + max_k):
                    for m in range(self.M):
                        angle = np.pi * (k + m * max_k) / max_k / self.M
                        omega[0, 0, k, m] = radius * np.cos(angle)
                        omega[0, 1, k, m] = radius * np.sin(angle)
            elif init == 1:
                for k in range(self.K):
                    for m in range(self.M):
                        omega[0, 0, k, m] = self.rng.random() - 0.5
                        omega[0, 1, k, m] = self.rng.random() / 2.0
                if self.DC:
                    omega[0, :, 0, :] = 0.0
            else:
                raise ValueError("numeric init must be 0 (radially) or 1 (random)")
        else:
            arr = np.asarray(init, dtype=float)
            if arr.size != 2 * self.K * self.M:
                raise ValueError(
                    "custom init must have size 2*K*M (shape (2, K, M)); "
                    "got size {}".format(arr.size)
                )
            arr = arr.reshape(2, self.K, self.M)
            omega[0, :, :, :] = arr

        return omega

    def fit_transform(
        self, image: np.ndarray, return_all: bool = False
    ) -> Union[
        np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Decompose a 2D image with compact / TV variational mode decomposition.

        :param image: real 2D array ``(Hy, Hx)``
        :param return_all: if True, also return ``v, omega, A, X``
        :return: modes ``u`` with shape ``(Hy, Hx, K, M)``, or the full tuple
        """
        signal = np.asarray(image, dtype=float)
        if signal.ndim != 2:
            raise ValueError("image must be a 2-D array")
        hy, hx = signal.shape
        if hy < 4 or hx < 4:
            raise ValueError("image spatial size must be at least 4×4")

        # Normalised spatial grid (MATLAB meshgrid)
        grid_x, grid_y = np.meshgrid(
            np.arange(1, hx + 1) / hx, np.arange(1, hy + 1) / hy
        )

        # Spectral domain discretisation — fy uses Hy (bugfix vs old Hx)
        fx = 1.0 / hx
        fy = 1.0 / hy
        freqs_1 = grid_x - 0.5 - fx
        freqs_2 = grid_y - 0.5 - fy

        # Storage (Fourier modes are complex)
        u_hat = np.zeros((hy, hx, self.K, self.M), dtype=complex)
        u = np.zeros((hy, hx, self.K, self.M), dtype=float)
        u_old = u.copy()
        v = u.copy()

        # Augmented Lagrangian variables
        lambda_k = np.zeros_like(u)  # u/v linking  ~ rho_k
        lambda_d = np.zeros((hy, hx), dtype=float)  # data fidelity ~ rho

        # Spatial supports & artifact map
        A = np.ones((hy, hx, self.K), dtype=float)
        A_old = A.copy()
        artifact = np.zeros((hy, hx), dtype=bool)

        omega = self._init_omega()

        u_diff = np.inf
        a_diff = np.inf
        omega_diff = np.inf
        sum_avk = 0.0

        # Phase-schedule lower bound (handles Inf in A_phase like MATLAB)
        finite_phases = self.A_phase[np.isfinite(self.A_phase)]
        phase_bound = float(np.max(finite_phases)) if finite_phases.size else 0.0

        # MATLAB starts n = 1; we use 0-based n and compare with (n + 1)
        n = 0
        while n < self.max_iter and (
            (u_diff > self.u_tol or a_diff > self.A_tol or omega_diff > self.omega_tol)
            or (n + 1) <= phase_bound
        ):
            # ---- modes / submodes ------------------------------------------------
            for k in range(self.K):
                for m in range(self.M):
                    hilbert_mask = (
                        np.sign(
                            freqs_1 * omega[n, 0, k, m] + freqs_2 * omega[n, 1, k, m]
                        )
                        + 1.0
                    )

                    # Update accumulator of A_j v_j for j ≠ (k,m)
                    if m == 0:
                        if k == 0:
                            sum_avk = (
                                sum_avk
                                + A[:, :, -1] * v[:, :, -1, -1]
                                - A[:, :, 0] * v[:, :, 0, 0]
                            )
                        else:
                            sum_avk = (
                                sum_avk
                                + A[:, :, k - 1] * v[:, :, k - 1, -1]
                                - A[:, :, k] * v[:, :, k, 0]
                            )
                    else:
                        sum_avk = (
                            sum_avk
                            + A[:, :, k] * v[:, :, k, m - 1]
                            - A[:, :, k] * v[:, :, k, m]
                        )

                    one_minus_x = 1.0 - artifact.astype(float)

                    # Update v (spatial-domain averaging)
                    v[:, :, k, m] = (
                        self.rho_k * u[:, :, k, m]
                        + lambda_k[:, :, k, m]
                        + self.rho
                        * A[:, :, k]
                        * (signal - sum_avk + lambda_d / self.rho)
                        * one_minus_x
                    ) / (self.rho_k + self.rho * one_minus_x * A[:, :, k] ** 2)

                    # Update u_hat (analytic spectrum via Wiener filter)
                    u_hat[:, :, k, m] = (
                        fftshift(
                            fft2d(self.rho_k * v[:, :, k, m] - lambda_k[:, :, k, m])
                        )
                        * hilbert_mask
                    ) / (
                        self.rho_k
                        + 2.0
                        * self.alpha
                        * (
                            (freqs_1 - omega[n, 0, k, m]) ** 2
                            + (freqs_2 - omega[n, 1, k, m]) ** 2
                        )
                    )

                    # Centre frequencies (keep first mode at 0 if DC)
                    if (not self.DC) or k > 0:
                        power = np.abs(u_hat[:, :, k, m]) ** 2
                        denom = np.sum(power)
                        if denom > 1e-30:
                            omega[n + 1, 0, k, m] = np.sum(freqs_1 * power) / denom
                            omega[n + 1, 1, k, m] = np.sum(freqs_2 * power) / denom
                        # Keep omegas on the top half-plane
                        if omega[n + 1, 1, k, m] < 0:
                            omega[n + 1, :, k, m] = -omega[n + 1, :, k, m]

                    # Recover real mode from analytic spectrum
                    u[:, :, k, m] = np.real(ifft2d(ifftshift(u_hat[:, :, k, m])))

                # Phase II: individual MBO / TV support propagation
                # MATLAB: n >= A_phase(1) && n < A_phase(2)  (1-based n)
                if self.A_phase[0] <= (n + 1) < self.A_phase[1]:
                    one_minus_x = 1.0 - artifact.astype(float)
                    sum_v_k = np.sum(v[:, :, k, :], axis=2)
                    A[:, :, k] = A[:, :, k] + self.t * (
                        -self.beta
                        + 2.0
                        * self.rho
                        * sum_v_k
                        * (
                            signal
                            - np.sum(A * np.sum(v, axis=3), axis=2)
                            + A[:, :, k] * sum_v_k
                            + lambda_d / self.rho
                        )
                        * one_minus_x
                    )
                    A[:, :, k] = A[:, :, k] / (
                        1.0 + self.t * 2.0 * self.rho * one_minus_x * sum_v_k**2
                    )

                    A[A > 1] = 1.0
                    A[A < 0] = 0.0

                    # Heat equation / spectral diffusion
                    A[:, :, k] = np.real(
                        ifft2d(
                            fft2d(A[:, :, k])
                            / (
                                1.0
                                + self.t
                                * self.gamma
                                * ifftshift(freqs_1**2 + freqs_2**2)
                            )
                        )
                    )
                    A[:, :, k] = (A[:, :, k] >= 0.5).astype(float)

            # Phase III: joint MBO + winner-takes-all segmentation
            if (n + 1) >= self.A_phase[1]:
                sum_av = np.sum(A * np.sum(v, axis=3), axis=2)
                for k in range(self.K):
                    sum_v_k = np.sum(v[:, :, k, :], axis=2)
                    A[:, :, k] = A[:, :, k] + self.t * (
                        -self.beta
                        + 2.0
                        * self.rho
                        * sum_v_k
                        * (signal - sum_av + A[:, :, k] * sum_v_k + lambda_d / self.rho)
                    )
                    A[:, :, k] = A[:, :, k] / (
                        1.0 + self.t * 2.0 * self.rho * sum_v_k**2
                    )
                    A[:, :, k] = np.real(
                        ifft2d(
                            fft2d(A[:, :, k])
                            / (
                                1.0
                                + self.t
                                * self.gamma
                                * ifftshift(freqs_1**2 + freqs_2**2)
                            )
                        )
                    )

                # Winner-takes-all (column-major / Fortran order like MATLAB)
                a_flat = A.reshape(hy * hx, self.K, order="F")
                winners = np.argmax(a_flat, axis=1)
                a_new = np.zeros_like(a_flat)
                a_new[np.arange(hy * hx), winners] = 1.0
                A = a_new.reshape(hy, hx, self.K, order="F")

            # Artifact thresholding
            residual = signal - np.sum(A * np.sum(v, axis=3), axis=2)
            artifact = residual**2 >= self.delta

            # Dual ascent — data fidelity
            lambda_d = lambda_d + self.tau * residual
            # Dual ascent — u/v splitting (must use tau_k, not tau)
            lambda_k = lambda_k + self.tau_k * (u - v)

            n += 1

            # Stopping criteria (match MATLAB formulas)
            u_norm = norm(u.ravel())
            if u_norm > 1e-30:
                u_diff = (norm((u - u_old).ravel()) ** 2) / (u_norm**2) / (hx * hy)
            else:
                u_diff = 0.0
            a_diff = norm(A.ravel() - A_old.ravel(), ord=1) / (hx * hy)
            omega_diff = float(norm(omega[n, :, :, :] - omega[n - 1, :, :, :]) ** 2)

            u_old = u.copy()
            A_old = A.copy()

        omega_final = omega[n, :, :, :]

        if return_all:
            return u, v, omega_final, A, artifact.astype(float)
        return u
