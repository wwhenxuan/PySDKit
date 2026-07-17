# -*- coding: utf-8 -*-
"""
Created on 2025/02/05 20:55:47
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Adaptive Variational Nonlinear Chirp Mode Decomposition (AVNCMD).

MATLAB reference: https://github.com/HauLiang/AVNCMD
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm, solve
from scipy.sparse import block_diag, diags, eye as speye, spdiags
from scipy.sparse.linalg import spsolve

try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid

from typing import Optional, Tuple

from pysdkit._vmd.base import Base


class AVNCMD(Base):
    """
    Adaptive Variational Nonlinear Chirp Mode Decomposition.

    [1] Liang, Hao and Ding, Xinghao and Jakobsson, Andreas and Tu, Xiaotong and Huang, Yue.
    "Adaptive Variational Nonlinear Chirp Mode Decomposition",
    in 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.
    [2] Chen S, Dong X, Peng Z, et al. Nonlinear chirp mode decomposition: A variational method.
    IEEE Transactions on Signal Processing, 2017.
    """

    def __init__(
        self,
        iniIF: Optional[np.ndarray] = None,
        fs: Optional[float] = None,
        beta: float = 1e-6,
        tol: float = 1e-5,
        max_iter: int = 500,
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        :param iniIF: initial instantaneous frequencies; each row is one mode (K, N)
        :param fs: sampling frequency; defaults to signal length when None
        :param beta: filter parameter controlling IF-increment smoothness
        :param tol: tolerance of the convergence criterion
        :param max_iter: maximum number of iterations
        :param dtype: floating dtype used by internal arrays
        """
        self.iniIF = None if iniIF is None else np.asarray(iniIF, dtype=dtype)
        self.fs = fs
        self.beta = beta
        self.tol = tol
        self.max_iter = max_iter
        self.DTYPE = dtype

        self.estIF: Optional[np.ndarray] = None
        self.estMode: Optional[np.ndarray] = None
        self.estIA: Optional[np.ndarray] = None

    def __call__(
        self,
        signal: np.ndarray,
        iniIF: Optional[np.ndarray] = None,
        fs: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Allow instances to be called like functions."""
        return self.fit_transform(signal=signal, iniIF=iniIF, fs=fs)

    def __str__(self) -> str:
        return "Adaptive Variational Nonlinear Chirp Mode Decomposition (AVNCMD)"

    @staticmethod
    def differ(y: np.ndarray, delta: float) -> np.ndarray:
        """
        Discrete derivative of a 1-D time series (MATLAB Differ.m).

        :param y: input signal
        :param delta: sampling interval of y
        :return: derivative of the same length as y
        """
        y = np.asarray(y).ravel()
        L = y.size
        ybar = np.empty(L, dtype=y.dtype)
        ybar[0] = (y[1] - y[0]) / delta
        ybar[-1] = (y[-1] - y[-2]) / delta
        if L > 2:
            ybar[1:-1] = (y[2:] - y[:-2]) / (2.0 * delta)
        return ybar

    def _resolve_iniIF(
        self, signal: np.ndarray, iniIF: Optional[np.ndarray]
    ) -> Tuple[int, int, np.ndarray]:
        """Validate / broadcast the initial IF matrix."""
        signal = np.asarray(signal, dtype=self.DTYPE).ravel()
        N = signal.size

        if iniIF is None:
            iniIF = self.iniIF
        if iniIF is None:
            raise ValueError(
                "iniIF must be provided either in __init__ or fit_transform."
            )

        iniIF = np.asarray(iniIF, dtype=self.DTYPE)
        if iniIF.ndim == 1:
            iniIF = iniIF.reshape(1, -1)

        K, n_if = iniIF.shape
        if n_if != N:
            if n_if > N:
                iniIF = iniIF[:, :N]
            else:
                # Broadcast constant IF seeds to the signal length.
                iniIF = np.repeat(iniIF[:, :1], N, axis=1)

        return K, N, iniIF.copy()

    def fit_transform(
        self,
        signal: np.ndarray,
        iniIF: Optional[np.ndarray] = None,
        fs: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run AVNCMD on a univariate signal.

        :param signal: sampled time-domain signal (1-D)
        :param iniIF: initial IFs, shape (K, N); overrides the constructor value
        :param fs: sampling frequency; overrides the constructor value
        :return: (estIF, estMode, estIA) of the last iteration
                 - estIF: estimated instantaneous frequencies, shape (K, N)
                 - estMode: estimated modes, shape (K, N)
                 - estIA: estimated instantaneous amplitudes, shape (K, N)
        """
        signal = np.asarray(signal, dtype=self.DTYPE).ravel()
        K, N, iniIF = self._resolve_iniIF(signal, iniIF)

        fs = self.fs if fs is None else fs
        if fs is None:
            fs = float(N)

        t = np.arange(N, dtype=self.DTYPE) / fs

        # Second-order difference matrix H and HtH (AVNCMD.m)
        e = np.ones(N, dtype=self.DTYPE)
        H = spdiags(
            np.vstack([e, -2.0 * e, e]),
            np.array([0, 1, 2]),
            N - 2,
            N,
            format="csc",
        )
        HtH = (H.T @ H).tocsc()

        # 2K block second-order difference matrix D
        D = block_diag([H] * (2 * K), format="csc")

        sinm = np.zeros((K, N), dtype=self.DTYPE)
        cosm = np.zeros((K, N), dtype=self.DTYPE)
        uk = np.zeros((K, N), dtype=self.DTYPE)
        vk = np.zeros((K, N), dtype=self.DTYPE)

        IFsetiter = np.zeros((K, N, self.max_iter + 1), dtype=self.DTYPE)
        IFsetiter[:, :, 0] = iniIF
        Modeset_iter = np.zeros((K, N, self.max_iter + 1), dtype=self.DTYPE)

        # Initialize dictionary matrix A
        A = np.zeros((N, 2 * K * N), dtype=self.DTYPE)
        for i in range(K):
            phase = 2.0 * np.pi * cumulative_trapezoid(iniIF[i, :], t, initial=0)
            sinm[i, :] = np.sin(phase)
            cosm[i, :] = np.cos(phase)
            Sk = diags(sinm[i, :], 0, shape=(N, N)).toarray()
            Ck = diags(cosm[i, :], 0, shape=(N, N)).toarray()
            A[:, 2 * i * N : 2 * (i + 1) * N] = np.hstack([Ck, Sk])

        it = 0
        sDif = self.tol + 1.0

        while sDif > self.tol and it < self.max_iter:
            # Gradually increase the filter parameter during iterations
            # MATLAB uses a 1-based iteration counter here.
            beta_thin = 10.0 ** ((it + 1) / 36.0 - 10.0)
            if beta_thin > self.beta:
                beta_thin = self.beta

            # Estimating the Nonlinear Chirp Signal (Section 3.1)
            x = estimate_ncs(A, D, signal, K, N)

            # Data-driven IF / mode update (Section 3.3)
            for i in range(K):
                uk[i, :] = x[2 * i * N : (2 * i + 1) * N]
                vk[i, :] = x[(2 * i + 1) * N : (2 * i + 2) * N]

                # Arctangent demodulation for the IF increment
                # (ukdif is d(vk)/dt, vkdif is d(uk)/dt — matches MATLAB)
                ukdif = self.differ(vk[i, :], 1.0 / fs)
                vkdif = self.differ(uk[i, :], 1.0 / fs)
                denom = uk[i, :] ** 2 + vk[i, :] ** 2
                deltaIF = (uk[i, :] * ukdif - vk[i, :] * vkdif) / denom / (2.0 * np.pi)

                # Low-pass correction of the IF increment
                deltaIF = spsolve(
                    (2.0 / beta_thin) * HtH + speye(N, dtype=self.DTYPE, format="csc"),
                    deltaIF,
                )
                iniIF[i, :] = np.abs(
                    iniIF[i, :] - np.asarray(deltaIF, dtype=self.DTYPE)
                )

                # Update demodulation bases
                phase = 2.0 * np.pi * cumulative_trapezoid(iniIF[i, :], t, initial=0)
                sinm[i, :] = np.sin(phase)
                cosm[i, :] = np.cos(phase)
                Sk = diags(sinm[i, :], 0, shape=(N, N)).toarray()
                Ck = diags(cosm[i, :], 0, shape=(N, N)).toarray()
                Ak = np.hstack([Ck, Sk])
                xk = np.concatenate([uk[i, :], vk[i, :]])

                Modeset_iter[i, :, it + 1] = Ak @ xk
                A[:, 2 * i * N : 2 * (i + 1) * N] = Ak

            IFsetiter[:, :, it + 1] = iniIF

            # Convergence criterion
            sDif = 0.0
            for i in range(K):
                prev = Modeset_iter[i, :, it]
                curr = Modeset_iter[i, :, it + 1]
                prev_norm = norm(prev)
                if prev_norm < np.finfo(self.DTYPE).eps:
                    sDif = np.inf
                    break
                sDif += (norm(curr - prev) / prev_norm) ** 2

            it += 1

        self.estIF = IFsetiter[:, :, : it + 1]
        self.estMode = Modeset_iter[:, :, : it + 1]
        self.estIA = np.sqrt(uk**2 + vk**2)

        return self.estIF[:, :, -1], self.estMode[:, :, -1], self.estIA


def estimate_ncs(
    A: np.ndarray,
    D,
    g: np.ndarray,
    K: int,
    N: int,
) -> np.ndarray:
    """
    Section 3.1 — Estimating the Nonlinear Chirp Signal (Estimate_NCS.m).

    Solves:  minimize  alpha * ||g - A x||_2^2 + ||D x||_2^2
    """
    g = np.asarray(g).ravel()

    # Rows of M are orthogonal to those of D (last two samples of each block)
    M = np.zeros((4 * K, 2 * K * N), dtype=A.dtype)
    ii = 0
    for i in range(2 * K):
        M[ii, (i + 1) * N - 2] = 1.0
        M[ii + 1, (i + 1) * N - 1] = 1.0
        ii += 2

    # Full-rank D_tilde = [D; M]
    if hasattr(D, "toarray"):
        D_tilde = np.vstack([D.toarray(), M])
    else:
        D_tilde = np.vstack([np.asarray(D), M])

    # Lambda = A / D_tilde  (MATLAB mrdivide)
    Lambda = solve(D_tilde.T, A.T).T
    Lambda1 = Lambda[:, : 2 * K * (N - 2)]
    Lambda2 = Lambda[:, 2 * K * (N - 2) :]

    theta_matrix = solve(Lambda2.T @ Lambda2, Lambda2.T)
    w_matrix = np.eye(N, dtype=A.dtype) - Lambda2 @ theta_matrix
    y = w_matrix @ g
    Phi = w_matrix @ Lambda1

    w = bayesian_strategy(Phi, y)
    theta = theta_matrix @ (g - Lambda1 @ w)

    # x = D_tilde \ [w; theta]
    x = solve(D_tilde, np.concatenate([w, theta]))
    return x


def bayesian_strategy(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Section 3.2 — Bayesian Strategy (Bayesian_strategy.m).

    Fast sparse Bayesian / RVM style solver for
        minimize  alpha * ||y - Phi w||_2^2 + ||w||_2^2
    """
    Phi = np.asarray(Phi)
    y = np.asarray(y).ravel()
    n_samples, m = Phi.shape

    gamma_0 = (np.std(y) ** 2) / 1e2
    eta = 1e-8
    max_iter = 1000

    PHIt = Phi.T @ y
    PHI2 = np.sum(Phi**2, axis=0)
    ratio = (PHIt**2) / PHI2
    index0 = int(np.argmax(ratio))
    maxr = ratio[index0]
    gamma = np.array([PHI2[index0] / (maxr - gamma_0)], dtype=float)

    # Active set (0-based column indices into Phi)
    index = np.array([index0], dtype=int)

    phi = Phi[:, index]  # (n_samples, 1)
    phi_energy = float(np.asarray(phi.T @ phi).reshape(-1)[0])
    Hessian = gamma[0] + phi_energy / gamma_0
    Sig = np.array([[1.0 / Hessian]], dtype=float)
    mu = np.array([Sig[0, 0] * PHIt[index0] / gamma_0], dtype=float)

    left = (Phi.T @ phi[:, 0]) / gamma_0
    S = PHI2 / gamma_0 - Sig[0, 0] * (left**2)
    Q = PHIt / gamma_0 - Sig[0, 0] * PHIt[index0] / gamma_0 * left

    ML = np.empty(max_iter, dtype=float)

    for count in range(max_iter):
        s = S.copy()
        q = Q.copy()
        s[index] = gamma * S[index] / (gamma - S[index])
        q[index] = gamma * Q[index] / (gamma - S[index])
        theta = q**2 - s

        ml = np.full(m, -np.inf, dtype=float)
        ig0 = np.where(theta > 0)[0]

        # Re-estimate candidates
        ire, which_re = _intersect_indices(ig0, index)
        if ire.size > 0:
            Gamma = s[ire] ** 2 / theta[ire]
            delta = (gamma[which_re] - Gamma) / (Gamma * gamma[which_re])
            ml[ire] = Q[ire] ** 2 * delta / (S[ire] * delta + 1.0) - np.log(
                1.0 + S[ire] * delta
            )

        # Adding candidates
        iad = np.setdiff1d(ig0, ire, assume_unique=False)
        if iad.size > 0:
            ml[iad] = (Q[iad] ** 2 - S[iad]) / S[iad] + np.log(S[iad] / (Q[iad] ** 2))

        # Deleting candidates
        is0 = np.setdiff1d(np.arange(m), ig0, assume_unique=False)
        ide, which_de = _intersect_indices(is0, index)
        if ide.size > 0:
            ml[ide] = Q[ide] ** 2 / (S[ide] - gamma[which_de]) - np.log(
                1.0 - S[ide] / gamma[which_de]
            )

        idx = int(np.argmax(ml))
        ML[count] = ml[idx]

        if count > 1 and abs(ML[count] - ML[count - 1]) < abs(ML[count] - ML[0]) * eta:
            break

        which = np.where(index == idx)[0]

        if theta[idx] > 0:
            if which.size > 0:
                # Re-estimate
                w_idx = int(which[0])
                Gamma = s[idx] ** 2 / theta[idx]
                Sigii = Sig[w_idx, w_idx]
                mui = mu[w_idx]
                Sigi = Sig[:, w_idx]
                delta = Gamma - gamma[w_idx]
                ki = delta / (1.0 + Sigii * delta)

                mu = mu - ki * mui * Sigi
                Sig = Sig - ki * np.outer(Sigi, Sigi)
                comm = (Phi.T @ (phi @ Sigi)) / gamma_0
                S = S + ki * (comm**2)
                Q = Q + ki * mui * comm
                gamma[w_idx] = Gamma
            else:
                # Adding
                Gamma = s[idx] ** 2 / theta[idx]
                phii = Phi[:, idx]
                Sigii = 1.0 / (Gamma + S[idx])
                mui = Sigii * Q[idx]
                comm1 = Sig @ ((phi.T @ phii) / gamma_0)
                ei = phii - phi @ comm1
                off = -Sigii * comm1

                Sig = np.block(
                    [
                        [Sig + Sigii * np.outer(comm1, comm1), off[:, None]],
                        [off[None, :], np.array([[Sigii]])],
                    ]
                )
                mu = np.concatenate([mu - mui * comm1, [mui]])
                comm2 = (Phi.T @ ei) / gamma_0
                S = S - Sigii * (comm2**2)
                Q = Q - mui * comm2

                index = np.append(index, idx)
                gamma = np.append(gamma, Gamma)
                phi = np.column_stack([phi, phii])
        else:
            if which.size > 0:
                # Deleting
                w_idx = int(which[0])
                Sigii = Sig[w_idx, w_idx]
                mui = mu[w_idx]
                Sigi = Sig[:, w_idx]

                Sig = Sig - np.outer(Sigi, Sigi) / Sigii
                Sig = np.delete(np.delete(Sig, w_idx, axis=0), w_idx, axis=1)
                mu = mu - (mui / Sigii) * Sigi
                mu = np.delete(mu, w_idx)

                comm = (Phi.T @ (phi @ Sigi)) / gamma_0
                S = S + (comm**2) / Sigii
                Q = Q + (mui / Sigii) * comm

                index = np.delete(index, w_idx)
                gamma = np.delete(gamma, w_idx)
                phi = np.delete(phi, w_idx, axis=1)

    w = np.zeros(m, dtype=float)
    w[index] = mu
    return w


def _intersect_indices(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB-style intersect on index arrays.

    Returns (common_values_sorted, indices_into_b).
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.size == 0 or b.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Preserve MATLAB: common values in sorted order, first occurrence in b
    common = np.intersect1d(a, b, assume_unique=False)
    which = np.array([np.where(b == c)[0][0] for c in common], dtype=int)
    return common.astype(int), which


if __name__ == "__main__":
    # Demo aligned with repo/AVNCMD/Demo_AVNCMD.m (shortened for a quick check)
    T = 1.0
    fs = 2000.0
    t = np.arange(0.0, T + 1.0 / fs, 1.0 / fs)

    a1 = np.exp(-0.03 * t)
    a2 = np.exp(-0.06 * t)
    f_1t = 25 + 8 * t - 3 * t**2 + 0.4 * t**3
    f_2t = 40 + 16 * t - 6 * t**2 + 0.4 * t**3

    g_1t = a1 * np.cos(2 * np.pi * (0.8 + 25 * t + 4 * t**2 - 1 * t**3 + 0.1 * t**4))
    g_2t = a2 * np.cos(2 * np.pi * (1 + 40 * t + 8 * t**2 - 2 * t**3 + 0.1 * t**4))
    g = g_1t + g_2t

    iniIF = np.vstack([28.0 * np.ones_like(t), 48.0 * np.ones_like(t)])
    avncmd = AVNCMD(beta=1e-6, tol=1e-5, max_iter=50)
    estIF, estMode, estIA = avncmd.fit_transform(g, iniIF=iniIF, fs=fs)

    print("estIF", estIF.shape, "estMode", estMode.shape, "estIA", estIA.shape)
    print("IF1 RE", norm(estIF[0] - f_1t) / norm(f_1t))
    print("IF2 RE", norm(estIF[1] - f_2t) / norm(f_2t))
    print("Mode1 RE", norm(estMode[0] - g_1t) / norm(g_1t))
    print("Mode2 RE", norm(estMode[1] - g_2t) / norm(g_2t))
