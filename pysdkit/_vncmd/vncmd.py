# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 12:11:34 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

MATLAB code source:
https://www.mathworks.com/matlabcentral/fileexchange/64292-variational-nonlinear-chirp-mode-decomposition
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from scipy.sparse import diags, eye as speye, csr_matrix
from scipy.sparse.linalg import spsolve

try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:  # pragma: no cover
    from scipy.integrate import cumtrapz as cumulative_trapezoid

from typing import Optional, Tuple, Union

from pysdkit._vmd.base import Base


class VNCMD(Base):
    """
    Variational Nonlinear Chirp Mode Decomposition

    Chen S, Dong X, Peng Z, et al. Nonlinear chirp mode decomposition: A variational method.
    IEEE Transactions on Signal Processing, 2017, 65(22): 6024-6037.
    """

    def __init__(
        self,
        eIF: Optional[np.ndarray] = None,
        fs: Optional[float] = None,
        alpha: float = 3e-4,
        beta: float = 1e-9,
        var: float = 1.0,
        max_iter: int = 300,
        tol: float = 1e-5,
        dtype: np.dtype = np.float64,
    ) -> None:
        """
        :param eIF: initial instantaneous frequency (IF) time series for all modes;
                    shape (K, N), each row is the IF of one mode
        :param fs: sampling frequency (Hz)
        :param alpha: penalty controlling the filtering bandwidth of VNCMD;
                      smaller alpha -> narrower bandwidth
        :param beta: penalty controlling smoothness of the IF increment;
                     smaller beta -> smoother IF updates
        :param var: variance of the Gaussian white noise; set to 0 to drop the
                    noise slack variable ``u``
        :param max_iter: maximum number of iterations
        :param tol: tolerance of the convergence criterion
        :param dtype: floating dtype used by internal arrays
        """
        self.fs = fs
        self.eIF = None if eIF is None else np.asarray(eIF, dtype=dtype)
        if self.eIF is None:
            self.K, self.N = None, None
        else:
            self.K, self.N = self.eIF.shape

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.var = float(var)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.DTYPE = dtype

        self.IFmset: Optional[np.ndarray] = None
        self.smset: Optional[np.ndarray] = None
        self.IA: Optional[np.ndarray] = None

    def __call__(
        self,
        signal: np.ndarray,
        eIF: Optional[np.ndarray] = None,
        return_all: bool = False,
    ):
        return self.fit_transform(signal=signal, eIF=eIF, return_all=return_all)

    def __str__(self) -> str:
        return "Variational Nonlinear Chirp Mode Decomposition (VNCMD)"

    @staticmethod
    def projec(vec: np.ndarray, var: float) -> np.ndarray:
        """Projection onto the ball of radius ``sqrt(M * var)`` (MATLAB ``projec``)."""
        vec = np.asarray(vec, dtype=float).ravel()
        e = np.sqrt(vec.size * var)
        nrm = norm(vec)
        if nrm > e and nrm > 0:
            return (e / nrm) * vec
        return vec.copy()

    def difference_matrix(self, n: int) -> csr_matrix:
        """
        Modified second-order difference matrix.

        MATLAB: ``oper = spdiags([e e2 e], -1:1, N, N)``
        """
        e = np.ones(n, dtype=self.DTYPE)
        e2 = -2 * np.ones(n, dtype=self.DTYPE)
        e2[0] = -1
        e2[-1] = -1
        return diags([e, e2, e], offsets=[-1, 0, 1], shape=(n, n), format="csr")

    def differ(self, y: np.ndarray, delta: float) -> np.ndarray:
        """
        Discrete derivative (MATLAB ``Differ.m``).

        :param y: input series
        :param delta: sampling interval (``1/fs``)
        """
        y = np.asarray(y, dtype=self.DTYPE).ravel()
        l = y.size
        if l < 2:
            return np.zeros_like(y)

        ybar = np.empty(l, dtype=self.DTYPE)
        ybar[0] = (y[1] - y[0]) / delta
        if l > 2:
            ybar[1:-1] = (y[2:] - y[:-2]) / (2 * delta)
        ybar[-1] = (y[-1] - y[-2]) / delta
        return ybar

    def init_K_N(
        self, eIF: Optional[np.ndarray], signal: np.ndarray
    ) -> Tuple[int, int, np.ndarray]:
        if eIF is not None:
            eIF_arr = np.array(eIF, dtype=self.DTYPE, copy=True)
        elif self.eIF is not None:
            eIF_arr = np.array(self.eIF, dtype=self.DTYPE, copy=True)
        else:
            raise ValueError(
                "Initial instantaneous frequencies `eIF` of shape (K, N) are required."
            )

        if eIF_arr.ndim != 2:
            raise ValueError("`eIF` must be a 2-D array of shape (K, N).")

        k, n = eIF_arr.shape
        if signal.size != n:
            raise ValueError(
                f"Signal length ({signal.size}) must match eIF length N={n}."
            )
        return k, n, eIF_arr

    @staticmethod
    def _diag(v: np.ndarray) -> csr_matrix:
        return diags(v, offsets=0, shape=(v.size, v.size), format="csr")

    def fit_transform(
        self,
        signal: np.ndarray,
        eIF: Optional[np.ndarray] = None,
        return_all: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute the VNCMD algorithm (faithful to MATLAB ``VNCMD.m``).

        :param signal: 1-D time-domain signal
        :param eIF: optional initial IF array of shape (K, N)
        :param return_all: if True, return full histories ``(IFmset, smset, IA)``;
                           else return final ``(modes, IF, IA)`` with shape ``(K, N)``
        """
        signal = np.asarray(signal, dtype=self.DTYPE).ravel()
        if self.fs is None:
            raise ValueError("Sampling frequency `fs` must be provided.")
        fs = float(self.fs)

        k, n, eIF_arr = self.init_K_N(eIF=eIF, signal=signal)
        t = np.arange(n, dtype=self.DTYPE) / fs
        dt = 1.0 / fs

        oper = self.difference_matrix(n)
        opedoub = (oper.T @ oper).tocsr()
        eye_n = speye(n, dtype=self.DTYPE, format="csr")

        sinm = np.zeros((k, n), dtype=self.DTYPE)
        cosm = np.zeros((k, n), dtype=self.DTYPE)
        xm = np.zeros((k, n), dtype=self.DTYPE)
        ym = np.zeros((k, n), dtype=self.DTYPE)

        if_hist = np.zeros((k, n, self.max_iter + 1), dtype=self.DTYPE)
        mode_hist = np.zeros((k, n, self.max_iter + 1), dtype=self.DTYPE)
        if_hist[:, :, 0] = eIF_arr

        lamuda = np.zeros(n, dtype=self.DTYPE)

        for i in range(k):
            phase = 2 * np.pi * cumulative_trapezoid(eIF_arr[i], t, initial=0)
            sinm[i] = np.sin(phase)
            cosm[i] = np.cos(phase)

            am = self._diag(cosm[i])
            bm = self._diag(sinm[i])
            adoubm = self._diag(cosm[i] ** 2)
            bdoubm = self._diag(sinm[i] ** 2)

            xm[i] = spsolve(2 / self.alpha * opedoub + adoubm, am.T @ signal)
            ym[i] = spsolve(2 / self.alpha * opedoub + bdoubm, bm.T @ signal)
            mode_hist[i, :, 0] = xm[i] * cosm[i] + ym[i] * sinm[i]

        # MATLAB uses 1-based iter starting at 1
        it = 1
        s_dif = self.tol + 1.0
        sum_x = np.sum(xm * cosm, axis=0)
        sum_y = np.sum(ym * sinm, axis=0)

        while s_dif > self.tol and it <= self.max_iter:
            betathr = 10 ** (it / 36.0 - 10.0)
            if betathr > self.beta:
                betathr = self.beta

            u = self.projec(signal - sum_x - sum_y - lamuda / self.alpha, self.var)

            for i in range(k):
                am = self._diag(cosm[i])
                bm = self._diag(sinm[i])
                adoubm = self._diag(cosm[i] ** 2)
                bdoubm = self._diag(sinm[i] ** 2)

                # x-update
                sum_x = sum_x - xm[i] * cosm[i]
                rhs = signal - sum_x - sum_y - u - lamuda / self.alpha
                xm[i] = spsolve(2 / self.alpha * opedoub + adoubm, am.T @ rhs)
                interx = xm[i] * cosm[i]
                sum_x = sum_x + interx

                # y-update — do not restore ym*sinm until after IF / phase update
                sum_y = sum_y - ym[i] * sinm[i]
                rhs = signal - sum_x - sum_y - u - lamuda / self.alpha
                ym[i] = spsolve(2 / self.alpha * opedoub + bdoubm, bm.T @ rhs)

                # IF update (Differ uses sampling interval 1/fs)
                xbar = self.differ(xm[i], dt)
                ybar = self.differ(ym[i], dt)
                denom = xm[i] ** 2 + ym[i] ** 2
                denom = np.where(denom < 1e-30, 1e-30, denom)
                delta_if = (xm[i] * ybar - ym[i] * xbar) / denom / (2 * np.pi)
                delta_if = spsolve(2 / betathr * opedoub + eye_n, delta_if)
                eIF_arr[i] = eIF_arr[i] - 0.5 * delta_if

                phase = 2 * np.pi * cumulative_trapezoid(eIF_arr[i], t, initial=0)
                sinm[i] = np.sin(phase)
                cosm[i] = np.cos(phase)

                sum_x = sum_x - interx + xm[i] * cosm[i]
                sum_y = sum_y + ym[i] * sinm[i]
                mode_hist[i, :, it] = xm[i] * cosm[i] + ym[i] * sinm[i]

            if_hist[:, :, it] = eIF_arr
            lamuda = lamuda + self.alpha * (u + sum_x + sum_y - signal)

            # Restart scheme
            if norm(u + sum_x + sum_y - signal) > norm(signal):
                lamuda = np.zeros(n, dtype=self.DTYPE)
                for i in range(k):
                    am = self._diag(cosm[i])
                    bm = self._diag(sinm[i])
                    adoubm = self._diag(cosm[i] ** 2)
                    bdoubm = self._diag(sinm[i] ** 2)
                    xm[i] = spsolve(2 / self.alpha * opedoub + adoubm, am.T @ signal)
                    ym[i] = spsolve(2 / self.alpha * opedoub + bdoubm, bm.T @ signal)
                    mode_hist[i, :, it] = xm[i] * cosm[i] + ym[i] * sinm[i]
                sum_x = np.sum(xm * cosm, axis=0)
                sum_y = np.sum(ym * sinm, axis=0)

            s_dif = 0.0
            for i in range(k):
                prev = mode_hist[i, :, it - 1]
                curr = mode_hist[i, :, it]
                prev_n = norm(prev)
                if prev_n > 0:
                    s_dif += (norm(curr - prev) / prev_n) ** 2

            it += 1

        self.IFmset = if_hist[:, :, :it]
        self.smset = mode_hist[:, :, :it]
        self.IA = np.sqrt(xm**2 + ym**2)

        if return_all:
            return self.IFmset, self.smset, self.IA
        return self.smset[:, :, -1], self.IFmset[:, :, -1], self.IA
