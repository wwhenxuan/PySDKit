# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 12:11:34 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

MATLAB code source https://www.mathworks.com/matlabcentral/fileexchange/64292-variational-nonlinear-chirp-mode-decomposition
"""
import numpy as np
from numpy.linalg import solve, norm
from scipy.sparse import diags, eye

try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid
from typing import Tuple, Optional
from pysdkit._vmd.base import Base


class VNCMD(Base):
    """
    Variational Nonlinear Chirp Mode Decomposition

    Chen S, Dong X, Peng Z, et al. Nonlinear chirp mode decomposition: A variational method[J].
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
    ):
        """
        :param eIF: initial instantaneous frequency (IF) time series for all the signal modes; each row of eIF corresponds to the IF of each mode
        :param fs: sampling frequency
        :param alpha: penalty parameter controling the filtering bandwidth of VNCMD;the smaller the alpha is, the narrower the bandwidth would be
        :param beta: penalty parameter controling the smooth degree of the IF increment during iterations;the smaller the beta is, the more smooth the IF increment would be
        :param var: the variance of the Gaussian white noise; if we set var to zero, the noise variable u (see the following code) will be dropped.
        :param max_iter: the maximum number of iterations
        :param tol: tolerance of convergence criterion; typically 1e-7, 1e-8, 1e-9...
        :param dtype: data types used by all operations
        """

        self.fs = fs

        self.eIF = eIF
        if self.eIF is None:
            self.K, self.N = None, None
        else:
            self.K, self.N = self.eIF.shape[0], self.eIF.shape[1]

        self.alpha = alpha
        self.beta = beta
        self.var = var

        self.max_iter = max_iter
        self.tol = tol

        # 保存本次运算结果
        self.IFmset = None
        self.smset = None
        self.IA = None

        self.DTYPE = dtype

    def __call__(self, signal: np.ndarray, eIF: Optional[np.ndarray] = None):
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal, eIF=eIF)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Variational Nonlinear Chirp Mode Decomposition (VNCMD)"

    @staticmethod
    def projec(vec: np.ndarray, var: float) -> np.ndarray:
        """
        Projection operation.

        :param vec: The vector for projection.
        :param var: The variance of the noise.
        :return: numpy.ndarray: The projected vector.
        """
        M = len(vec)
        e = np.sqrt(M * var)  # Upper bound determined by the noise level
        u = vec.copy()  # Copy the input vector to avoid modifying the original

        if np.linalg.norm(vec) > e:
            u = (e / np.linalg.norm(vec)) * vec

        return u

    def difference_matrix(self, N: int) -> np.ndarray:
        """
        Constructs an NxN second-order difference matrix.

        :param N: The size of the matrix.
        :return: The second-order difference matrix.
        """
        # Create an K-element column vector filled with ones
        e = np.ones(N, dtype=self.DTYPE)
        # Create an K-element column vector filled with -2
        e2 = -2 * np.ones(N, dtype=self.DTYPE)
        # Set the first element of e2 to -1
        e2[0] = -1
        # Set the last element of e2 to -1
        e2[-1] = -1

        # Create an NxN matrix filled with zeros
        oper = np.zeros((N, N), dtype=self.DTYPE)

        # Fill the main diagonal with e2
        np.fill_diagonal(oper, e2)
        # Fill the first upper diagonal with 1s, leaving the last element
        np.fill_diagonal(oper[1:], e[:-1])
        # Fill the first lower diagonal with 1s, leaving the last element
        np.fill_diagonal(oper[:, 1:], e[:-1])

        return oper

    def differ(self, y: np.ndarray, delta: float) -> np.ndarray:
        """
        Compute the derivative of a discrete time series y.

        :param y: The input time series.
        :param delta: The sampling time interval of y.
        :return: numpy.ndarray: The derivative of the time series.
        """
        L = len(y)
        ybar = np.zeros(L - 2, dtype=self.DTYPE)

        for i in range(1, L - 1):
            ybar[i - 1] = (y[i + 1] - y[i - 1]) / (2 * delta)

        # Prepend and append the boundary differences
        ybar = np.concatenate(
            (
                np.array([(y[1] - y[0]) / delta], dtype=self.DTYPE),
                ybar,
                np.array([(y[-1] - y[-2]) / delta], dtype=self.DTYPE),
            )
        )

        return ybar

    def init_K_N(self, eIF: Optional[np.ndarray]) -> Tuple[int, int, np.ndarray]:
        """Initialize the frequency and get its size"""
        if eIF is not None:
            K, N = eIF.shape[0], eIF.shape[1]
        elif self.eIF is None:
            raise ValueError()
        else:
            K, N = self.K, self.N
            eIF = self.eIF
        return K, N, eIF

    def fit_transform(self, signal: np.ndarray, eIF: Optional[np.ndarray] = None):
        """
        Execute VNCMD algorithm for signal decomposition

        :param signal: the time domain signal (1D numpy array)  to be decomposed
        :param eIF: initial instantaneous frequency (IF) time series for all the signal modes; each row of eIF corresponds to the IF of each mode
        :return: - IFmset: the collection of the obtained IF time series of all the signal modes at each iteration
                 - smset: the collection of the obtained signal modes at each iteration
                 - IA: the finally estimated instantaneous amplitudes of the obtained signal modes
        """
        signal = signal.astype(self.DTYPE)
        K, N, eIF = self.init_K_N(eIF=eIF)
        t = np.arange(0, N, dtype=self.DTYPE) / self.fs

        # Get the improved second-order difference matrix
        oper = self.difference_matrix(N)
        opedoub = np.dot(oper.T, oper)

        # Used to store the demodulated orthogonal signal components
        sinm, cosm = np.zeros((K, N), dtype=self.DTYPE), np.zeros(
            (K, N), dtype=self.DTYPE
        )

        # Used to store the demodulated orthogonal signal components
        xm, ym = np.zeros((K, N), dtype=self.DTYPE), np.zeros((K, N), dtype=self.DTYPE)

        # Stores a collection of instantaneous frequency sequences of all signal modes obtained at each iteration
        IFsetiter = np.zeros((K, N, self.max_iter + 1), dtype=self.DTYPE)
        # Initialize the instantaneous frequency of the first iteration
        IFsetiter[:, :, 0] = eIF

        # Stores the collection of signal patterns obtained at each iteration
        ssetiter = np.zeros((K, N, self.max_iter + 1), dtype=self.DTYPE)

        # Lagrange multiplier, used to adjust constraints
        lamuda = np.zeros(N, dtype=self.DTYPE)

        # Initialize the variables defined above through loops
        for i in range(K):
            sinm[i, :] = np.sin(
                2 * np.pi * cumulative_trapezoid(eIF[i, :], t, initial=0),
                dtype=self.DTYPE,
            )
            cosm[i, :] = np.cos(
                2 * np.pi * cumulative_trapezoid(eIF[i, :], t, initial=0),
                dtype=self.DTYPE,
            )

            Bm = diags(
                sinm[i, :].T, offsets=0, shape=(N, N), dtype=self.DTYPE
            ).toarray()
            Bdoubm = diags(
                (sinm[i, :] ** 2).T, offsets=0, shape=(N, N), dtype=self.DTYPE
            ).toarray()

            Am = diags(
                cosm[i, :].T, offsets=0, shape=(N, N), dtype=self.DTYPE
            ).toarray()
            Adoubm = diags(
                (cosm[i, :] ** 2).T, offsets=0, shape=(N, N), dtype=self.DTYPE
            ).toarray()

            xm[i, :] = solve(2 / self.alpha * opedoub + Adoubm, np.dot(Am.T, signal))
            ym[i, :] = solve(2 / self.alpha * opedoub + Bdoubm, np.dot(Bm.T, signal))

            ssetiter[i, :, 0] = xm[i, :] * cosm[i, :] + ym[i, :] * sinm[i, :]

        # iteration counter
        iter = 0

        # Indicates the change difference between the current iteration and the previous iteration
        sDif = self.tol + 1

        # The cumulative sum
        sum_x, sum_y = np.sum(xm * cosm, axis=0), np.sum(ym * sinm, axis=0)
        sum_x, sum_y = np.squeeze(sum_x), np.squeeze(sum_y)

        # Start main loop
        while sDif > self.tol and iter <= self.max_iter:
            # Gradually increase betathr during the cycle but not exceed beta.
            betathr = 10 ** (iter / 36 - 10)
            if betathr > self.beta:
                betathr = self.beta

            u = self.projec(
                vec=signal - sum_x - sum_y - lamuda / self.alpha, var=self.var
            )

            for i in range(K):
                Bm = diags(sinm[i, :].T, offsets=0, shape=(N, N)).toarray()
                Bdoubm = diags((sinm[i, :] ** 2).T, offsets=0, shape=(N, N)).toarray()

                Am = diags(cosm[i, :].T, offsets=0, shape=(N, N)).toarray()
                Adoubm = diags((cosm[i, :] ** 2).T, offsets=0, shape=(N, N)).toarray()

                # remove the relevant component from the sum
                sum_x = sum_x - xm[i, :] * cosm[i, :]

                xm[i, :] = np.linalg.solve(
                    2 / self.alpha * opedoub + Adoubm,
                    np.dot(Am.T, (signal - sum_x - sum_y - u - lamuda / self.alpha).T),
                )
                interx = xm[i, :] * cosm[i, :]
                sum_x = sum_x + interx

                sum_y = sum_y - ym[i, :] * sinm[i, :]
                ym[i, :] = np.linalg.solve(
                    2 / self.alpha * opedoub + Bdoubm,
                    np.dot(Bm.T, (signal - sum_x - sum_y - u - lamuda / self.alpha).T),
                )
                sum_y = sum_y + ym[i, :] * sinm[i, :]

                # compute the derivative of the functions
                xbar = self.differ(xm[i, :], self.fs)
                ybar = self.differ(ym[i, :], self.fs)

                # obtain the frequency increment by arctangent demodulation
                deltaIF = (
                    (xm[i, :] * ybar - ym[i, :] * xbar)
                    / (xm[i, :] ** 2 + ym[i, :] ** 2)
                    / 2
                    / np.pi
                )
                # smooth the frequency increment by low pass filtering
                deltaIF = solve(2 / betathr * opedoub + eye(N), deltaIF.T)
                # update the IF
                eIF[i, :] = eIF[i, :] - 0.5 * deltaIF.T

                # update cos and sin functions
                sinm[i, :] = np.sin(2 * np.pi * cumtrapz(eIF[i, :], t, initial=0))
                cosm[i, :] = np.cos(2 * np.pi * cumtrapz(eIF[i, :], t, initial=0))

                # update sums
                sum_x = sum_x - interx + xm[i, :] * cosm[i, :]
                sum_y = sum_y + ym[i, :] * sinm[i, :]
                ssetiter[i, :, iter + 1] = xm[i, :] * cosm[i, :] + ym[i, :] * sinm[i, :]

            IFsetiter[:, :, iter + 1] = eIF

            # update the Lagrangian multiplier
            lamuda = lamuda + self.alpha * (u + sum_x + sum_y - signal)

            # restart scheme
            if norm(u + sum_x + sum_y - signal) > norm(signal):
                lamuda = np.zeros(shape=(1, len(t)))
                for i in range(K):
                    Bm = diags(sinm[i, :].T, offsets=0, shape=(N, N)).toarray()
                    Bdoubm = diags(
                        (sinm[i, :] ** 2).T, offsets=0, shape=(N, N)
                    ).toarray()

                    Am = diags(cosm[i, :].T, offsets=0, shape=(N, N)).toarray()
                    Adoubm = diags(
                        (cosm[i, :] ** 2).T, offsets=0, shape=(N, N)
                    ).toarray()

                    xm[i, :] = solve(
                        2 / self.alpha * opedoub + Adoubm, np.dot(Am.T, signal)
                    )
                    ym[i, :] = solve(
                        2 / self.alpha * opedoub + Bdoubm, np.dot(Bm.T, signal)
                    )

                    ssetiter[i, :, iter + 1] = (
                        xm[i, :] * cosm[i, :] + ym[i, :] * sinm[i, :]
                    )

                sum_x, sum_y = np.sum(xm * cosm, axis=1), np.sum(ym * sinm, axis=1)

            # compute the convergence index
            sDif = 0
            for i in range(K):
                sDif += (
                    norm(ssetiter[i, :, iter + 1] - ssetiter[i, :, iter])
                    / norm(ssetiter[i, :, iter])
                ) ** 2

            # Increase the number of push iterations
            iter += 1

            print(sDif)

        self.IFmset = IFsetiter[:, :, 0:iter]
        self.smset = ssetiter[:, :, 0:iter]
        self.IA = np.sqrt(xm**2 + ym**2)

        # print(self.IFmset.shape, self.smset.shape)
        return self.IFmset[:, :, -1], self.smset[:, :, -1], self.IA
