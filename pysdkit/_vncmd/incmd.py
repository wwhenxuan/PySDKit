# -*- coding: utf-8 -*-
"""
Created on 2025/02/12 12:31:39
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from scipy import linalg
from scipy.integrate import cumtrapz
from scipy.sparse import spdiags, diags, identity, csc_matrix
from scipy.signal import welch
from scipy.sparse.linalg import inv as sparseinv
from scipy.sparse.linalg import spsolve

from typing import Optional, Tuple

from pysdkit.utils import get_timeline


class INCMD(object):
    """
    Iterative nonlinear chirp mode decomposition.

    A novel method termed the INCMD is proposed to deal with nonlinear data, which The combines the framework of the HHT with that of the VNCMD.
    Using the INCMD, intra-wave modulations can be captured with high accuracy and strong noise-robustness.
    Extracted modulation features by the INCMD greatly help to detect and identify nonlinear systems.

    Note: This algorithm is very sensitive to parameters. If you are not satisfied with the decomposition effect,
    please readjust the parameters `K`, `max_iter`, `mu` and `rho` to initialize the algorithm instance.

    Tu, Guowei, Xingjian Dong, Shiqian Chen, Baoxuan Zhao, Lan Hu, and Zhike Peng.
    “Iterative Nonlinear Chirp Mode Decomposition: A Hilbert-Huang Transform-like Method in Capturing Intra-Wave Modulations of Nonlinear Responses.”
    Journal of Sound and Vibration 485 (October 2020): 115571. https://doi.org/10.1016/j.jsv.2020.115571.

    Python code: https://github.com/sheadan/IterativeNCMD
    """

    def __init__(
        self,
        K: int = 2,
        max_iter: Optional[int] = 8000,
        rho: Optional[float] = 1e-3,
        mu: Optional[float] = 1e-4,
        tol: Optional[float] = 1e-6,
    ) -> None:
        """
        Initialize the parameters of the Iterative nonlinear chirp mode decomposition algorithm.

        Note: This algorithm is very sensitive to parameters. If you are not satisfied with the decomposition effect,
        please readjust the parameters `K`, `max_iter`, `mu` and `rho` to initialize the algorithm instance.
        :param K: The number of intrinsic mode functions obtained by decomposition,
                  that is the number of sub-signals obtained by decomposition.
        :param max_iter: Maximum number of iterations within one mode decomposition
        :param rho: To initialize the g-mode prefactor
        :param mu: The mean value used to initialize the filter
        :param tol: Maximum tolerance of the algorithm
        """
        self.K = K
        self.max_iter = max_iter
        self.rho = rho
        self.mu = mu
        self.tol = tol

    def __call__(
            self,
            signal: np.ndarray,
            time: Optional[np.ndarray] = None,
            K: Optional[int] = None,
            return_all: Optional[bool] = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal, time=time, K=K, return_all=return_all)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Iterative nonlinear chirp mode decomposition (INCMD)"

    @staticmethod
    def _get_D_matrix(signal: np.ndarray) -> np.ndarray:
        """Get a D matrix that's sized n x n for a given signal of length n"""
        n = len(signal)
        superdiag = np.ones(n)
        diagonal = -2 * np.ones(n)
        diagonal[-1] = -1
        diagonal[0] = -1
        D = spdiags(
            np.asarray([diagonal, superdiag, superdiag]),
            np.array([0, -1, 1]),
            n,
            n,
            format="csc",
        )
        return D

    def _NCMD(
        self, signal: np.ndarray, time: np.ndarray, g_factor: np.ndarray, f_filter: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute a nonlinear frequency modulation mode decomposition algorithm"""

        # Initial frequency estimate by welch method
        f_0 = self._welch_method_estimate(signal=signal, time=time)

        # Initialize phi matrices
        phi_1 = self._compute_new_phi_1(new_f=f_0, time=time)
        phi_2 = self._compute_new_phi_2(new_f=f_0, time=time)

        # Initialize estimates for gd1 and gd2
        g_d1 = self._compute_new_g(signal=signal, phi=phi_1, g_factor=g_factor)
        g_d2 = self._compute_new_g(signal=signal, phi=phi_2, g_factor=g_factor)

        # Initialize g_0 computation
        g_0 = self._compute_mode(phi_1=phi_1, g_d1=g_d1, phi_2=phi_2, g_d2=g_d2)

        # Set the g_previous and f_previous variables
        g_previous = g_0
        f_previous = f_0

        # Iterate and refine NCM
        for i in range(self.max_iter):
            # Update the estimates for gd1 and gd2
            g_d1 = self._compute_new_g(signal=signal, phi=phi_1, g_factor=g_factor)
            g_d2 = self._compute_new_g(signal=signal, phi=phi_2, g_factor=g_factor)

            # Update the frequency estimate
            f_bar = self._compute_new_f(
                g1=g_d1, g2=g_d2, old_f=f_previous, time=time, f_filter=f_filter
            )

            # Initialize phi matrices
            phi_1 = self._compute_new_phi_1(new_f=f_bar, time=time)
            phi_2 = self._compute_new_phi_2(new_f=f_bar, time=time)

            # Compute the new mode g^i
            g = self._compute_mode(phi_1=phi_1, g_d1=g_d1, phi_2=phi_2, g_d2=g_d2)

            # Compute change in the mode definition
            loss = linalg.norm(g - g_previous, ord=2) ** 2
            loss /= linalg.norm(g_previous, ord=2) ** 2

            # If the change is less than tol
            if loss < self.tol:
                break

            # Update previous-iteration variables
            g_previous = g
            f_previous = f_bar

        return f_bar, g, g_d1, g_d2

    @staticmethod
    def _welch_method_estimate(signal: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Create an initial estimate on the frequency vector using the Welch PSD method"""
        nperseg = int(signal.shape[0] / 16)
        nperseg += 0.5 * nperseg
        f, Pxx = welch(signal, 1 / (time[1] - time[0]), nperseg=nperseg)
        init_guess = f[np.argmax(Pxx)]
        return init_guess * np.ones(signal.shape)

    @staticmethod
    def _compute_new_phi_1(new_f: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Compute new phi_1"""
        integrated_f = cumtrapz(new_f, time, initial=0)
        phi_diag = np.cos(2 * np.pi * integrated_f)
        return diags(phi_diag)

    @staticmethod
    def _compute_new_phi_2(new_f: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Compute new phi_2"""
        integrated_f = cumtrapz(new_f, time, initial=0)
        phi_diag = np.sin(2 * np.pi * integrated_f)
        return diags(phi_diag)

    @staticmethod
    def _compute_new_g(
        signal: np.ndarray, phi: np.ndarray, g_factor: np.ndarray
    ) -> np.ndarray:
        # Compute the phi^T phi term (which, for diagonal phi is diag(phi) squared)
        phi_diagonal = phi.data[0, :]
        phi_square_diag = phi_diagonal * phi_diagonal
        # Compute the new mode
        new_G_A = g_factor + diags(phi_square_diag, format="csc")
        new_G_b = phi @ signal
        new_G = spsolve(new_G_A, new_G_b)
        # new_G = new_G.flatten() # Commented this out because flatten seemed to densify this sparse matrix (bad for memory)
        return new_G

    @staticmethod
    def _compute_mode(
        phi_1: np.ndarray, g_d1: np.ndarray, phi_2: np.ndarray, g_d2: np.ndarray
    ) -> np.ndarray:
        """Compute and return the estimated signal from mode g_k"""
        # mode = np.dot(phi_1, g_d1) + np.dot(phi_2, g_d2)
        # mode = phi_1 @ g_d1 + phi_2 @ g_d2
        mode = phi_1.data[0, :] * g_d1 + phi_2.data[0, :] * g_d2
        return mode

    @staticmethod
    def _compute_new_f(
        g1: np.ndarray,
        g2: np.ndarray,
        old_f: np.ndarray,
        time: np.ndarray,
        f_filter: np.ndarray,
    ) -> np.ndarray:
        """Get a new filter based on the input of the original filter"""
        dt = time[1] - time[0]
        # Compute gradients
        dg1dt = np.gradient(g1, dt)
        dg2dt = np.gradient(g2, dt)
        # Compute frequency step
        delta_f_num = (1 / (2 * np.pi)) * (g2 * dg1dt - g1 * dg2dt)
        delta_f_denom = g1**2 + g2**2
        delta_f = delta_f_num / delta_f_denom
        # And compute the frequency step vector
        step = f_filter @ delta_f
        # Finally, compute the vector of frequency steps
        new_f = old_f.reshape(-1, 1) + step.reshape(-1, 1)
        new_f = np.asarray(new_f).flatten()

        return new_f

    def fit_transform(
        self,
        signal: np.ndarray,
        time: Optional[np.ndarray] = None,
        K: Optional[int] = None,
        return_all: Optional[bool] = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """
        Start executing the INCMD algorithm for signal decomposition.

        Note: This algorithm is very sensitive to parameters. If you are not satisfied with the decomposition effect,
        please readjust the parameters `K`, `max_iter`, `mu` and `rho` to initialize the algorithm instance.
        :param signal: The input numpy ndarray 1D univariate signal
        :param time: Timestamp array for the input signal,
                     initialized by the length of the signal if not specified explicitly
        :param K: The number of modes to be decomposed. If not explicitly specified in the `fit_transform` method,
                  the default parameter when creating the instance is used.
        :param return_all: Whether to return all information about the decomposition process
        :return: The intrinsic mode function obtained by signal decomposition
        """
        # Get the length of the input signal
        seq_len = signal.shape[0]

        # Determine whether the timestamp array is valid
        if time is None:
            time = get_timeline(range_max=seq_len, dtype=signal.dtype)

        # Copy the original input signal
        original_signal = signal.copy()

        # The number of sub-signals to be decomposed
        modes = self.K if K is None else K

        # Generate D matrix (modified second-order differences)
        D = self._get_D_matrix(signal=original_signal)

        # Compute matrix product D^T D, which is used for a number of computations
        DTD = csc_matrix(D.T @ D)

        # Compute the frequency-step filter and g-mode prefactor:
        n = D.shape[0]

        f_filter = sparseinv(((2 / self.mu) * DTD + identity(n, format="csc")))
        g_factor = (1 / self.rho) * DTD

        # Results lists
        f_bars = []
        gs = []
        g_d1s = []
        g_d2s = []

        # Compute NCM for each prescribed mode number:
        for i in range(modes):
            f_bar, g, g_d1, g_d2 = self._NCMD(
                signal=original_signal, time=time, g_factor=g_factor, f_filter=f_filter
            )

            # Save the results
            f_bars.append(f_bar.copy())
            gs.append(g.copy())
            g_d1s.append(g_d1.copy())
            g_d2s.append(g_d2.copy())

            # Subtract the current mode from the data
            original_signal -= g

        # Return the results
        if return_all is True:
            return np.array(gs), np.array(f_bars), np.array(g_d1s), np.array(g_d2s)

        return np.array(gs)


if __name__ == "__main__":
    from pysdkit.data import test_emd, test_univariate_signal
    from pysdkit.plot import plot_IMFs, plot_signal
    from matplotlib import pyplot as plt

    t, s = test_univariate_signal(case=1, sampling_rate=512)

    incmd = INCMD(K=3, rho=1e-3, mu=1e-4, tol=1e-6, max_iter=6000)
    IMFs = incmd.fit_transform(signal=s.copy())

    print(IMFs.shape)

    plot_IMFs(signal=s, IMFs=IMFs)
    plt.show()
