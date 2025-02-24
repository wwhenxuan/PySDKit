# -*- coding: utf-8 -*-
"""
Created on 2025/02/12 11:06:23
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from scipy.linalg import norm
from scipy.sparse import coo_matrix, csr_matrix

from typing import Optional, Union, Tuple, Any

from pysdkit.utils import fft2d, ifft2d, fftshift, ifftshift


class CVMD2D(object):
    """
    Compact Variational Mode Decomposition for 2D Images

    Spatially Compact and Spectrally Sparse Image Decomposition and Segmentation.
    Decomposing multidimensional signals, such as images, into spatially compact,
    potentially overlapping modes of essentially wavelike nature makes these components accessible for further downstream analysis.
    This decomposition enables space-frequency analysis, demodulation, estimation of local orientation, edge and corner detection,
    texture analysis, denoising, inpainting, or curvature estimation.

    [1] D. Zosso, K. Dragomiretskiy, A.L. Bertozzi, P.S. Weiss, Two-Dimensional Compact Variational Mode Decomposition,
    Journal of Mathematical Imaging and Vision, 58(2):294–320, 2017. DOI:10.1007/s10851-017-0710-z.

    [2] K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition,
    IEEE Trans. on Signal Processing, 62(3):531-544, 2014. DOI:10.1109/TSP.2013.2288675.

    [3] K. Dragomiretskiy, D. Zosso, Two-Dimensional Variational Mode Decomposition,
    EMMCVPR 2015, Hong Kong, LNCS 8932:197-208, 2015. DOI:10.1007/978-3-319-14612-6_15.

    MATLAB code: https://ww2.mathworks.cn/matlabcentral/fileexchange/67285-two-dimensional-compact-variational-mode-decomposition-2d-tv-vmd?s_tid=FX_rc2_behav
    """

    def __init__(
            self,
            K: Optional[int] = 5,
            alpha: Optional[int] = 1000,
            beta: Optional[float] = 0.5,
            gamma: Optional[int] = 500,
            delta: Optional[np.ndarray] = np.inf,
            rho: Optional[int] = 10,
            rho_k: Optional[int] = 10,
            tau: Optional[float] = 0.0,
            tau_k: Optional[float] = 2.5,
            t: Optional[float] = 1.5,
            DC: Optional[bool] = False,
            init: Union[str, int] = "uniform",
            u_tol: Optional[float] = 1e-10,
            A_tol: Optional[float] = 1e-4,
            omega_tol: Optional[float] = 1e-10,
            max_iter: Optional[int] = 130,
            M: Optional[int] = 1,
            A_phase: Optional[np.ndarray] = np.array([100, 150]),
            random_seed: Optional[int] = 42,
    ) -> None:
        """
        Please note that this algorithm is very sensitive to hyperparameters.
        When using it, please try multiple configurations based on the input image features.
        You can try the VMD2D algorithm instead of the complex hyperparameter configuration of the algorithm.
        :param K: the number of modes to be recovered
        :param alpha: narrowbandedness of subsignals coefficient (scalar)
        :param beta: L1 penalty coefficient of spatial mode support (scalar)
        :param gamma: Spatial support TV-term: heat diffusion coefficient
        :param delta: Artifact classification threshold (inf for no artifacts)
        :param rho: data fidelity coefficient
        :param rho_k: u-v splitting coefficient
        :param tau: time-step of dual ascent for data ( pick 0 for noise-slack )
        :param tau_k: time-step of dual ascent for u-v splitting
        :param t: spatial support TV-term: time-step of ODE/PDE
        :param DC: true, if the first mode is put and kept at DC (0-freq)
        :param init: "radially" = all omegas start initialized radially uniformly
                     "random" = all omegas start initialized randomly on half plane
                     2*K*M = use given omega list for initialization; should be 2xKxM
        :param u_tol: Tolerance for u convergence
        :param A_tol: Tolerance for A convergence
        :param omega_tol: Tolerance for omega convergence
        :param max_iter: maximum number of iterations
        :param M: number of submodes
        :param A_phase: 2D VMD - 2D-TV-VMD - 2D-TV-VMD-Seg scheduling
                        % iterations 1:a --> 2D VMD
                        % iterations a:b --> 2D TV VMD
                        % iterations b:end --> 2D TV VMD Segmentation
        :param: random_seed: random seed for the omega initialization
        """
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rho = rho
        self.rho_k = rho_k
        self.tau = tau
        self.tau_k = tau_k
        self.t = t
        self.DC = DC
        self.init = init
        self.u_tol = u_tol
        self.A_tol = A_tol
        self.omega_tol = omega_tol
        self.max_iter = max_iter
        self.M = M
        self.A_phase = A_phase

        # A list of initialization methods
        self.init_omega_list = ["uniform", "random"]
        # Parameter Test
        if self.init not in self.init_omega_list:
            raise ValueError("init should be one of {}".format(self.init_omega_list))

        # Create a generator based on a random number seed
        self.rng = np.random.default_rng(seed=random_seed)

    def __call__(self, image: np.ndarray, return_all: Optional[str] = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(image=image, return_all=return_all)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Compact Variational Mode Decomposition for 2D Images (CVMD2D)"

    def _init_omega(self) -> np.ndarray:
        """Initialization of omega_k"""
        # N iterations at most, 2 spatial coordinates, K modes, M submodes
        omega = np.zeros(shape=(self.max_iter, 2, self.K, self.M))

        if isinstance(self.init, str):
            # 如果输入是字符串则通过已有的三种方法进行初始化
            if self.init == "radially":
                # Radially Uniform
                # if DC, Keep first mode at 0, 0
                if self.DC is True:
                    maxK = self.K - 1
                else:
                    maxK = self.K
                radius = 0.3
                for k in range(int(self.DC), int(self.DC) + maxK):
                    for m in range(0, self.M):
                        omega[0, 0, k, m] = radius * np.cos(np.pi * (k - 1 + (m - 1) * maxK) / maxK / self.M)
                        omega[0, 1, k, m] = radius * np.sin(np.pi * (k - 1 + (m - 1) * maxK) / maxK / self.M)
            elif self.init == "random":
                # Random on Half-Plane
                for k in range(0, self.K):
                    for m in range(0, self.M):
                        omega[0, 0, k, m] = self.rng.random() - 1 / 2
                        omega[0, 1, k, m] = self.rng.random() / 2
                # DC component (if expected)
                if self.DC is True:
                    omega[0, 0, 0, :] = 0.0
                    omega[0, 1, 0, :] = 0.0
            else:
                raise ValueError("init should be one of {}".format(self.init_omega_list))
        elif np.size(self.init) == 2 * self.K * self.M:
            # use given omega list for initialization; should be 2xKxM
            if np.size(self.init, 0) != 2 or np.size(self.init, 1) != self.K:
                raise ValueError("init parameter has inappropriate size")
            omega[0, :, :, :] = self.init
        else:
            raise ValueError("Wrong input for `init`")

        return omega

    def fit_transform(self, image: np.ndarray, return_all: Optional[str] = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | np.ndarray:
        """
        Execute the signal decomposition algorithm for 2D images

        Please note that this algorithm is very sensitive to hyperparameters.
        Please try multiple configurations based on the input image features.
        You can try the VMD2D algorithm instead of the complex hyperparameter configuration of the algorithm.
        """

        # Resolution of image
        Hy, Hx = image.shape
        X, Y = np.meshgrid(np.arange(1, Hx + 1) / Hx, np.arange(1, Hy + 1) / Hy)

        # Spectral Domain discretization
        fx = 1 / Hx
        fy = 1 / Hx
        freqs_1 = X - 0.5 - fx
        freqs_2 = Y - 0.5 - fy

        # Storage matrices for (Fourier) modes. Iterations are not recorded
        u_hat = np.zeros(shape=(Hy, Hx, self.K, self.M))

        # copy the u_hat
        u = u_hat.copy()
        u_old = u.copy()
        v = u.copy()

        print("origin v", v.shape)

        # Augmented Lagrangian Variables

        # linking variable u/v   ~rho_k
        lambda_k = u.copy()
        # data fidelity          ~rho
        lambda_d = np.zeros(shape=(Hy, Hx))

        # Spatial support variables
        A = np.ones(shape=(Hy, Hx, self.K))
        A_old = A.copy()

        # Artifact map
        X = np.zeros(shape=(Hy, Hx))

        # Initialization of omega_k
        omega = self._init_omega()

        ##### Main Loop for iterative updates #####

        # Stopping criteria tolerances
        uDiff = np.inf
        ADiff = np.inf
        omegaDiff = np.inf

        # Stores the sum of A_j*v_j for all j not equal to k
        sum_Avk = 0

        # Init the interation counter
        n = 0

        # Main loop - run until convergence or max number of iterations
        while ((uDiff > self.u_tol or ADiff > self.A_tol or omegaDiff > self.omega_tol) and n < self.max_iter - 1) or n <= np.max(np.isfinite(self.A_phase) * self.A_phase):
            # Mode
            for k in range(0, self.K):
                # Submodes
                for m in range(0, self.M):

                    # Compute the halfplane spectral mask for the 2D "analytic signal"
                    HilbertMask = (np.sign(freqs_1 * omega[n, 0, k, m] + freqs_2 * omega[n, 1, k, m]) + 1)

                    # Update accumulator
                    if m == 0:
                        if k == 0:
                            sum_Avk = sum_Avk + A[:, :, -1] * v[:, :, -1, -1] - A[:, :, 0] * v[:, :, 0, 0]
                        else:
                            sum_Avk = sum_Avk + A[:, :, k - 1] * v[:, :, k - 1, -1] - A[:, :, k] * v[:, :, k, 0]
                    else:
                        sum_Avk = sum_Avk + A[:, :, k] * v[:, :, k, m - 1] - A[:, :, k] * v[:, :, k, m]

                    # Update v (time domain averaging)
                    v[:, :, k, m] = (self.rho_k * u[:, :, k, m] + lambda_k[:, :, k, m] + self.rho * A[:, :, k] * (image - sum_Avk + lambda_d / self.rho) * (1 - X)) / (self.rho_k + self.rho * (1 - X) * A[:, :, k] ** 2)

                    # Update u_hat (analytic signal spectrum via Wiener filter)
                    u_hat[:, :, k, m] = (fftshift(fft2d(self.rho_k * v[:, :, k, m] - lambda_k[:,:,k,m])) * HilbertMask) / (self.rho_k + 2 * self.alpha * ((freqs_1 - omega[n, 0, k, m]) ** 2 + (freqs_2 - omega[n, 1, k, m]) ** 2))

                    # Update center frequencies (first mode is kept at omega = 0 if DC = 1)
                    if not self.DC or k > 0:

                        # Update signal frequencies as center of mass of power spectrum
                        omega[n + 1, 0, k, m] = np.sum(np.sum(freqs_1 * (np.abs(u_hat[:, :, k, m]) ** 2))) / np.sum(np.sum(np.abs(u_hat[:, :, k, m]) ** 2))
                        omega[n + 1, 1, k, m] = np.sum(np.sum(freqs_2 * (np.abs(u_hat[:, :, k, m]) ** 2))) / np.sum(np.sum(np.abs(u_hat[:, :, k, m]) ** 2))

                        # Keep omegas on same halfplane (top half)
                        if omega[n + 1, 1, k, m] < 0:
                            omega[n + 1, :, k, m] = -omega[n + 1, :, k, m]

                    # recover full spectrum (and signal) from analytic signal spectrum
                    u[:, :, k, m] = np.real(ifft2d(ifftshift(np.squeeze(u_hat[:, :, k, m]))))

                # No MBO/TV-term propagation in phase I (2D VMD, only)

                # Individual MBO for TV-term Propagation in phase II (2D TV VMD)
                if self.A_phase[0] <= n + 1 < self.A_phase[1]:
                    # Reconstruction Fidelity + Area Penalty + Segmentation Penalty
                    A[:, :, k] = A[:, :, k] + self.t * (-self.beta + 2 * self.rho * np.sum(v[:, :, k, :], axis=2) * (image - np.sum(A * np.sum(v, axis=3), axis=2) + A[:, :, k] * np.sum(v[:, :, k, :], axis=2) + lambda_d / self.rho) * (1 - X))
                    A[:, :, k] = A[:, :, k] / (1 + self.t * 2 * self.rho * (1 - X) * np.sum(v[:, :, k, :], axis=2) ** 2)

                    # Project to characteristic range
                    A[A > 1] = 1
                    A[A < 0] = 0

                    # Propagate by heat equation
                    A[:, :, k] = ifft2d(fft2d(A[:, :, k]) / (1 + self.t * self.gamma * ifftshift(freqs_1 ** 2 + freqs_2 ** 2)))

                    # individual MBO thresholding [0,1] (no segmentation constraint)
                    A[:, :, k] = (A[:, :, k] >= 0.5)

            # Joint MBO prop. with segmentation, "Winner Takes All", phase III
            if n + 1 >= self.A_phase[1]:
                sum_Avk = np.sum(A * np.sum(v, axis=3), axis=2)
                for k in range(0, self.K):
                    A[:, :, k] = A[:, :, k] + self.t * (-self.beta + 2 * self.rho * np.sum(v[:, :, k, :], axis=2) * (image - sum_Avk + A[:, :, k] * np.sum(v[:, :, k, :], axis=2) + lambda_d / self.rho))
                    A[:, :, k] = A[:, :, k] / (1 + self.t * 2 * self.rho * np.sum(v[:, :, k, :], axis=2) ** 2)
                    A[:, :, k] = ifft2d(fft2d(A[:, :, k]) / (1 + self.t * self.gamma * ifftshift(freqs_1 ** 2 + freqs_2 ** 2)))

                # Reshape A to a 2D array of shape [Hx*Hy, K]
                A_reshaped = A.reshape(Hx * Hy, self.K)

                # Find the index of the maximum value along the second axis (axis=1)
                I = np.argmax(A_reshaped, axis=1)

                # Create a sparse matrix with 1s at the positions specified by I
                # Use scipy.sparse.csr_matrix to create a sparse matrix
                row_indices = np.arange(Hx * Hy)  # Row indices (1:Hx*Hy)
                data = np.ones(Hx * Hy)  # All values are 1
                sparse_matrix = csr_matrix((data, (row_indices, I)), shape=(Hx * Hy, self.K))

                # Convert the sparse matrix to a dense array and reshape back to 3D
                A = sparse_matrix.toarray().reshape(Hx, Hy, self.K)

            # Artifact thresholding
            DF = (image - np.sum(A * np.sum(v, axis=3), axis=2))

            # Update the X array
            X = DF ** 2 >= self.delta

            # data fidelity dual ascent
            lambda_d = lambda_d + self.tau * DF

            # Update Lagrangian multiplier variables via gradient ascent
            lambda_k = lambda_k + self.tau * (u - v)

            # Update counter
            n += 1

            # Tolerance calculation for stopping criteria
            uDiff = norm(u - u_old) ** 2 / norm(u ** 2 / (Hx * Hy))
            ADiff = norm(A.ravel() - A_old.ravel(), ord=1) / (Hx * Hy)

            # Storage of n-th iteration
            u_old = u.copy()
            A_old = A.copy()

        omega = omega[n, :, :, :]

        # Whether to return all the information of decomposition
        if return_all is True:
            return u, v, omega, A, X
        return u


if __name__ == '__main__':
    from pysdkit.data import test_grayscale

    img = test_grayscale()

    vmd2d = CVMD2D(
        K=5, alpha=1000, tau=2.5, DC=True, init="uniform", max_iter=130, M=1, A_phase=np.array([100, np.inf]),
        beta=0.5, gamma=500, rho=10, rho_k=10, tau_k=2.5, t=1.5
    )
    u = vmd2d.fit_transform(img)
    print(u.shape)

    from matplotlib import pyplot as plt

    plt.imshow(img, cmap="gray")
    plt.show()

    for i in range(u.shape[2]):
        plt.imshow(u[:, :, i], cmap="gray")
        plt.show()


