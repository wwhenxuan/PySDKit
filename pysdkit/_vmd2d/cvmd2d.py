# -*- coding: utf-8 -*-
"""
Created on 2025/02/12 11:06:23
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Union

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
            init: Union[str, int] = "radially",
            u_tol: Optional[float] = 1e-10,
            A_tol: Optional[float] = 1e-4,
            omega_tol: Optional[float] = 1e-10,
            max_iter: Optional[int] = 130,
            M: Optional[int] = 1,
            A_phase: Optional[np.ndarray] = np.array([100, 150]),
            random_seed: Optional[int] = 42,
    ) -> None:
        """

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
                     "graphical" = all omegas initalized by user graphical input
                     2*K*M = use given omega list for initialization; should be 2xKxM
        :param u_tol:
        :param A_tol:
        :param omega_tol:
        :param max_iter: maximum number of iterations
        :param M: number of submodes
        :param A_phase: 2D VMD - 2D-TV-VMD - 2D-TV-VMD-Seg scheduling
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

        # 存放初始化方式的列表
        self.init_omega_list = ["radially", "random", "graphical"]
        # 参数检验
        if self.init not in self.init_omega_list:
            raise ValueError("init should be one of {}".format(self.init_omega_list))

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Compact Variational Mode Decomposition for 2D Images (CVMD2D)"

    def _init_omega(self, omega: np.ndarray, image: np.ndarray):

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
                        omega[0, 0, k, m] = np.random

    def fit_transform(self, image: np.ndarray) -> np.ndarray:
        """执行信号分解算法"""

        # Resolution of image
        Hy, Hx = image.shape
        X, Y = np.meshgrid(np.arange(1, Hx + 1) / Hx, np.arange(1, Hy + 1) / Hy)

        # Spectral Domain discretization
        fx = 1 / Hx
        fy = 1 / Hx
        freqs_1 = X - 0.5 - fx
        freqs_2 = Y - 0.5 - fy

        # N iterations at most, 2 spatial coordinates, K modes, M submodes
        omega = np.zeros(shape=(self.max_iter, 2, self.K, self.M))

        # Storage matrices for (Fourier) modes. Iterations are not recorded
        u_hat = np.zeros(shape=(Hy, Hx, self.K, self.M))

        # copy the u_hat
        u = u_hat.copy()
        u_old = u.copy()
        v = u.copy()

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
