# -*- coding: utf-8 -*-
"""
Created on 2025/02/02 17:00:47
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
from pysdkit.utils import fft2d, ifft2d, fftshift, ifftshift

from typing import Optional, Tuple


class VMD2D(object):
    """
    Variational Mode Decomposition for 2D Image

    Konstantin, Dragomiretskiy, and Dominique Zosso. "Two-dimensional variational mode decomposition."
    Energy Minimization Methods in Computer Vision and Pattern Recognition. Vol. 8932. 2015.
    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/45918-two-dimensional-variational-mode-decomposition?s_tid=srchtitle
    """

    def __init__(
        self,
        K: int,
        alpha: int,
        tau: Optional[float] = 0.0,
        DC: Optional[bool] = False,
        init: Optional[str] = "zero",
        max_iter: Optional[int] = 3000,
        tol: Optional[float] = 1e-7,
        random_seed: Optional[int] = 42,
    ) -> None:
        """
        :param K: the number of modes to be recovered
        :param alpha: the balancing parameter for data fidelity constraint
        :param tau: time-step of dual ascent ( pick 0 for noise-slack )
        :param DC: true, if the first mode is put and kept at DC (0-freq)
        :param init: zero means all omegas start at 0 and random means all omegas start initialized randomly
        :param max_iter: the maximum number of iterations
        """
        self.K = K  # 分解子信号的数目
        # 调控分解效果的参数
        self.alpha, self.tau, self.DC, self.init = alpha, tau, DC, init
        # 算法停止迭代的准则
        self.max_iter, self.tol = max_iter, tol
        # 创建一个随机数生成器用于初始化
        self.rng = np.random.RandomState(random_seed)

    def __call__(
            self,
            img: np.ndarray,
            K: Optional[int] = None,
            return_all: Optional[bool] = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """allow instances to be called like functions"""
        return self.fit_transform(img=img, K=K, return_all=return_all)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Variational mode decomposition for 2D Images (VMD2D)"

    def init_omega(self, K: int) -> np.ndarray:
        """Initialization of omega_k"""
        omega = np.zeros(shape=(self.max_iter, 2, K))
        if self.init == "zero":
            # spread omegas radially
            if self.DC is True:
                # if DC, keep first mode at 0, 0
                maxK = K + 1
            else:
                maxK = K
            for k in range(0, maxK):
                omega[0, 0, k] = 0.25 * np.cos(np.pi * (k - 1) / maxK)
                omega[0, 1, k] = 0.25 * np.sin(np.pi * (k - 1) / maxK)
        elif self.init == "random":
            # random on half-plane
            for k in range(0, K):
                omega[0, 0, k] = self.rng.rand() - 1 / 2
                omega[0, 1, k] = self.rng.rand() / 2
            if self.DC is True:
                # DC component (if expected)
                omega[0, 0, 0] = 0
                omega[0, 1, 0] = 0
        else:
            raise ValueError("init must be 'zero' or 'random'")
        return omega

    def fit_transform(
        self,
        img: np.ndarray,
        K: Optional[int] = None,
        return_all: Optional[bool] = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Image decomposition using VMD algorithm
        :param img: the input image of 2D ndarray in numpy
        :param K: the maximum number of modes to be decomposed
        :param return_all: Whether to return all results of the algorithm, False only return the collection of decomposed modes,
                           True plus the spectra of the modes and the estimated mode center-frequencies
        :return: The extracted modes from the signals / images
        """
        K = K if K is not None else self.K

        # Resolution of image
        Hy, Hx = img.shape
        X, Y = np.meshgrid(np.arange(1, Hx + 1) / Hx, np.arange(1, Hy + 1) / Hy)

        # Spectral domain discretization
        fx = 1 / Hx
        fy = 1 / Hy
        freqs_1 = X - 0.5 - fx
        freqs_2 = Y - 0.5 - fy

        # For future generalizations: alpha might be individual for each mode
        Alpha = self.alpha * np.ones(K)

        # Construct img and image_hat
        img_hat = fftshift(fft2d(img))

        # Storage matrices for (Fourier) modes. All iterations are not recorded.
        u_hat = np.zeros(shape=(Hy, Hx, K))  # 用于记录每个模态的分解结果
        u_hat_old = u_hat.copy()
        sum_uk = 0

        # Storage matrices for (Fourier) Lagrange Multiplier
        mu_hat = np.zeros(shape=(Hy, Hx))

        # N iterations at most, 2 spatial coordinates, K clusters
        omega = self.init_omega(K)

        ##### Main loop for iterative updates #####

        # Stopping criteria tolerances
        uDiff = self.tol + np.spacing(1)
        omegaDiff = self.tol + np.spacing(1)

        # first run
        n = 0

        # run until convergence or max number of iterations
        while (uDiff > self.tol or omegaDiff > self.tol) and n < self.max_iter - 1:
            # first things first
            k = 0

            # compute the halfplane mask for the 2D "analytic signal"
            HilbertMask = (
                np.sign(freqs_1 * omega[n, 0, k] + freqs_2 * omega[n, 1, k]) + 1
            )

            # update first mode accumulator
            sum_uk = u_hat[:, :, -1] + sum_uk - u_hat[:, :, k]

            # update first mode's spectrum through wiener filter (on half plane)
            u_hat[:, :, k] = ((img_hat - sum_uk - mu_hat[:, :] / 2) * HilbertMask) / (
                1
                + Alpha[k]
                * ((freqs_1 - omega[n, 0, k]) ** 2 + (freqs_2 - omega[n, 1, k]) ** 2)
            )

            # update first mode's central frequency as spectral center of gravity
            if self.DC is False:
                omega[n + 1, 0, k] = np.sum(
                    np.sum(freqs_1 * (np.abs(u_hat[:, :, k]) ** 2))
                ) / np.sum(np.sum(np.abs(u_hat[:, :, k]) ** 2))
                omega[n + 1, 1, k] = np.sum(
                    np.sum(freqs_2 * (np.abs(u_hat[:, :, k]) ** 2))
                ) / np.sum(np.sum(np.abs(u_hat[:, :, k]) ** 2))

                # Keep omegas on same halfplane
                if omega[n + 1, 1, k] < 0:
                    omega[n + 1, :, k] = -omega[n + 1, :, k]

            # recover full spectrum from analytic signal
            u_hat[:, :, k] = fftshift(
                fft2d(np.real(ifft2d(ifftshift(np.squeeze(u_hat[:, :, k])))))
            )

            # work on other modes
            for k in range(1, K):
                # recompute Hilbert mask
                HilbertMask = (
                    np.sign(freqs_1 * omega[n, 0, k] + freqs_2 * omega[n, 1, k]) + 1
                )

                # update accumulator
                sum_uk = u_hat[:, :, k - 1] + sum_uk - u_hat[:, :, k]

                # update signal spectrum
                u_hat[:, :, k] = (
                    (img_hat - sum_uk - mu_hat[:, :] / 2) * HilbertMask
                ) / (
                    1
                    + Alpha[k]
                    * (
                        (freqs_1 - omega[n, 0, k]) ** 2
                        + (freqs_2 - omega[n, 1, k]) ** 2
                    )
                )

                # update signal frequencies
                omega[n + 1, 0, k] = np.sum(
                    np.sum(freqs_1 * (np.abs(u_hat[:, :, k]) ** 2))
                ) / np.sum(np.sum(np.abs(u_hat[:, :, k]) ** 2))
                omega[n + 1, 1, k] = np.sum(
                    np.sum(freqs_2 * (np.abs(u_hat[:, :, k]) ** 2))
                ) / np.sum(np.sum(np.abs(u_hat[:, :, k]) ** 2))

                # Keep omegas on same halfplane
                if omega[n + 1, 1, k] < 0:
                    omega[n + 1, :, k] = -omega[n + 1, :, k]

                # recover full spectrum from analytic signal
                u_hat[:, :, k] = fftshift(
                    fft2d(np.real(ifft2d(ifftshift(np.squeeze(u_hat[:, :, k])))))
                )

            # Gradient ascent for augmented Lagrangian
            mu_hat[:, :] = mu_hat[:, :] + self.tau * (np.sum(u_hat, axis=2) - img_hat)

            # increment iteration counter
            n += 1

            # whether convergence?
            uDiff = np.spacing(1)
            omegaDiff = np.spacing(1)

            for k in range(0, K):
                omegaDiff = omegaDiff + np.sum(
                    np.sum(np.abs(omega[n, :, :] - omega[n - 1, :, :]) ** 2)
                )
                uDiff = uDiff + np.sum(
                    np.sum(
                        1
                        / (Hx * Hy)
                        * (u_hat[:, :, k] - u_hat_old[:, :, k])
                        * np.conj((u_hat[:, :, k] - u_hat_old[:, :, k]))
                    )
                )

            uDiff = np.abs(uDiff)

            # update the u_hat old
            u_hat_old = u_hat.copy()

        ##### Signal or Image Reconstruction #####

        # Inverse Fourier Transform to compute (spatial) modes
        u = np.zeros(shape=(Hy, Hx, K))
        for k in range(0, K):
            u[:, :, k] = np.real(ifft2d(ifftshift(np.squeeze(u_hat[:, :, k]))))

        # Should the omega-history be returned, or just the final results?
        if return_all:
            return u, u_hat, omega
        else:
            return u


if __name__ == "__main__":
    from pysdkit.data import test_grayscale

    img = test_grayscale()

    vmd2d = VMD2D(
        K=5, alpha=5000, tau=0.25, DC=True, init="random", tol=1e-6, max_iter=3000
    )
    u = vmd2d.fit_transform(img)
    print(u.shape)

    from matplotlib import pyplot as plt

    plt.imshow(img, cmap="gray")
    plt.show()

    for i in range(u.shape[2]):
        plt.imshow(u[:, :, i], cmap="gray")
        plt.show()
