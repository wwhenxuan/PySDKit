# -*- coding: utf-8 -*-
"""
Created on 2025/02/06 18:26:39
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""

import numpy as np
from numpy import linalg

from typing import Optional

from pysdkit.utils import diagonal_average
from pysdkit.utils import lags_matrix


class SSA(object):
    """
    Singular Spectral Analysis (SSA) algorithm

    Zhigljavsky, Anatoly Alexandrovich.
    "Singular spectrum analysis for time series: Introduction to this special issue."
    Statistics and its Interface 3.3 (2010): 255-258.

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/58967-singular-spectrum-analysis-beginners-guide

    following steps of an SSA analysis:
    - creation of the trajectory matrix
    - calculation of the covariance matrix
    - eigendecomposition of the covariance matrix
    - resulting eigenvalues, eigenvectors
    - calculation of the principal components
    - reconstruction of the time series.
    """

    def __init__(
        self,
        K: int = 3,
        mode: str = "traj",
        lags: Optional[int] = None,
        averaging: Optional[bool] = True,
        extra_size: Optional[bool] = False,
    ) -> None:
        """
        Estimation the signal components based on the Singular Spectral Analysis (SSA) algorithm

        :param K: order of the model (number of valuable components, size of signal subspace)
        :param mode: the mode of lags matrix (i.e. trajectory (or caterpillar) matrix or its analouge), mode = {traj, full, covar, toeplitz, hankel}
        :param lags: number of lags in correlation function (x.shape[0]//2 by default)
        :param averaging: if True, then mean of each diagonal will be taken for diagonal averaging instead of just summarizing (True, by default)
        :param extra_size: if True, than near doubled size of output will be returned
        """
        if K < 1:
            raise ValueError("K must be a positive integer")

        self.K = K
        self.mode = mode
        self.lags = lags
        self.averaging = averaging
        self.extra_size = extra_size

        self.EPSILON = 1e-4

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Singular Spectral Analysis (SSA)"

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Execute the Singular Spectral Analysis (SSA) algorithm to perform signal decomposition

        :param signal: input signal of 1D ndarray
        :return: the decomposed results of IMFs
        """

        # Make sure the data type is correct
        signal = np.asarray(signal)
        if signal.ndim != 1 or signal.size < 2:
            raise ValueError("signal must be a 1D array with at least 2 samples")

        # Get the length of the signal
        seq_len = signal.shape[0]

        # for toeplitz and hankel N_lags always = N
        if self.lags is None:
            lags = max(seq_len // 2, 2)
        else:
            lags = int(self.lags)

        if lags < 2 or lags >= seq_len:
            raise ValueError(
                f"lags must satisfy 2 <= lags < len(signal); got lags={lags}, len={seq_len}"
            )

        # Whether to set the inversion
        reverse = False
        if self.mode in ["traj", "hankel", "trajectory", "caterpillar"]:
            reverse = True

        # Generate lag / trajectory matrix
        base = lags_matrix(signal, lags=lags, mode=self.mode)

        # R = X^H X; its eigenvectors are the right singular vectors of X
        R = np.dot(base.T, np.conj(base))

        # Hermitian eigendecomposition, sorted by eigenvalue magnitude (descending)
        es, ev = linalg.eigh(R)
        order = np.argsort(es)[::-1]
        es = es[order]
        ev = ev[:, order]

        # Singular values σ = sqrt(λ); floor tiny / negative numerical values
        singular = np.sqrt(np.maximum(es.real, 0.0)) + self.EPSILON

        n_components = min(self.K, ev.shape[1])
        # Array used to store decomposition results
        imfs = np.zeros(
            shape=(n_components, base.shape[0] + base.shape[1] - 1),
            dtype=np.result_type(signal.dtype, np.float64),
        )

        # Reconstruct each elementary component via X_i = (X v_i) v_i^H
        for i in range(n_components):
            v = ev[:, i]
            # Left singular vector direction (scaled): u σ = X v
            xv = np.dot(base, v)
            hankel = np.outer(xv, np.conj(v))

            diag = diagonal_average(
                hankel,
                reverse=reverse,
                samesize=self.extra_size,
                averaging=self.averaging,
            )

            imfs[i, : diag.size] = diag

        # Adjust the scale of the result after the iteration
        imfs = imfs[:, : diag.size]

        # For trajectory embedding, match the original series length by default
        if not self.extra_size and imfs.shape[1] >= seq_len:
            if self.mode in ["traj", "trajectory", "caterpillar", "covar", "valid"]:
                imfs = imfs[:, :seq_len]

        # Handling complex mappings
        if self.mode in ["traj", "trajectory", "caterpillar"]:
            imfs = np.conj(imfs)

        # Return real components when the input is real-valued
        if np.isrealobj(signal):
            imfs = np.real(imfs)

        return np.asarray(imfs)


if __name__ == "__main__":
    from pysdkit.data import test_emd
    from pysdkit.plot import plot_IMFs
    from matplotlib import pyplot as plt

    time, signal = test_emd()

    ssa = SSA(K=2, mode="traj")
    IMFs = ssa.fit_transform(signal)

    print(IMFs.shape)

    plot_IMFs(signal, IMFs)

    plt.show()
