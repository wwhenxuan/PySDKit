# -*- coding: utf-8 -*-
"""
Nonuniformly Sampled Trivariate Empirical Mode Decomposition (NS-TEMD).

Faithful Python port of Hemakom, Ahrabian, Looney, Rehman and Mandic's
MATLAB ``nstemd.m``.  The sifting loop is MEMD; the difference is the
projection set.  For trivariate data a Hammersley net is mapped onto an
ellipsoid whose axes are the cube roots of the PCA eigenvalues, rotated
by the eigenvectors, then **unioned** with the original uniform Hammersley
set (so the envelope mean uses ``2 * n_dir`` directions).

Hemakom, A., Ahrabian, A., Looney, D., Rehman, N. and Mandic, D. P.
"Nonuniformly sampled trivariate empirical mode decomposition."
IEEE ICASSP, 2015.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import warnings

import numpy as np
from scipy.interpolate import CubicSpline

from pysdkit._emd.memd import MEMD, unit_direction, local_peaks
from pysdkit._emd.memd import zero_crossings, boundary_conditions
from pysdkit._emd.memd import _check_multivariate_signal

_MIN_N_DIR = 6


class NSTEMD(MEMD):
    """
    Nonuniformly Sampled Trivariate Empirical Mode Decomposition (NS-TEMD).

    Standard MEMD / TEMD place Hammersley directions **uniformly** on the
    2-sphere.  When the three channels are power-imbalanced or correlated,
    that net undersamples the direction of highest curvature unless
    ``n_dir`` is large.  NS-TEMD:

    1. estimates the covariance PCA of the **input** (MATLAB ``princomp``,
       once, not every sifting step);
    2. maps the Hammersley spherical angles onto an ellipsoid with
       semi-axes ``λ_i^{1/3}``, re-normalises to the sphere and rotates
       by the eigenvector matrix ``V``;
    3. estimates the local mean from **both** that nonuniform set **and**
       the original uniform Hammersley set (paper Algorithm 1, step 4).

    The MATLAB header therefore requires the NS-TEMD ``n_dir`` to be
    **half** the MEMD count used in a comparison (total projections
    ``2 * n_dir``).  Default ``n_dir`` is still 64, matching ``nstemd.m``.

    For ``n_channels != 3`` the MATLAB file falls back to two copies of
    the uniform n-sphere Hammersley net (no ellipsoid).  That path is
    preserved.

    Input layout is ``(n_channels, n_samples)`` with
    ``3 <= n_channels <= 16``.  Output IMFs have shape
    ``(n_imfs, n_samples, n_channels)``; the last slice is the residue.

    :param stop_crit: str,
        ``"stop"`` (Rilling et al., 2003) or ``"fix_h"`` (Huang et al., 2003).
    :param max_iter: int,
        Maximum sifting iterations per IMF (MATLAB default 1000).
    :param n_dir: int,
        Number of Hammersley samples.  Envelope averaging uses
        ``2 * n_dir`` projections.  Must be ``>= 6``.
    :param stop_vec: sequence of float or None,
        ``[sd, sd2, tol]`` used when ``stop_crit="stop"``.
    :param stop_cnt: int,
        Number of consecutive siftings required when ``stop_crit="fix_h"``.
    """

    def __init__(
        self,
        stop_crit: str = "stop",
        max_iter: int = 1000,
        n_dir: int = 64,
        stop_vec: Optional[Sequence[float]] = None,
        stop_cnt: int = 2,
    ) -> None:
        """
        Store NS-TEMD hyperparameters used by later sifting calls.

        :param stop_crit: str,
            Stopping rule, either ``"stop"`` or ``"fix_h"``.
        :param max_iter: int,
            Maximum inner sifting iterations for each IMF (must be ``>= 1``).
        :param n_dir: int,
            Hammersley sample count (must be ``>= 6``). Total projections
            are ``2 * n_dir``.
        :param stop_vec: sequence of float or None,
            Three thresholds ``[sd, sd2, tol]``. ``None`` selects
            ``[0.075, 0.75, 0.075]``.
        :param stop_cnt: int,
            Consecutive successful siftings for ``"fix_h"`` (must be ``>= 0``).
        :return: None
        """
        super().__init__(
            stop_crit=stop_crit,
            max_iter=max_iter,
            n_dir=n_dir,
            stop_vec=stop_vec,
            stop_cnt=stop_cnt,
        )
        self.eig_vec_: Optional[np.ndarray] = None
        self.eig_val_: Optional[np.ndarray] = None

    def __str__(self) -> str:
        """
        Return the full algorithm name and abbreviation.

        :return: str,
            ``"Nonuniformly Sampled Trivariate Empirical Mode Decomposition (NS-TEMD)"``.
        """
        return (
            "Nonuniformly Sampled Trivariate Empirical Mode Decomposition " "(NS-TEMD)"
        )

    def principal_components(
        self, residue: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        PCA of a time-by-channel record (MATLAB ``princomp``).

        :param residue: ndarray,
            Array of shape ``(n_samples, n_channels)``.
        :return eig_vec: ndarray,
            Columns are principal axes, shape ``(n_channels, n_channels)``,
            ordered by decreasing variance.
        :return eig_val: ndarray,
            Corresponding eigenvalues (``latent``), shape ``(n_channels,)``.
        """
        return princomp(residue)

    def ellipsoid_directions(
        self,
        seq: np.ndarray,
        eig_vec: np.ndarray,
        eig_val: np.ndarray,
    ) -> np.ndarray:
        """
        Map Hammersley spherical angles onto the PCA ellipsoid, then
        re-normalise (MATLAB trivariate branch of ``stop_emd``).

        :param seq: ndarray,
            Hammersley samples of shape ``(n_dir, 2)``.
        :param eig_vec: ndarray,
            PCA loadings, shape ``(3, 3)``.
        :param eig_val: ndarray,
            PCA eigenvalues of length at least 3.
        :return dirs: ndarray,
            Unit vectors of shape ``(n_dir, 3)``.
        """
        return ellipsoid_directions(seq, eig_vec, eig_val)

    def projection_directions(
        self,
        seq: np.ndarray,
        n_dim: int,
        eig_vec: np.ndarray,
        eig_val: np.ndarray,
    ) -> np.ndarray:
        """
        Concatenate nonuniform (ellipsoid) and uniform Hammersley directions.

        :param seq: ndarray,
            Hammersley samples from :meth:`init_hammersley`.
        :param n_dim: int,
            Number of channels.
        :param eig_vec: ndarray,
            PCA loadings.
        :param eig_val: ndarray,
            PCA eigenvalues.
        :return dirs: ndarray,
            Unit vectors of shape ``(2 * n_dir, n_dim)``.
        """
        return projection_directions(seq, n_dim, eig_vec, eig_val, self.n_dir)

    def _pca(self, signal: np.ndarray, n_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return stored PCA if it matches ``n_dim``, otherwise compute it."""
        if (
            self.eig_vec_ is not None
            and self.eig_val_ is not None
            and self.eig_vec_.shape[0] == n_dim
        ):
            return self.eig_vec_, self.eig_val_
        return princomp(signal)

    def stop_emd(self, signal: np.ndarray, seq: np.ndarray, n_dim: int) -> bool:
        """
        Decide whether IMF extraction should stop for the current residue.

        Projects onto **both** nonuniform and uniform directions (MATLAB
        ``stop_emd``, ``ner`` of length ``2 * n_dir``).  Extraction stops
        when all projections have fewer than three extrema.

        :param signal: ndarray,
            Current residue of shape ``(n_samples, n_channels)``.
        :param seq: ndarray,
            Hammersley samples from :meth:`init_hammersley`.
        :param n_dim: int,
            Number of channels.
        :return stop_flag: bool,
            ``True`` if no further oscillatory IMF should be extracted.
        """
        eig_vec, eig_val = self._pca(signal, n_dim)
        dirs = self.projection_directions(seq, n_dim, eig_vec, eig_val)
        ner = np.zeros(dirs.shape[0], dtype=float)
        for it in range(dirs.shape[0]):
            y = np.dot(signal, dirs[it])
            indmin, indmax = local_peaks(y)
            ner[it] = len(indmin) + len(indmax)
        return bool(np.all(ner < 3))

    def stop(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        seq: np.ndarray,
        seq_len: int,
        n_dim: int,
    ) -> Tuple[bool, np.ndarray]:
        """
        Rilling sifting stop test (MATLAB ``stop_sifting``).

        Uses NS-TEMD envelopes (uniform + ellipsoid projections).

        :param signal: ndarray,
            Current proto-IMF of shape ``(n_samples, n_channels)``.
        :param time: ndarray,
            Sample indices of length ``n_samples``.
        :param seq: ndarray,
            Hammersley samples from :meth:`init_hammersley`.
        :param seq_len: int,
            Number of time samples ``N``.
        :param n_dim: int,
            Number of channels.
        :return stop_flag: bool,
            ``True`` if sifting of this IMF should stop.
        :return env_mean: ndarray,
            Local-mean estimate of shape ``(n_samples, n_channels)``.
        """
        try:
            eig_vec, eig_val = self._pca(signal, n_dim)
            env_mean, nem, _nzm, amp = envelope_mean(
                signal,
                time,
                seq,
                self.n_dir,
                seq_len,
                n_dim,
                eig_vec,
                eig_val,
            )
            sx = np.sqrt(np.sum(env_mean**2, axis=1))
            if np.all(amp):
                sx = sx / amp
            continue_sift = (
                np.mean(sx > self.sd) > self.tol or np.any(sx > self.sd2)
            ) and np.any(nem > 2)
            stop_flag = not bool(continue_sift)
        except Exception:
            env_mean = np.zeros((seq_len, n_dim), dtype=float)
            stop_flag = True
        return stop_flag, env_mean

    def fix(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        seq: np.ndarray,
        seq_len: int,
        n_dim: int,
        counter: int,
    ) -> Tuple[bool, np.ndarray, int]:
        """
        Huang ``fix_h`` sifting stop test (MATLAB ``stop_sifting_fix``).

        :param signal: ndarray,
            Current proto-IMF of shape ``(n_samples, n_channels)``.
        :param time: ndarray,
            Sample indices of length ``n_samples``.
        :param seq: ndarray,
            Hammersley samples from :meth:`init_hammersley`.
        :param seq_len: int,
            Number of time samples ``N``.
        :param n_dim: int,
            Number of channels.
        :param counter: int,
            Number of consecutive successful ``fix_h`` iterations so far.
        :return stop_flag: bool,
            ``True`` if sifting of this IMF should stop.
        :return env_mean: ndarray,
            Local-mean estimate of shape ``(n_samples, n_channels)``.
        :return counter: int,
            Updated consecutive-success counter.
        """
        try:
            eig_vec, eig_val = self._pca(signal, n_dim)
            env_mean, nem, nzm, _amp = envelope_mean(
                signal,
                time,
                seq,
                self.n_dir,
                seq_len,
                n_dim,
                eig_vec,
                eig_val,
            )
            if np.all(np.abs(nzm - nem) > 1):
                stop_flag = False
                counter = 0
            else:
                counter += 1
                stop_flag = counter >= self.stop_cnt
        except Exception:
            env_mean = np.zeros((seq_len, n_dim), dtype=float)
            stop_flag = True
        return stop_flag, env_mean, counter

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Decompose a multivariate signal into aligned IMFs plus a residue.

        PCA of the **original** record is computed once (MATLAB
        ``princomp(q)`` in ``set_value``) and reused for every sifting.

        :param signal: ndarray,
            Multivariate record of shape ``(n_channels, n_samples)`` with
            ``3 <= n_channels <= 16``.
        :return imfs: ndarray,
            Stack of shape ``(n_imfs, n_samples, n_channels)``.
        """
        x = _check_multivariate_signal(signal)
        n_dim, seq_len = x.shape

        current = x.T.copy()
        self.eig_vec_, self.eig_val_ = princomp(current)

        seq = self.init_hammersley(n_dim)
        time = np.arange(1, seq_len + 1, dtype=float)

        imfs: List[np.ndarray] = []
        nbit = 0
        orig_amp = np.max(np.abs(current))

        while not self.stop_emd(signal=current, seq=seq, n_dim=n_dim):
            m = current.copy()

            if self.stop_crit == "stop":
                stop_flag, env_mean = self.stop(
                    signal=m, time=time, seq=seq, n_dim=n_dim, seq_len=seq_len
                )
            else:
                counter = 0
                stop_flag, env_mean, counter = self.fix(
                    signal=m,
                    time=time,
                    seq=seq,
                    seq_len=seq_len,
                    n_dim=n_dim,
                    counter=counter,
                )

            if np.max(np.abs(m)) < 1e-10 * orig_amp:
                if not stop_flag:
                    warnings.warn(
                        "emd:warning, forced stop of EMD : too small amplitude"
                    )
                break

            while (not stop_flag) and nbit < self.max_iter:
                m = m - env_mean
                if self.stop_crit == "stop":
                    stop_flag, env_mean = self.stop(
                        signal=m, time=time, seq=seq, n_dim=n_dim, seq_len=seq_len
                    )
                else:
                    stop_flag, env_mean, counter = self.fix(
                        signal=m,
                        time=time,
                        seq=seq,
                        seq_len=seq_len,
                        n_dim=n_dim,
                        counter=counter,
                    )
                nbit += 1
                if nbit == (self.max_iter - 1) and nbit > 100:
                    warnings.warn(
                        "emd:warning, forced stop of sifting : too many iterations"
                    )

            imfs.append(m)
            current = current - m
            nbit = 0

        imfs.append(current)
        return np.asarray(imfs, dtype=float)


def princomp(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB ``princomp``: SVD PCA of a time-by-channel matrix.

    Columns of ``x`` are variables.  The data are centred, then
    ``Xc = U S V^T`` with ``coeff = V`` and
    ``latent = S^2 / (n - 1)`` in decreasing order.

    :param x: ndarray,
        Array of shape ``(n_samples, n_channels)``.
    :return coeff: ndarray,
        Principal axes as columns, shape ``(n_channels, n_channels)``.
    :return latent: ndarray,
        Eigenvalues of the sample covariance, descending.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("princomp expects shape (n_samples, n_channels)")
    n_samples, n_channels = x.shape
    xc = x - x.mean(axis=0, keepdims=True)
    if n_samples < 2:
        return np.eye(n_channels, dtype=float), np.zeros(n_channels, dtype=float)

    _u, singular, vt = np.linalg.svd(xc, full_matrices=True)
    # ``vt`` may have extra rows when n_samples < n_channels; take n_channels.
    coeff = np.asarray(vt[:n_channels, :].T, dtype=float)
    latent = np.zeros(n_channels, dtype=float)
    n_comp = min(singular.size, n_channels)
    latent[:n_comp] = (singular[:n_comp] ** 2) / float(n_samples - 1)
    return coeff, latent


def ellipsoid_directions(
    seq: np.ndarray,
    eig_vec: np.ndarray,
    eig_val: np.ndarray,
) -> np.ndarray:
    """
    Hammersley angles → PCA ellipsoid → unit sphere (MATLAB trivariate map).

    Semi-axes are ``nthroot(λ_i, 3)``.  Spherical angles come from the
    same 2-sphere Hammersley map as MEMD (``tt = 2u-1``, ``φ = 2π v``),
    then

    ``(x, y, z) = (a sinθ cosφ, b sinθ sinφ, c cosθ)``

    with ``θ = arccos(tt)``.  The point is re-normalised and rotated by
    ``eig_vec``.

    :param seq: ndarray,
        Hammersley samples of shape ``(n_dir, 2)``.
    :param eig_vec: ndarray,
        Rotation ``V``, shape ``(3, 3)``.
    :param eig_val: ndarray,
        Eigenvalues of length ``>= 3``.
    :return dirs: ndarray,
        Unit vectors of shape ``(n_dir, 3)``.
    """
    seq = np.asarray(seq, dtype=float)
    eig_vec = np.asarray(eig_vec, dtype=float)
    eig_val = np.asarray(eig_val, dtype=float).reshape(-1)
    if seq.ndim != 2 or seq.shape[1] < 2:
        raise ValueError("seq must have shape (n_dir, 2) for trivariate data")
    if eig_vec.shape != (3, 3):
        raise ValueError("eig_vec must have shape (3, 3)")
    if eig_val.size < 3:
        raise ValueError("eig_val must contain at least 3 eigenvalues")

    a, b, c = np.cbrt(eig_val[0]), np.cbrt(eig_val[1]), np.cbrt(eig_val[2])
    n_dir = seq.shape[0]
    dirs = np.zeros((n_dir, 3), dtype=float)
    for it in range(n_dir):
        tt = float(np.clip(2.0 * seq[it, 0] - 1.0, -1.0, 1.0))
        phirad = float(seq[it, 1] * 2.0 * np.pi)
        phi = float(np.arccos(tt))
        x = a * np.sin(phi) * np.cos(phirad)
        y = b * np.sin(phi) * np.sin(phirad)
        z = c * np.cos(phi)
        nrm = float(np.sqrt(x * x + y * y + z * z))
        if nrm < 1e-15:
            st = np.sqrt(max(1.0 - tt * tt, 0.0))
            vec = np.array([st * np.cos(phirad), st * np.sin(phirad), tt], dtype=float)
        else:
            vec = np.array([x / nrm, y / nrm, z / nrm], dtype=float)
        rotated = eig_vec @ vec
        n2 = float(np.linalg.norm(rotated))
        dirs[it] = rotated / n2 if n2 >= 1e-15 else vec
    return dirs


def projection_directions(
    seq: np.ndarray,
    n_dim: int,
    eig_vec: np.ndarray,
    eig_val: np.ndarray,
    n_dir: int,
) -> np.ndarray:
    """
    Build the ``2 * n_dir`` projection set used by MATLAB ``nstemd.m``.

    For ``n_dim == 3`` this is ``[ellipsoid; uniform]``.  Otherwise both
    blocks are the uniform n-sphere Hammersley set (MATLAB ``N_dim ~= 3``
    branch).

    :param seq: ndarray,
        Hammersley samples from :meth:`NSTEMD.init_hammersley`.
    :param n_dim: int,
        Number of channels.
    :param eig_vec: ndarray,
        PCA loadings.
    :param eig_val: ndarray,
        PCA eigenvalues.
    :param n_dir: int,
        Number of Hammersley samples.
    :return dirs: ndarray,
        Unit vectors of shape ``(2 * n_dir, n_dim)``.
    """
    if n_dir < _MIN_N_DIR:
        raise ValueError("n_dir should be an integer greater than or equal to 6.")
    uniform = np.zeros((n_dir, n_dim), dtype=float)
    for it in range(n_dir):
        uniform[it] = unit_direction(seq, it, n_dim)
    if n_dim == 3:
        nonuniform = ellipsoid_directions(seq, eig_vec, eig_val)
        return np.vstack((nonuniform, uniform))
    return np.vstack((uniform, uniform))


def envelope_mean(
    m: np.ndarray,
    t: np.ndarray,
    seq: np.ndarray,
    n_dir: int,
    seq_len: int,
    n_dim: int,
    eig_vec: np.ndarray,
    eig_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Average multivariate envelopes over NS-TEMD projections.

    MATLAB averages ``2 * ndir`` envelopes (nonuniform + uniform) and
    divides by ``2*ndir - count``.  Extrema / zero-crossing counts are
    stored for **both** blocks (length ``2 * n_dir``).  The MATLAB file
    overwrites ``nem(1:ndir)`` in the second loop; that is treated as a
    copy-paste error — ``stop_emd`` already expands ``ner`` to
    ``2 * ndir``.

    :param m: ndarray,
        Current proto-IMF of shape ``(n_samples, n_channels)``.
    :param t: ndarray,
        Sample indices of length ``n_samples``.
    :param seq: ndarray,
        Hammersley samples.
    :param n_dir: int,
        Hammersley sample count.
    :param seq_len: int,
        Number of time samples.
    :param n_dim: int,
        Number of channels.
    :param eig_vec: ndarray,
        PCA loadings.
    :param eig_val: ndarray,
        PCA eigenvalues.
    :return env_mean: ndarray,
        Mean envelope of shape ``(n_samples, n_channels)``.
    :return nem: ndarray,
        Extrema counts, shape ``(2 * n_dir,)``.
    :return nzm: ndarray,
        Zero-crossing counts, shape ``(2 * n_dir,)``.
    :return amp: ndarray,
        Mode-amplitude estimate of shape ``(n_samples,)``.
    """
    nbsym = 2
    count = 0
    n_proj = 2 * n_dir
    env_mean = np.zeros((len(t), n_dim), dtype=float)
    amp = np.zeros(len(t), dtype=float)
    nem = np.zeros(n_proj, dtype=float)
    nzm = np.zeros(n_proj, dtype=float)

    dirs = projection_directions(seq, n_dim, eig_vec, eig_val, n_dir)
    for it in range(dirs.shape[0]):
        y = np.dot(dirs[it], m.T)
        indmin, indmax = local_peaks(y)
        nem[it] = len(indmin) + len(indmax)
        nzm[it] = len(zero_crossings(y))

        tmin, tmax, zmin, zmax, mode = boundary_conditions(
            indmin, indmax, t, y, m, nbsym
        )
        if mode:
            env_min = CubicSpline(tmin, zmin, bc_type="not-a-knot")(t)
            env_max = CubicSpline(tmax, zmax, bc_type="not-a-knot")(t)
            amp = amp + np.sqrt(np.sum((env_max - env_min) ** 2, axis=1)) / 2.0
            env_mean = env_mean + (env_max + env_min) / 2.0
        else:
            count += 1

    if n_proj > count:
        env_mean = env_mean / float(n_proj - count)
        amp = amp / float(n_proj - count)
    else:
        env_mean = np.zeros((seq_len, n_dim), dtype=float)
        amp = np.zeros(seq_len, dtype=float)
        nem = np.zeros(n_proj, dtype=float)
    return env_mean, nem, nzm, amp
