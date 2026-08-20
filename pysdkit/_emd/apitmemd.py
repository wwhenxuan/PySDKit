# -*- coding: utf-8 -*-
"""
Adaptive-Projection Intrinsically Transformed MEMD (APIT-MEMD).

Faithful Python port of Hemakom, Goverdovsky, Looney and Mandic's MATLAB
``apitmemd.m`` / ``nonuniform_nD_2.m``.  The sifting loop is the same as
MEMD; the difference is that Hammersley directions are **relocated** toward
the first principal component (and its opposite) of the current residue,
controlled by ``alpha``.

Hemakom, A., Goverdovsky, V., Looney, D. and Mandic, D. P.
"Adaptive-projection intrinsically transformed multivariate empirical mode
decomposition in cooperative brain–computer interface applications."
Phil. Trans. R. Soc. A 374:20150199 (2016).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import warnings

import numpy as np
from scipy.interpolate import CubicSpline

from pysdkit._emd.memd import MEMD, unit_direction
from pysdkit._emd.memd import peaks as _memd_peaks
from pysdkit._emd.memd import zero_crossings, boundary_conditions
from pysdkit._emd.memd import _as_1d

# MATLAB ``apitmemd.m`` ``Max_channels`` (comment still says 3–16).
_MAX_CHANNELS = 32
_MIN_CHANNELS = 3
_MIN_N_DIR = 6


class APITMEMD(MEMD):
    """
    Adaptive-Projection Intrinsically Transformed MEMD (APIT-MEMD).

    Standard MEMD places Hammersley directions uniformly on the
    ``(n-1)``-sphere.  That sampling is suboptimal when channels are
    **power-imbalanced or correlated**, unless ``n_dir`` is very large.
    APIT-MEMD keeps the same Hammersley set as a starting point, estimates
    the first principal component of the current multivariate residue, and
    relocates the ``n_dir / 2`` directions closest to that axis (and the
    ``n_dir / 2`` closest to its opposite) by a step of size ``alpha``.

    * ``alpha = 0`` — no relocation (MEMD-like; the used set is still the
      subset nearest PC1 / −PC1, not the original Hammersley order).
    * ``alpha ∈ (0, 1]`` — directions are pulled toward the power-imbalance
      axis (paper Figure 2e–g).  MATLAB default is ``0.3``.

    Input layout is ``(n_channels, n_samples)`` with
    ``3 <= n_channels <= 32``.  A MATLAB-style ``(n_samples, n_channels)``
    array is transposed automatically.  Output IMFs have shape
    ``(n_imfs, n_samples, n_channels)``; the last slice is the residue.

    :param stop_crit: str,
        ``"stop"`` (Rilling et al., 2003) or ``"fix_h"`` (Huang et al., 2003).
    :param max_iter: int,
        Maximum sifting iterations per IMF (MATLAB default 1000).
    :param n_dir: int,
        Number of projection directions (MATLAB default 64). Must be ``>= 6``.
    :param stop_vec: sequence of float or None,
        ``[sd, sd2, tol]`` used when ``stop_crit="stop"``.
    :param stop_cnt: int,
        Number of consecutive siftings required when ``stop_crit="fix_h"``.
    :param alpha: float,
        Relocation strength toward the first principal component
        (MATLAB ``alpha_var``, default ``0.3``). Must be finite and ``>= 0``.
    """

    def __init__(
        self,
        stop_crit: str = "stop",
        max_iter: int = 1000,
        n_dir: int = 64,
        stop_vec: Optional[Sequence[float]] = None,
        stop_cnt: int = 2,
        alpha: float = 0.3,
    ) -> None:
        """
        Store APIT-MEMD hyperparameters used by later sifting calls.

        :param stop_crit: str,
            Stopping rule, either ``"stop"`` or ``"fix_h"``.
        :param max_iter: int,
            Maximum inner sifting iterations for each IMF (must be ``>= 1``).
        :param n_dir: int,
            Number of Hammersley projection directions (must be ``>= 6``).
        :param stop_vec: sequence of float or None,
            Three thresholds ``[sd, sd2, tol]``. ``None`` selects
            ``[0.075, 0.75, 0.075]``.
        :param stop_cnt: int,
            Consecutive successful siftings for ``"fix_h"`` (must be ``>= 0``).
        :param alpha: float,
            Non-negative relocation strength (MATLAB default ``0.3``).
        :return: None
        """
        super().__init__(
            stop_crit=stop_crit,
            max_iter=max_iter,
            n_dir=n_dir,
            stop_vec=stop_vec,
            stop_cnt=stop_cnt,
        )
        if not np.isscalar(alpha) or not np.isfinite(alpha) or float(alpha) < 0.0:
            raise ValueError(
                "invalid alpha. alpha should be a non-negative finite number "
                "(MATLAB default is 0.3; alpha=0 is MEMD-like)."
            )
        self.alpha = float(alpha)

    def __str__(self) -> str:
        """
        Return the full algorithm name and abbreviation.

        :return: str,
            ``"Adaptive-Projection Intrinsically Transformed MEMD (APIT-MEMD)"``.
        """
        return "Adaptive-Projection Intrinsically Transformed MEMD (APIT-MEMD)"

    def first_principal_component(self, residue: np.ndarray) -> np.ndarray:
        """
        Leading eigenvector of the covariance of a multivariate residue.

        Matches MATLAB ``[V, E] = eig(cov(m))`` followed by sorting
        eigenvalues in descending order and taking ``V(:, 1)``.

        :param residue: ndarray,
            Current sifting input of shape ``(n_samples, n_channels)``.
        :return pc1: ndarray,
            Unit vector of shape ``(n_channels,)``.
        """
        return first_principal_component(residue)

    def adaptive_directions(self, residue: np.ndarray, n_dim: int) -> np.ndarray:
        """
        Relocate Hammersley directions toward PC1 of ``residue``.

        Builds the uniform Hammersley set of length ``n_dir`` and applies
        MATLAB ``nonuniform_nD_2`` with this instance's ``alpha``.

        :param residue: ndarray,
            Current sifting input of shape ``(n_samples, n_channels)``.
        :param n_dim: int,
            Number of channels (must match ``residue.shape[1]``).
        :return dirs: ndarray,
            Unit vectors of shape ``(n_adaptive, n_dim)`` where
            ``n_adaptive = 2 * floor(n_dir / 2)``.
        """
        uniform = self.direction_vectors(n_dim)
        return nonuniform_directions(uniform, residue, self.alpha)

    def stop_emd(self, signal: np.ndarray, seq: np.ndarray, n_dim: int) -> bool:
        """
        Decide whether IMF extraction should stop for the current residue.

        Projects the residue onto **adaptive** directions (MATLAB
        ``stop_emd``).  If relocation fails, falls back to the uniform
        Hammersley set.  Extraction stops when **all** projections have
        fewer than three extrema.

        :param signal: ndarray,
            Current residue of shape ``(n_samples, n_channels)``.
        :param seq: ndarray,
            Hammersley samples from :meth:`init_hammersley`.
        :param n_dim: int,
            Number of channels.
        :return stop_flag: bool,
            ``True`` if no further oscillatory IMF should be extracted.
        """
        uniform = hammersley_unit_directions(seq, self.n_dir, n_dim)
        try:
            dirs = nonuniform_directions(uniform, signal, self.alpha)
        except Exception:
            dirs = uniform

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

        Uses APIT envelopes (adaptive projections) rather than uniform
        Hammersley projections.

        :param signal: ndarray,
            Current proto-IMF of shape ``(n_samples, n_channels)``.
        :param time: ndarray,
            Sample indices of length ``n_samples`` (MATLAB ``t = 1:N``).
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
            env_mean, nem, _nzm, amp = envelope_mean(
                signal, time, seq, self.n_dir, seq_len, n_dim, self.alpha
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

        Same consecutive extrema / zero-crossing rule as MEMD, but the
        envelope mean is estimated from adaptive projections.

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
            env_mean, nem, nzm, _amp = envelope_mean(
                signal, time, seq, self.n_dir, seq_len, n_dim, self.alpha
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

        :param signal: ndarray,
            Multivariate record of shape ``(n_channels, n_samples)`` with
            ``3 <= n_channels <= 32``. A ``(n_samples, n_channels)`` array is
            accepted and transposed when the channel axis is unambiguous.
        :return imfs: ndarray,
            Stack of shape ``(n_imfs, n_samples, n_channels)``. Oscillatory
            IMFs occupy slices ``0 .. n_imfs-2``; slice ``-1`` is the residue.
            Summing along axis 0 reconstructs the (possibly transposed) input.
        """
        x = check_multivariate_signal(signal)
        n_dim, seq_len = x.shape

        seq = self.init_hammersley(n_dim)
        time = np.arange(1, seq_len + 1, dtype=float)
        current = x.T.copy()  # (n_samples, n_channels)

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


def check_multivariate_signal(signal: np.ndarray) -> np.ndarray:
    """
    Validate and orient a multivariate APIT-MEMD input.

    Accepts ``(n_channels, n_samples)`` or, when the other layout is the only
    one whose first dimension lies in ``[3, 32]``, MATLAB
    ``(n_samples, n_channels)``.

    :param signal: array-like,
        Candidate 2-D multivariate record.
    :return x: ndarray,
        Float array of shape ``(n_channels, n_samples)`` with
        ``3 <= n_channels <= 32`` and at least 3 samples.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 2:
        raise ValueError(
            "APIT-MEMD expects a 2-D array of shape (n_channels, n_samples)"
        )
    n0, n1 = x.shape
    if not (_MIN_CHANNELS <= n0 <= _MAX_CHANNELS) and (
        _MIN_CHANNELS <= n1 <= _MAX_CHANNELS
    ):
        x = x.T
        n0, n1 = x.shape
    if not (_MIN_CHANNELS <= n0 <= _MAX_CHANNELS):
        raise ValueError(
            "APIT-MEMD processes signals with 3 to 32 channels. "
            "Got shape {}; try EMD / BEMD for fewer channels.".format(signal.shape)
        )
    if n1 < 3:
        raise ValueError("APIT-MEMD requires at least 3 samples")
    return x


def hammersley_unit_directions(seq: np.ndarray, n_dir: int, n_dim: int) -> np.ndarray:
    """
    Convert a Hammersley point set into unit projection directions.

    :param seq: ndarray,
        Hammersley samples from :meth:`APITMEMD.init_hammersley`
        (rows = directions).
    :param n_dir: int,
        Number of directions (rows to convert).
    :param n_dim: int,
        Ambient dimension of each direction vector.
    :return dirs: ndarray,
        Unit vectors of shape ``(n_dir, n_dim)``.
    """
    dirs = np.zeros((n_dir, n_dim), dtype=float)
    for it in range(n_dir):
        dirs[it] = unit_direction(seq, it, n_dim)
    return dirs


def first_principal_component(residue: np.ndarray) -> np.ndarray:
    """
    Unit first principal component of a time-by-channel residue.

    MATLAB ``cov(m)`` treats columns as variables.  Eigenvectors are sorted
    by descending eigenvalue (MATLAB ``sort(diag(E), 'descend')``).

    :param residue: ndarray,
        Array of shape ``(n_samples, n_channels)``.
    :return pc1: ndarray,
        Unit vector of shape ``(n_channels,)``.  If the covariance is
        degenerate, the first canonical axis is returned.
    """
    m = np.asarray(residue, dtype=float)
    if m.ndim != 2 or m.shape[1] < 1:
        raise ValueError("residue must have shape (n_samples, n_channels)")
    n_samples, n_channels = m.shape
    if n_samples < 2:
        axis = np.zeros(n_channels, dtype=float)
        axis[0] = 1.0
        return axis

    cov = np.cov(m, rowvar=False, ddof=1)
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if cov.shape != (n_channels, n_channels) or not np.all(np.isfinite(cov)):
        raise ValueError("covariance of residue is not a finite square matrix")

    evals, evecs = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(evals)[::-1]
    pc1 = np.asarray(evecs[:, order[0]], dtype=float).reshape(-1)
    nrm = float(np.linalg.norm(pc1))
    if nrm < 1e-15:
        axis = np.zeros(n_channels, dtype=float)
        axis[0] = 1.0
        return axis
    return pc1 / nrm


def principal_components(residue: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full eigendecomposition of ``cov(residue)`` sorted descending.

    :param residue: ndarray,
        Array of shape ``(n_samples, n_channels)``.
    :return eigenvectors: ndarray,
        Columns are principal axes, shape ``(n_channels, n_channels)``.
    :return eigenvalues: ndarray,
        Corresponding eigenvalues, shape ``(n_channels,)``, descending.
    """
    m = np.asarray(residue, dtype=float)
    if m.ndim != 2 or m.shape[0] < 2:
        n_channels = int(m.shape[1]) if m.ndim == 2 else 1
        return np.eye(n_channels, dtype=float), np.zeros(n_channels, dtype=float)
    cov = np.atleast_2d(np.cov(m, rowvar=False, ddof=1))
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    return np.asarray(evecs[:, order], dtype=float), np.asarray(
        evals[order], dtype=float
    )


def nonuniform_directions(
    uniform_dirs: np.ndarray,
    residue: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Relocate Hammersley directions toward PC1 (MATLAB ``nonuniform_nD_2``).

    This MATLAB version assigns **all** ``floor(n_dir / 2)`` slots to the
    first principal component and the same number to its opposite
    (``N(1) = ndir/2``, ``N(2:end) = 0``).  Selected vectors are then
    shifted by ``± alpha * PC1`` and re-normalised.

    :param uniform_dirs: ndarray,
        Uniform unit directions of shape ``(n_dir, n_dim)``.
    :param residue: ndarray,
        Current sifting input of shape ``(n_samples, n_dim)``.
    :param alpha: float,
        Relocation strength (``0`` leaves the selected subset unshifted
        except for re-normalisation).
    :return adapted: ndarray,
        Unit vectors of shape ``(2 * floor(n_dir / 2), n_dim)``.
        Rows ``0 : n_half`` lie toward ``+PC1``; the remainder toward ``-PC1``.
    """
    uniform = np.asarray(uniform_dirs, dtype=float)
    if uniform.ndim != 2:
        raise ValueError("uniform_dirs must have shape (n_dir, n_dim)")
    n_dir, n_dim = uniform.shape
    residue = np.asarray(residue, dtype=float)
    if residue.ndim != 2 or residue.shape[1] != n_dim:
        raise ValueError(
            "residue must have shape (n_samples, n_dim) matching uniform_dirs"
        )
    if n_dir < 2:
        raise ValueError("need at least 2 uniform directions")

    pc1 = first_principal_component(residue)
    n_half = int(np.floor(float(n_dir) / 2.0))
    if n_half < 1:
        raise ValueError("floor(n_dir / 2) is zero; increase n_dir")

    # MATLAB: uniform_dir_vec is transposed to (n_dim, n_dir) before distances.
    dist_pos = np.linalg.norm(uniform - pc1.reshape(1, -1), axis=1)
    dist_neg = np.linalg.norm(uniform - (-pc1).reshape(1, -1), axis=1)
    idx_pos = np.argsort(dist_pos, kind="mergesort")[:n_half]
    idx_neg = np.argsort(dist_neg, kind="mergesort")[:n_half]

    dir_pc = uniform[idx_pos].copy()
    dir_opp = uniform[idx_neg].copy()
    dir_pc = dir_pc + float(alpha) * pc1.reshape(1, -1)
    dir_opp = dir_opp - float(alpha) * pc1.reshape(1, -1)

    nrm_pc = np.linalg.norm(dir_pc, axis=1, keepdims=True)
    nrm_opp = np.linalg.norm(dir_opp, axis=1, keepdims=True)
    if np.any(nrm_pc < 1e-15) or np.any(nrm_opp < 1e-15):
        raise ValueError("relocated direction has vanishing norm")
    dir_pc = dir_pc / nrm_pc
    dir_opp = dir_opp / nrm_opp
    return np.vstack((dir_pc, dir_opp))


def local_peaks(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect local minima and maxima of a projected 1-D signal.

    Matches MATLAB ``local_peaks`` in ``apitmemd.m``: plateaus are collapsed
    to their midpoints before the first-difference peak test.  A near-zero
    projection (all samples ``< 1e-6``) is treated as identically zero.
    (MEMD uses ``1e-5``; APIT-MEMD uses ``1e-6``.)

    :param x: ndarray,
        Scalar projection, any shape that flattens to length ``N``.
    :return indmin: ndarray of int,
        0-based indices of local minima (empty if none).
    :return indmax: ndarray of int,
        0-based indices of local maxima (empty if none).
    """
    x = _as_1d(x)
    if x.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if np.all(x < 1e-6):
        x = np.zeros(x.size, dtype=float)

    m = x.size - 1
    dy = np.diff(x)
    a = np.where(dy != 0)[0]
    if a.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    lm = np.where(np.diff(a) != 1)[0] + 1
    if lm.size:
        d = a[lm] - a[lm - 1]
        a = a.astype(float)
        a[lm] = a[lm] - np.floor(d / 2.0)
    a = np.concatenate((a, [m])).astype(int)
    ya = x[a]

    if ya.size <= 1:
        return np.array([], dtype=int), np.array([], dtype=int)

    pks_max, loc_max = _memd_peaks(ya)
    pks_min, loc_min = _memd_peaks(-ya)

    indmin = a[loc_min] if pks_min.size else np.array([], dtype=int)
    indmax = a[loc_max] if pks_max.size else np.array([], dtype=int)
    return np.asarray(indmin, dtype=int), np.asarray(indmax, dtype=int)


def envelope_mean(
    m: np.ndarray,
    t: np.ndarray,
    seq: np.ndarray,
    n_dir: int,
    seq_len: int,
    n_dim: int,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Average multivariate envelopes over adaptive (APIT) projections.

    Uniform Hammersley directions are built from ``seq``, relocated with
    :func:`nonuniform_directions`, then each projection is envelope-interpolated
    exactly as in MEMD.  The divisor is ``n_dir - count`` (MATLAB uses the
    original ``ndir``, not ``size(nonuniform_dir_vec, 1)``).

    :param m: ndarray,
        Current proto-IMF of shape ``(n_samples, n_channels)``.
    :param t: ndarray,
        Sample indices of length ``n_samples``.
    :param seq: ndarray,
        Hammersley samples from :meth:`APITMEMD.init_hammersley`.
    :param n_dir: int,
        Number of (uniform) projection directions.
    :param seq_len: int,
        Number of time samples (used if every projection is skipped).
    :param n_dim: int,
        Number of channels.
    :param alpha: float,
        Relocation strength passed to :func:`nonuniform_directions`.
    :return env_mean: ndarray,
        Mean envelope of shape ``(n_samples, n_channels)``.
    :return nem: ndarray,
        Extrema counts per *uniform* direction slot, shape ``(n_dir,)``.
    :return nzm: ndarray,
        Zero-crossing counts per direction slot, shape ``(n_dir,)``.
    :return amp: ndarray,
        Mode-amplitude estimate of shape ``(n_samples,)``.
    """
    nbsym = 2
    count = 0
    env_mean = np.zeros((len(t), n_dim), dtype=float)
    amp = np.zeros(len(t), dtype=float)
    nem = np.zeros(n_dir, dtype=float)
    nzm = np.zeros(n_dir, dtype=float)

    uniform = hammersley_unit_directions(seq, n_dir, n_dim)
    adapted = nonuniform_directions(uniform, m, alpha)

    for it in range(adapted.shape[0]):
        y = np.dot(adapted[it], m.T)
        indmin, indmax = local_peaks(y)
        if it < n_dir:
            nem[it] = len(indmin) + len(indmax)
            indzer = zero_crossings(y)
            nzm[it] = len(indzer)

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

    if n_dir > count:
        env_mean = env_mean / float(n_dir - count)
        amp = amp / float(n_dir - count)
    else:
        env_mean = np.zeros((seq_len, n_dim), dtype=float)
        amp = np.zeros(seq_len, dtype=float)
        nem = np.zeros(n_dir, dtype=float)
    return env_mean, nem, nzm, amp
