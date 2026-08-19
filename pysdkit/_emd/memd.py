# -*- coding: utf-8 -*-
"""
Multivariate Empirical Mode Decomposition (MEMD).

Faithful Python port of Rehman & Mandic's MATLAB toolbox (``memd.m``),
with Hammersley direction vectors on the (n-1)-sphere.

Rehman, N. and Mandic, D. P. "Multivariate empirical mode decomposition."
Proceedings of the Royal Society A 466.2117 (2010): 1291-1302.

MATLAB: http://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import warnings

import numpy as np
from scipy.interpolate import CubicSpline

# MATLAB ``memd.m`` hard limit (``Max_channels``).
_MAX_CHANNELS = 16
_MIN_CHANNELS = 3
_MIN_N_DIR = 6


class MEMD(object):
    """
    Multivariate Empirical Mode Decomposition (MEMD).

    Projects an n-variate signal onto a Hammersley set of direction vectors
    on the unit ``(n-1)``-sphere, interpolates multivariate envelopes at the
    extrema of those projections, and sifts until a stopping criterion is met.
    All channels share the same number of IMFs (mode alignment).

    Input layout is ``(n_channels, n_samples)`` with ``3 <= n_channels <= 16``.
    A MATLAB-style ``(n_samples, n_channels)`` array is transposed automatically.
    Output IMFs have shape ``(n_imfs, n_samples, n_channels)``; the last slice
    is the residue.

    :param stop_crit: str,
        ``"stop"`` (Rilling et al., 2003) or ``"fix_h"`` (Huang et al., 2003).
    :param max_iter: int,
        Maximum sifting iterations per IMF (MATLAB default 1000).
    :param n_dir: int,
        Number of projection directions (MATLAB default 64). This value is
        **not** replaced by ``2 * n_channels``.
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
        Store MEMD hyperparameters used by later sifting calls.

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
        :return: None
        """
        if not isinstance(stop_crit, str) or stop_crit not in ("stop", "fix_h"):
            raise ValueError(
                "invalid stop_criteria. stop_criteria should be either fix_h or stop"
            )
        self.stop_crit = stop_crit

        if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
            raise ValueError("max_iter should be a positive integer")
        self.max_iter = int(max_iter)

        if not isinstance(n_dir, (int, np.integer)) or int(n_dir) < _MIN_N_DIR:
            raise ValueError(
                "invalid num_dir. num_dir should be an integer greater than or equal to 6."
            )
        self.n_dir = int(n_dir)

        if stop_vec is None:
            stop_vec = np.array([0.075, 0.75, 0.075], dtype=float)
        else:
            stop_vec = np.asarray(stop_vec, dtype=float).ravel()
        if stop_vec.size != 3 or not np.all(np.isfinite(stop_vec)):
            raise ValueError(
                "invalid stop_vector. stop_vector should be an array with three "
                "elements e.g. default is [0.075, 0.75, 0.075]"
            )
        self.stop_vec = stop_vec
        self.sd, self.sd2, self.tol = (
            float(stop_vec[0]),
            float(stop_vec[1]),
            float(stop_vec[2]),
        )

        if not isinstance(stop_cnt, (int, np.integer)) or int(stop_cnt) < 0:
            raise ValueError(
                "invalid stop_count. stop_count should be a non-negative integer."
            )
        self.stop_cnt = int(stop_cnt)

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Allow a ``MEMD`` instance to be called like a function.

        :param signal: ndarray,
            Multivariate record of shape ``(n_channels, n_samples)``, or
            MATLAB-style ``(n_samples, n_channels)``.
        :return imfs: ndarray,
            Decomposition of shape ``(n_imfs, n_samples, n_channels)``.
            The last slice along axis 0 is the residue.
        """
        return self.fit_transform(signal=signal)

    def __str__(self) -> str:
        """
        Return the full algorithm name and abbreviation.

        :return: str,
            ``"Multivariate Empirical Mode Decomposition (MEMD)"``.
        """
        return "Multivariate Empirical Mode Decomposition (MEMD)"

    def init_hammersley(self, n_dim: int) -> np.ndarray:
        """
        Build the Hammersley point set used to construct direction vectors.

        Each row is one direction sample. For trivariate data the sequence has
        two coordinates (shape ``(n_dir, 2)``); otherwise it has one coordinate
        per dimension (shape ``(n_dir, n_dim)``). This is the transpose of the
        MATLAB ``seq(:, it)`` layout.

        :param n_dim: int,
            Number of channels / embedding dimension (``3 <= n_dim <= 16``).
        :return seq: ndarray,
            Low-discrepancy samples in ``[0, 1]`` (the first Hammersley
            coordinate may slightly exceed 1, matching MATLAB ``hamm``).
        """
        n_dir = self.n_dir
        base: List[int] = [-n_dir]

        if n_dim == 3:
            base.append(2)
            seq = np.zeros((n_dir, n_dim - 1), dtype=float)
            for it in range(n_dim - 1):
                seq[:, it] = np.asarray(hamm(n_dir, base[it]), dtype=float).ravel()
            return seq

        primes = nth_prime(n_dim - 1)
        for itr in range(1, n_dim):
            base.append(int(primes[itr - 1]))
        seq = np.zeros((n_dir, n_dim), dtype=float)
        for it in range(n_dim):
            seq[:, it] = np.asarray(hamm(n_dir, base[it]), dtype=float).ravel()
        return seq

    def direction_vectors(self, n_dim: int) -> np.ndarray:
        """
        Convert the Hammersley sequence into unit projection directions.

        Matches MATLAB ``memd.m``: a 2-sphere map when ``n_dim == 3``, and the
        hyperspherical construction otherwise.

        :param n_dim: int,
            Ambient dimension of the multivariate signal.
        :return dirs: ndarray,
            Unit vectors of shape ``(n_dir, n_dim)``. Each row is one
            projection direction on the ``(n_dim - 1)``-sphere.
        """
        seq = self.init_hammersley(n_dim)
        dirs = np.zeros((self.n_dir, n_dim), dtype=float)
        for it in range(self.n_dir):
            dirs[it] = unit_direction(seq, it, n_dim)
        return dirs

    def stop_emd(self, signal: np.ndarray, seq: np.ndarray, n_dim: int) -> bool:
        """
        Decide whether IMF extraction should stop for the current residue.

        The residue is projected onto every Hammersley direction. Extraction
        stops when **all** projections have fewer than three extrema (MATLAB
        ``stop_emd``).

        :param signal: ndarray,
            Current residue of shape ``(n_samples, n_channels)``.
        :param seq: ndarray,
            Hammersley samples from :meth:`init_hammersley`.
        :param n_dim: int,
            Number of channels.
        :return stop_flag: bool,
            ``True`` if no further oscillatory IMF should be extracted.
        """
        ner = np.zeros(self.n_dir, dtype=float)
        for it in range(self.n_dir):
            y = np.dot(signal, unit_direction(seq, it, n_dim))
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

        Computes the envelope mean, forms the relative amplitude
        ``sx = ||env_mean|| / amp``, and stops unless a sufficient fraction of
        samples still exceed ``sd`` / ``sd2`` **and** some projection still has
        more than two extrema.

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
            Local-mean estimate of shape ``(n_samples, n_channels)``. On
            interpolation failure this is zeros (MATLAB ``catch``).
        """
        try:
            env_mean, nem, _nzm, amp = envelope_mean(
                signal, time, seq, self.n_dir, seq_len, n_dim
            )
            sx = np.sqrt(np.sum(env_mean**2, axis=1))
            # MATLAB ``if(amp)`` is true only when every amplitude sample is nonzero.
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

        Counts consecutive siftings for which the numbers of extrema and
        zero-crossings differ by at most one on every projection. Sifting
        stops after ``stop_cnt`` such iterations.

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
            Updated consecutive-success counter (reset to 0 when the
            extrema / zero-crossing test fails).
        """
        try:
            env_mean, nem, nzm, _amp = envelope_mean(
                signal, time, seq, self.n_dir, seq_len, n_dim
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
            ``3 <= n_channels <= 16``. A ``(n_samples, n_channels)`` array is
            accepted and transposed when the channel axis is unambiguous.
        :return imfs: ndarray,
            Stack of shape ``(n_imfs, n_samples, n_channels)``. Oscillatory
            IMFs occupy slices ``0 .. n_imfs-2``; slice ``-1`` is the residue.
            Summing along axis 0 reconstructs the (possibly transposed) input.
        """
        x = _check_multivariate_signal(signal)
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


def _check_multivariate_signal(signal: np.ndarray) -> np.ndarray:
    """
    Validate and orient a multivariate MEMD input.

    Accepts ``(n_channels, n_samples)`` or, when the other layout is the only
    one whose first dimension lies in ``[3, 16]``, MATLAB ``(n_samples, n_channels)``.

    :param signal: array-like,
        Candidate 2-D multivariate record.
    :return x: ndarray,
        Float array of shape ``(n_channels, n_samples)`` with
        ``3 <= n_channels <= 16`` and at least 3 samples.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 2:
        raise ValueError("MEMD expects a 2-D array of shape (n_channels, n_samples)")
    n0, n1 = x.shape
    if not (_MIN_CHANNELS <= n0 <= _MAX_CHANNELS) and (
        _MIN_CHANNELS <= n1 <= _MAX_CHANNELS
    ):
        x = x.T
        n0, n1 = x.shape
    if not (_MIN_CHANNELS <= n0 <= _MAX_CHANNELS):
        raise ValueError(
            "MEMD processes signals with 3 to 16 channels. "
            "Got shape {}; try MVMD for fewer channels.".format(signal.shape)
        )
    if n1 < 3:
        raise ValueError("MEMD requires at least 3 samples")
    return x


def _as_1d(x: np.ndarray) -> np.ndarray:
    """
    Flatten an array to a 1-D float vector.

    :param x: array-like,
        Scalar projection or any array that should be treated as a series.
    :return y: ndarray,
        Contiguous 1-D float copy/view of length ``x.size``.
    """
    return np.asarray(x, dtype=float).reshape(-1)


def unit_direction(seq: np.ndarray, index: int, n_dim: int) -> np.ndarray:
    """
    Map one Hammersley sample to a unit direction vector.

    For ``n_dim == 3`` the MATLAB toolbox maps the 2-D Hammersley sample
    ``(u, v)`` onto the 2-sphere by ``z = 2u-1``, ``φ = 2π v``. Otherwise it
    uses the hyperspherical construction
    ``atan2(sqrt(flip(cumsum(b(end:-1:2).^2))), b(1:end-1))``.

    :param seq: ndarray,
        Hammersley matrix from :meth:`MEMD.init_hammersley`
        (rows = directions).
    :param index: int,
        Row index of the sample to convert (``0 <= index < n_dir``).
    :param n_dim: int,
        Ambient dimension of the direction vector.
    :return dir_vec: ndarray,
        Unit vector of shape ``(n_dim,)``.
    """
    if n_dim == 3:
        tt = float(np.clip(2.0 * seq[index, 0] - 1.0, -1.0, 1.0))
        phirad = float(seq[index, 1] * 2.0 * np.pi)
        st = np.sqrt(max(1.0 - tt * tt, 0.0))
        return np.array([st * np.cos(phirad), st * np.sin(phirad), tt], dtype=float)

    b = 2.0 * seq[index, :] - 1.0
    # MATLAB ``flipud(cumsum(...))`` on a column; ``np.flipud`` is a no-op on 1-D.
    tht = np.arctan2(
        np.sqrt(np.flip(np.cumsum(b[:0:-1] ** 2))),
        b[: n_dim - 1],
    )
    dir_vec = np.cumprod(np.concatenate(([1.0], np.sin(tht))))
    dir_vec = np.asarray(dir_vec[:n_dim], dtype=float)
    dir_vec[: n_dim - 1] = np.cos(tht) * dir_vec[: n_dim - 1]
    return dir_vec


def spherical_coordinate_directions(n_phi: int = 16, n_theta: int = 8) -> np.ndarray:
    """
    Uniform *angular* sampling of the 2-sphere (paper Figure 1a).

    Uses the hyperspherical chart of Rehman & Mandic (2010), eq. (3.2),
    for n = 3: ``x = (cos θ1, sin θ1 cos θ2, sin θ1 sin θ2)``. Equal steps
    in ``(θ1, θ2)`` cluster at the poles and are **not** used inside MEMD;
    they are provided for comparison with Hammersley directions.

    :param n_phi: int,
        Number of azimuthal samples ``θ2 ∈ [0, 2π)`` (must be ``>= 2``).
    :param n_theta: int,
        Number of polar samples ``θ1 ∈ [0, π]`` (must be ``>= 2``).
    :return dirs: ndarray,
        Unit vectors of shape ``(n_theta * n_phi, 3)``.
    """
    if n_phi < 2 or n_theta < 2:
        raise ValueError("n_phi and n_theta must be at least 2")
    theta1 = np.linspace(0.0, np.pi, n_theta)
    theta2 = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dirs = np.empty((n_theta * n_phi, 3), dtype=float)
    k = 0
    for th1 in theta1:
        s1, c1 = np.sin(th1), np.cos(th1)
        for th2 in theta2:
            dirs[k] = (c1, s1 * np.cos(th2), s1 * np.sin(th2))
            k += 1
    return dirs


def local_peaks(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect local minima and maxima of a projected 1-D signal.

    Matches MATLAB ``local_peaks`` in ``memd.m``: plateaus are collapsed to
    their midpoints before the first-difference peak test. A near-zero
    projection (all samples ``< 1e-5``) is treated as identically zero.

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

    # MATLAB: ``if(all(x < 1e-5)); x = zeros(1, length(x)); end``
    if np.all(x < 1e-5):
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

    pks_max, loc_max = peaks(ya)
    pks_min, loc_min = peaks(-ya)

    indmin = a[loc_min] if pks_min.size else np.array([], dtype=int)
    indmax = a[loc_max] if pks_max.size else np.array([], dtype=int)
    return np.asarray(indmin, dtype=int), np.asarray(indmax, dtype=int)


def peaks(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect local maxima from the sign change of the first difference.

    A sample is a peak when the discrete derivative goes from strictly
    positive to strictly negative (MATLAB ``peaks``).

    :param x: ndarray,
        1-D series (flattened if necessary).
    :return pks_max: ndarray,
        Peak amplitudes ``x[locs_max]``.
    :return locs_max: ndarray of int,
        0-based peak indices. Both arrays are empty if ``x`` has fewer
        than 3 samples or no peaks.
    """
    x = _as_1d(x)
    if x.size < 3:
        return np.array([], dtype=float), np.array([], dtype=int)
    dX = np.sign(np.diff(x))
    locs_max = np.where(np.logical_and(dX[:-1] > 0, dX[1:] < 0))[0] + 1
    return x[locs_max], locs_max.astype(int)


def hamm(n: int, base: int) -> np.ndarray:
    """
    One-dimensional Hammersley / van der Corput sequence (MATLAB ``hamm``).

    If ``base > 1`` this is the van der Corput radical-inverse sequence in
    that integer base. If ``base < 1`` (MEMD uses ``base = -n_dir``) the
    MATLAB rule ``(mod(i, -base+1) + 0.5) / (-base)`` is used.

    :param n: int,
        Length of the sequence to generate.
    :param base: int,
        Sequence base. Positive primes give Halton coordinates; a negative
        value encodes the explicit Hammersley first coordinate.
    :return seq: ndarray,
        1-D float array of length ``n`` with values typically in ``(0, 1]``.
    """
    n = int(n)
    if 1 < base:
        seq = np.zeros(n, dtype=float)
        seed = np.arange(1, n + 1, dtype=float)
        base_inv = 1.0 / float(base)
        while np.any(seed != 0):
            digit = np.remainder(seed[:n], base)
            seq = seq + digit * base_inv
            base_inv = base_inv / float(base)
            seed = np.floor(seed / float(base))
        return seq
    temp = np.arange(1, n + 1, dtype=float)
    return (np.remainder(temp, (-base + 1)) + 0.5) / float(-base)


def nth_prime(n: int) -> List[int]:
    """
    Return the first ``n`` prime numbers (Hammersley bases for ``n_dim > 3``).

    :param n: int,
        How many primes to generate. ``n < 1`` yields an empty list.
    :return lst: list of int,
        ``[2, 3, 5, ...]`` of length ``n``.
    """
    n = int(n)
    if n < 1:
        return []
    lst: List[int] = [2]
    candidate = 3
    while len(lst) < n:
        if is_prime(candidate):
            lst.append(candidate)
        candidate += 2
        if candidate > 10**7:
            raise RuntimeError("nth_prime: exceeded search bound")
    return lst


def is_prime(x: int) -> bool:
    """
    Test whether an integer is prime.

    :param x: int,
        Candidate integer.
    :return: bool,
        ``True`` if ``x`` is a prime number (``x >= 2`` with no divisor
        other than 1 and itself), otherwise ``False``.
    """
    x = int(x)
    if x < 2:
        return False
    if x == 2:
        return True
    if x % 2 == 0:
        return False
    limit = int(x**0.5) + 1
    for number in range(3, limit, 2):
        if x % number == 0:
            return False
    return True


def envelope_mean(
    m: np.ndarray,
    t: np.ndarray,
    seq: np.ndarray,
    n_dir: int,
    seq_len: int,
    n_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Average multivariate envelopes over all Hammersley projections.

    For each direction the residue is projected, extrema of the scalar
    projection are mirror-extended, and cubic splines of the *vector* samples
    at those times form upper / lower envelopes. Envelopes that cannot be
    built (fewer than 3 extrema) are skipped.

    :param m: ndarray,
        Current proto-IMF of shape ``(n_samples, n_channels)``.
    :param t: ndarray,
        Sample indices of length ``n_samples``.
    :param seq: ndarray,
        Hammersley samples from :meth:`MEMD.init_hammersley`.
    :param n_dir: int,
        Number of projection directions.
    :param seq_len: int,
        Number of time samples (used if every projection is skipped).
    :param n_dim: int,
        Number of channels.
    :return env_mean: ndarray,
        Mean envelope of shape ``(n_samples, n_channels)``.
    :return nem: ndarray,
        Extrema counts per direction, shape ``(n_dir,)``.
    :return nzm: ndarray,
        Zero-crossing counts per direction, shape ``(n_dir,)``.
    :return amp: ndarray,
        Mode-amplitude estimate of shape ``(n_samples,)``.
    """
    nbsym = 2
    count = 0
    env_mean = np.zeros((len(t), n_dim), dtype=float)
    amp = np.zeros(len(t), dtype=float)
    nem = np.zeros(n_dir, dtype=float)
    nzm = np.zeros(n_dir, dtype=float)

    for it in range(n_dir):
        y = np.dot(m, unit_direction(seq, it, n_dim))
        indmin, indmax = local_peaks(y)
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


def zero_crossings(x: np.ndarray) -> np.ndarray:
    """
    Locate zero-crossings of a 1-D series.

    Sign changes between consecutive samples are reported, together with
    exact zeros. A run of consecutive zeros contributes its midpoint index
    (MATLAB ``zero_crossings``).

    :param x: ndarray,
        Scalar series (flattened if necessary).
    :return indzer: ndarray of int,
        Sorted 0-based indices of zero-crossings (empty if none).
    """
    x = _as_1d(x)
    if x.size < 2:
        return np.array([], dtype=int)

    indzer = np.where(x[:-1] * x[1:] < 0)[0]
    if np.any(x == 0):
        iz = np.where(x == 0)[0]
        if iz.size > 1 and np.any(np.diff(iz) == 1):
            zer = (x == 0).astype(np.int8)
            dz = np.diff(np.concatenate(([0], zer, [0])))
            debz = np.where(dz == 1)[0]
            finz = np.where(dz == -1)[0] - 1
            indz = np.round((debz + finz) / 2.0).astype(int)
        else:
            indz = iz
        indzer = np.unique(np.concatenate((indzer, indz)))
    return np.asarray(indzer, dtype=int)


def boundary_conditions(
    indmin: np.ndarray,
    indmax: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    nbsym: int,
) -> Union[
    Tuple[None, None, None, None, int],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int],
]:
    """
    Mirror-extend extrema so cubic splines are defined at the signal edges.

    0-based analogue of MATLAB ``boundary_conditions`` in ``memd.m``. If the
    projection has fewer than three extrema, interpolation is disabled.

    :param indmin: ndarray,
        0-based indices of minima of the projected signal ``x``.
    :param indmax: ndarray,
        0-based indices of maxima of the projected signal ``x``.
    :param t: ndarray,
        Sample indices of length ``N`` (typically ``1, 2, ..., N``).
    :param x: ndarray,
        Projected 1-D signal of length ``N``.
    :param z: ndarray,
        Multivariate samples of shape ``(N, n_channels)`` whose values at the
        extrema are interpolated.
    :param nbsym: int,
        Number of extrema to reflect at each edge (MATLAB uses 2).
    :return tmin: ndarray or None,
        Extended time knots of the minima (``None`` if ``mode == 0``).
    :return tmax: ndarray or None,
        Extended time knots of the maxima.
    :return zmin: ndarray or None,
        Multivariate values at ``tmin``, shape ``(n_knots, n_channels)``.
    :return zmax: ndarray or None,
        Multivariate values at ``tmax``.
    :return mode: int,
        ``1`` if envelopes can be built, ``0`` if there are too few extrema.
    """
    x = _as_1d(x)
    t = np.asarray(t, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    lx = x.size - 1
    indmin = np.asarray(indmin, dtype=int).reshape(-1)
    indmax = np.asarray(indmax, dtype=int).reshape(-1)
    end_max = indmax.size - 1
    end_min = indmin.size - 1

    if indmin.size + indmax.size < 3:
        return None, None, None, None, 0

    mode = 1

    if indmax[0] < indmin[0]:
        if x[0] > x[indmin[0]]:
            lmax = np.flipud(indmax[1 : min(end_max + 1, nbsym + 1)])
            lmin = np.flipud(indmin[: min(end_min + 1, nbsym)])
            lsym = int(indmax[0])
        else:
            lmax = np.flipud(indmax[: min(end_max + 1, nbsym)])
            lmin = np.concatenate(
                (np.flipud(indmin[: min(end_min + 1, nbsym - 1)]), [0])
            )
            lsym = 0
    else:
        if x[0] < x[indmax[0]]:
            lmax = np.flipud(indmax[: min(end_max + 1, nbsym)])
            lmin = np.flipud(indmin[1 : min(end_min + 1, nbsym + 1)])
            lsym = int(indmin[0])
        else:
            lmax = np.concatenate(
                (np.flipud(indmax[: min(end_max + 1, nbsym - 1)]), [0])
            )
            lmin = np.flipud(indmin[: min(end_min + 1, nbsym)])
            lsym = 0

    if indmax[-1] < indmin[-1]:
        if x[-1] < x[indmax[-1]]:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0) :])
            rmin = np.flipud(indmin[max(end_min - nbsym, 0) : -1])
            rsym = int(indmin[-1])
        else:
            rmax = np.concatenate(
                ([lx], np.flipud(indmax[max(end_max - nbsym + 2, 0) :]))
            )
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0) :])
            rsym = lx
    else:
        if x[-1] > x[indmin[-1]]:
            rmax = np.flipud(indmax[max(end_max - nbsym, 0) : -1])
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0) :])
            rsym = int(indmax[-1])
        else:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0) :])
            rmin = np.concatenate(
                ([lx], np.flipud(indmin[max(end_min - nbsym + 2, 0) :]))
            )
            rsym = lx

    lmin = np.asarray(lmin, dtype=int).reshape(-1)
    lmax = np.asarray(lmax, dtype=int).reshape(-1)
    rmin = np.asarray(rmin, dtype=int).reshape(-1)
    rmax = np.asarray(rmax, dtype=int).reshape(-1)

    tlmin = 2 * t[lsym] - t[lmin]
    tlmax = 2 * t[lsym] - t[lmax]
    trmin = 2 * t[rsym] - t[rmin]
    trmax = 2 * t[rsym] - t[rmax]

    if tlmin[0] > t[0] or tlmax[0] > t[0]:
        if lsym == indmax[0]:
            lmax = np.flipud(indmax[: min(end_max + 1, nbsym)]).astype(int)
        else:
            lmin = np.flipud(indmin[: min(end_min + 1, nbsym)]).astype(int)
        if lsym == 0:
            raise RuntimeError("MEMD boundary_conditions: left-edge mirror failed")
        lsym = 0
        tlmin = 2 * t[lsym] - t[lmin]
        tlmax = 2 * t[lsym] - t[lmax]

    if trmin[-1] < t[lx] or trmax[-1] < t[lx]:
        if rsym == indmax[-1]:
            rmax = np.flipud(indmax[max(end_max - nbsym + 1, 0) :]).astype(int)
        else:
            rmin = np.flipud(indmin[max(end_min - nbsym + 1, 0) :]).astype(int)
        if rsym == lx:
            raise RuntimeError("MEMD boundary_conditions: right-edge mirror failed")
        rsym = lx
        trmin = 2 * t[rsym] - t[rmin]
        trmax = 2 * t[rsym] - t[rmax]

    zlmax = z[lmax, :]
    zlmin = z[lmin, :]
    zrmax = z[rmax, :]
    zrmin = z[rmin, :]

    tmin = np.hstack((tlmin, t[indmin], trmin))
    tmax = np.hstack((tlmax, t[indmax], trmax))
    zmin = np.vstack((zlmin, z[indmin, :], zrmin))
    zmax = np.vstack((zlmax, z[indmax, :], zrmax))
    return tmin, tmax, zmin, zmax, mode
