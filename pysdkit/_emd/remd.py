# -*- coding: utf-8 -*-
"""
Robust Empirical Mode Decomposition (REMD).

Improved EMD powered by the Soft Sifting Stopping Criterion (SSSC).

Peng, D., Liu, Z., Jin, Y., Qin, Y.
Improved EMD with a Soft Sifting Stopping Criterion and Its Application to
Fault Diagnosis of Rotating Machinery. Journal of Mechanical Engineering, 2019.

Liu, Z., Peng, D., Zuo, M. J., Xia, J., Qin, Y.
Improved Hilbert-Huang transform with soft sifting stopping criterion and its
application to fault diagnosis of wheelset bearings.
ISA Transactions, 125:426–444, 2022.

Faithful Python port of the MATLAB File Exchange toolbox
``emd_sssc.m`` (https://www.mathworks.com/matlabcentral/fileexchange/70032).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import kurtosis


ArrayLike = Union[np.ndarray, float]


def extr(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract extrema and zero-crossing indices (MATLAB ``extr``).

    Port of Rilling / Flandrin EMD toolbox extrema detection as used in
    ``emd_sssc.m``. Indices are **0-based**.

    :param x: 1-D real signal
    :return: ``(indmin, indmax, indzer)``
    """
    x = np.asarray(x, dtype=float).ravel()
    m = x.size

    # zero crossings
    x1, x2 = x[:-1], x[1:]
    indzer = np.where(x1 * x2 < 0)[0]
    if np.any(x == 0):
        iz = np.where(x == 0)[0]
        if iz.size > 1 and np.any(np.diff(iz) == 1):
            zer = x == 0
            dz = np.diff(np.concatenate([[0], zer.astype(int), [0]]))
            debz = np.where(dz == 1)[0]
            finz = np.where(dz == -1)[0] - 1
            indz = np.round((debz + finz) / 2.0).astype(int)
        else:
            indz = iz
        indzer = np.unique(np.concatenate([indzer, indz]))

    d = np.diff(x)
    n = d.size
    d1, d2 = d[:-1], d[1:]
    indmin = np.where((d1 * d2 < 0) & (d1 < 0))[0] + 1
    indmax = np.where((d1 * d2 < 0) & (d1 > 0))[0] + 1

    # Plateau handling matches MATLAB ``emd_sssc.m``: the imax/imin appends
    # in the File Exchange source are commented out, so plateaus are ignored.

    return (
        np.asarray(indmin, dtype=int),
        np.asarray(indmax, dtype=int),
        np.asarray(indzer, dtype=int),
    )


def extend(
    x: np.ndarray,
    indmin: np.ndarray,
    indmax: np.ndarray,
    ext_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mirror-extend extrema to reduce end effects (MATLAB ``extend``).

    Input/output extrema indices are **0-based**. Internally the port mirrors
    the MATLAB 1-based indexing so boundary slices match ``emd_sssc.m``.

    Returns ``(ext_indmin, ext_indmax, ext_x, cut_index)`` with
    ``cut_index = [start, stop)`` for Python slicing back to the original
    support.
    """
    x = np.asarray(x, dtype=float).ravel()
    # work in MATLAB 1-based indices
    indmin = np.asarray(indmin, dtype=int).ravel() + 1
    indmax = np.asarray(indmax, dtype=int).ravel() + 1
    xlen = int(x.size)

    if ext_ratio == 0 or indmin.size == 0 or indmax.size == 0:
        return (
            indmin - 1,
            indmax - 1,
            x.copy(),
            np.array([0, xlen], dtype=int),
        )

    nbsym = int(np.ceil(float(ext_ratio) * len(indmax)))
    nbsym = max(nbsym, 1)
    t = np.arange(1, xlen + 1, dtype=int)

    # ---- left ----
    if indmax[0] < indmin[0]:
        if x[0] > x[indmin[0] - 1]:
            lmax = np.flip(indmax[1 : min(len(indmax), nbsym + 1)])
            lmin = np.flip(indmin[: min(len(indmin), nbsym)])
            lsym = int(indmax[0])
        else:
            lmax = np.flip(indmax[: min(len(indmax), nbsym)])
            lmin = np.concatenate(
                [np.flip(indmin[: min(len(indmin), max(nbsym - 1, 0))]), [1]]
            )
            lsym = 1
    else:
        if x[0] < x[indmax[0] - 1]:
            lmax = np.flip(indmax[: min(len(indmax), nbsym)])
            lmin = np.flip(indmin[1 : min(len(indmin), nbsym + 1)])
            lsym = int(indmin[0])
        else:
            lmax = np.concatenate(
                [np.flip(indmax[: min(len(indmax), max(nbsym - 1, 0))]), [1]]
            )
            lmin = np.flip(indmin[: min(len(indmin), nbsym)])
            lsym = 1

    # ---- right ----
    if indmax[-1] < indmin[-1]:
        if x[-1] < x[indmax[-1] - 1]:
            rmax = np.flip(indmax[max(len(indmax) - nbsym + 1, 1) - 1 :])
            rmin = np.flip(indmin[max(len(indmin) - nbsym, 1) - 1 : len(indmin) - 1])
            rsym = int(indmin[-1])
        else:
            rmax = np.concatenate(
                [
                    [xlen],
                    np.flip(indmax[max(len(indmax) - nbsym + 2, 1) - 1 :]),
                ]
            )
            rmin = np.flip(indmin[max(len(indmin) - nbsym + 1, 1) - 1 :])
            rsym = xlen
    else:
        if x[-1] > x[indmin[-1] - 1]:
            rmax = np.flip(indmax[max(len(indmax) - nbsym, 1) - 1 : len(indmax) - 1])
            rmin = np.flip(indmin[max(len(indmin) - nbsym + 1, 1) - 1 :])
            rsym = int(indmax[-1])
        else:
            rmax = np.flip(indmax[max(len(indmax) - nbsym + 1, 1) - 1 :])
            rmin = np.concatenate(
                [
                    [xlen],
                    np.flip(indmin[max(len(indmin) - nbsym + 2, 1) - 1 :]),
                ]
            )
            rsym = xlen

    tlmin = 2 * t[lsym - 1] - t[lmin - 1]
    tlmax = 2 * t[lsym - 1] - t[lmax - 1]
    trmin = 2 * t[rsym - 1] - t[rmin - 1]
    trmax = 2 * t[rsym - 1] - t[rmax - 1]

    if (tlmin.size and tlmin[0] > t[0]) or (tlmax.size and tlmax[0] > t[0]):
        if lsym == indmax[0]:
            lmax = np.flip(indmax[: min(len(indmax), nbsym)])
        else:
            lmin = np.flip(indmin[: min(len(indmin), nbsym)])
        if lsym == 1:
            raise ValueError("extend: left-boundary bug (lsym == 1)")
        lsym = 1

    if (trmin.size and trmin[-1] < t[xlen - 1]) or (
        trmax.size and trmax[-1] < t[xlen - 1]
    ):
        if rsym == indmax[-1]:
            rmax = np.flip(indmax[max(len(indmax) - nbsym + 1, 1) - 1 :])
        else:
            rmin = np.flip(indmin[max(len(indmin) - nbsym + 1, 1) - 1 :])
        if rsym == xlen:
            raise ValueError("extend: right-boundary bug (rsym == xlen)")
        rsym = xlen

    l_end = int(max(int(np.max(lmax)), int(np.max(lmin))))
    r_end = int(min(int(np.min(rmax)), int(np.min(rmin))))

    new_lmax = l_end + 1 - lmax
    new_lmin = l_end + 1 - lmin
    new_rmax = rsym - rmax
    new_rmin = rsym - rmin
    lx_length = l_end - lsym
    # MATLAB: lx = fliplr(x(lsym+1:l_end)); rx = fliplr(x(r_end:rsym-1));
    lx = np.flip(x[lsym:l_end])  # 0-based slice of 1-based (lsym+1:l_end)
    rx = np.flip(x[r_end - 1 : rsym - 1])

    ext_x = np.concatenate([lx, x[lsym - 1 : rsym], rx])
    ext_indmin = np.concatenate(
        [
            new_lmin,
            indmin + lx_length - lsym + 1,
            new_rmin + lx_length - lsym + 1 + rsym,
        ]
    ).astype(int)
    ext_indmax = np.concatenate(
        [
            new_lmax,
            indmax + lx_length - lsym + 1,
            new_rmax + lx_length - lsym + 1 + rsym,
        ]
    ).astype(int)

    # MATLAB inclusive cut -> Python [start, stop)
    cut_lo = lx_length - lsym + 2  # 1-based
    cut_hi = xlen + lx_length - lsym + 1  # 1-based inclusive
    cut_index = np.array([cut_lo - 1, cut_hi], dtype=int)
    return ext_indmin - 1, ext_indmax - 1, ext_x, cut_index


def emd_mean(
    x: np.ndarray,
    ext_ratio: float = 0.2,
    smooth_mode: str = "spline",
) -> Tuple[np.ndarray, int]:
    """
    Local mean via spline envelopes on a mirror-extended signal
    (MATLAB ``emd_mean``).

    :return: ``(m, n_extr)``; ``m`` is empty if fewer than 3 extrema.
    """
    x = np.asarray(x, dtype=float).ravel()
    indmin, indmax, _ = extr(x)
    n_extr = int(indmin.size + indmax.size)
    if n_extr < 3:
        return np.array([], dtype=float), n_extr

    if smooth_mode != "spline":
        raise ValueError("smooth_mode must be 'spline' (MATLAB default)")

    ext_indmin, ext_indmax, ext_x, cut_index = extend(x, indmin, indmax, ext_ratio)
    l = ext_x.size

    # drop endpoints from extrema lists (MATLAB)
    ext_indmax = ext_indmax[(ext_indmax != 0) & (ext_indmax != l - 1)]
    ext_indmin = ext_indmin[(ext_indmin != 0) & (ext_indmin != l - 1)]

    xx = np.arange(l, dtype=float)
    max_xp = np.concatenate([[0], ext_indmax, [l - 1]])
    max_yp = ext_x[np.concatenate([[0], ext_indmax, [l - 1]]).astype(int)]
    # MATLAB min envelope uses ext_x([1, ext_indmin, 1]) i.e. both ends = first sample
    min_xp = np.concatenate([[0], ext_indmin, [l - 1]])
    min_yp = np.concatenate([[ext_x[0]], ext_x[ext_indmin.astype(int)], [ext_x[0]]])

    # unique/sorted knots for CubicSpline
    def _unique_knots(xp: np.ndarray, yp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        order = np.argsort(xp)
        xp, yp = xp[order], yp[order]
        uniq_x, idx = np.unique(xp, return_index=True)
        return uniq_x, yp[idx]

    max_xp, max_yp = _unique_knots(max_xp.astype(float), max_yp)
    min_xp, min_yp = _unique_knots(min_xp.astype(float), min_yp)

    if max_xp.size < 2 or min_xp.size < 2:
        return np.array([], dtype=float), n_extr

    max_ext = CubicSpline(max_xp, max_yp, extrapolate=True)(xx)
    min_ext = CubicSpline(min_xp, min_yp, extrapolate=True)(xx)
    ext_m = 0.5 * (max_ext + min_ext)
    m = ext_m[cut_index[0] : cut_index[1]]
    return np.asarray(m, dtype=float), n_extr


def stop_emd(xs: np.ndarray, x_energy: float) -> bool:
    """Stop outer EMD loop (MATLAB ``stopemd``)."""
    indmin, indmax, _ = extr(xs)
    peak = indmin.size + indmax.size
    ratio = float(np.sum(xs**2) / (x_energy + np.finfo(float).eps))
    return peak < 3 or ratio < 0.001


def is_sifting_process_stop(
    m: np.ndarray,
    s: np.ndarray,
    j: int,
    fv_i: np.ndarray,
    ssc: str = "liu",
) -> Tuple[bool, np.ndarray]:
    """
    Soft sifting stopping criterion (MATLAB ``is_sifting_process_stop``).

    ``j`` is **1-based** iteration counter (as in MATLAB).
    """
    fv_i = np.asarray(fv_i, dtype=float).copy()
    df = np.asarray(m, dtype=float).ravel()
    indmin, indmax, indzer = extr(s)
    nem = int(indmin.size + indmax.size)
    nzm = int(indzer.size)
    stop = False

    if ssc == "liu":
        # MATLAB kurtosis is Pearson (normal -> 3); scipy default is excess
        fv_i[j - 1] = float(np.sqrt(np.mean(df**2))) + abs(
            float(kurtosis(df, fisher=False, bias=True) - 3.0)
        )
        if j >= 3 and abs(nzm - nem) < 2:
            if (fv_i[j - 1] >= fv_i[j - 2]) and (fv_i[j - 2] >= fv_i[j - 3]):
                stop = True
    else:
        raise ValueError(f"Unknown sifting stopping criterion: {ssc!r}")

    return stop, fv_i


def index_of_orthogonality(x: np.ndarray, imf: np.ndarray) -> float:
    """Index of orthogonality (MATLAB ``io``)."""
    x = np.asarray(x, dtype=float).ravel()
    imf = np.asarray(imf, dtype=float)
    if imf.ndim == 1:
        imf = imf.reshape(1, -1)
    n = imf.shape[0]
    denom = float(np.sum(x**2)) + np.finfo(float).eps
    s = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                s += abs(float(np.sum(imf[i] * np.conj(imf[j]))) / denom)
    return 0.5 * s


def remd(
    x: np.ndarray,
    max_imfs: int = 10,
    max_iter: int = 30,
    ext_ratio: float = 0.2,
    ssc: str = "liu",
    smooth_mode: str = "spline",
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Functional REMD / ``emd_sssc`` interface.

    :return: ``(imf, ort, fvs, iter_num)`` where the last IMF row is the residual
    """
    x = np.asarray(x, dtype=float).ravel()
    nx = x.size
    x_energy = float(np.sum(x**2))
    imf = np.zeros((max_imfs, nx), dtype=float)
    iter_num = np.zeros(max_imfs, dtype=float)
    fvs = np.zeros((max_imfs, max_iter), dtype=float)

    i = 0
    xs = x.copy()
    while i < max_imfs and not stop_emd(xs, x_energy):
        s_j = np.zeros((max_iter, nx), dtype=float)
        j = 0
        stop_flag = False
        s = xs.copy()
        m_j = np.array([])
        n_extr = 0

        while j < max_iter and not stop_flag:
            j += 1
            if j == 1:
                m_j, n_extr = emd_mean(s, ext_ratio=ext_ratio, smooth_mode=smooth_mode)
            if n_extr < 3:
                break
            s = s - m_j
            s_j[j - 1, :] = s
            m_j, n_extr = emd_mean(s, ext_ratio=ext_ratio, smooth_mode=smooth_mode)
            if m_j.size == 0:
                break
            stop_flag, fvs[i, :] = is_sifting_process_stop(
                m_j, s, j, fvs[i, :], ssc=ssc
            )

        if j == 0:
            break

        if ssc == "liu":
            filled = fvs[i, :j]
            # ignore unused zeros if broken early before writing fv
            valid = filled.copy()
            # MATLAB: [~, opt0] = min(fvs(i,1:j)); opt = min(j, opt0)
            opt0 = int(np.argmin(valid)) + 1  # 1-based
            opt_iter = min(j, opt0)
        else:
            opt_iter = j

        imf[i, :] = s_j[opt_iter - 1, :]
        xs = xs - imf[i, :]
        iter_num[i] = opt_iter
        i += 1

    # residual
    if i < max_imfs:
        imf[i, :] = xs
        n_rows = i + 1
    else:
        # hit max_imfs IMFs; append residual by replacing unused? MATLAB keeps residual at i+1
        # If i == max_imfs, residual would need an extra row — match MATLAB by expanding
        imf = np.vstack([imf, xs.reshape(1, -1)])
        n_rows = max_imfs + 1

    imf = imf[:n_rows, :]
    fvs = fvs[: max(i, 1), :]
    iter_num = iter_num[:i]
    ort = index_of_orthogonality(x, imf)
    return imf, ort, fvs, iter_num


class REMD(object):
    """
    Robust Empirical Mode Decomposition (REMD / EMD-SSSC).

    Soft sifting stopping criterion (Liu / Peng) selects a locally optimal
    sifting iteration instead of a fixed SD threshold.
    """

    def __init__(
        self,
        max_imfs: int = 10,
        max_iter: int = 30,
        ext_ratio: float = 0.2,
        ssc: str = "liu",
        smooth_mode: str = "spline",
    ) -> None:
        """
        :param max_imfs: maximum number of IMFs (excluding residual), as in MATLAB
        :param max_iter: maximum sifting iterations per IMF
        :param ext_ratio: end-extension length ratio ``(0, 1]``; ``0`` disables
        :param ssc: sifting stopping criterion (``'liu'``)
        :param smooth_mode: envelope smoother (``'spline'``)
        """
        self.max_imfs = int(max_imfs)
        self.max_iter = int(max_iter)
        self.ext_ratio = float(ext_ratio)
        self.ssc = str(ssc).lower()
        self.smooth_mode = str(smooth_mode).lower()

        self.imfs: Optional[np.ndarray] = None
        self.ort: Optional[float] = None
        self.fvs: Optional[np.ndarray] = None
        self.iter_num: Optional[np.ndarray] = None
        self.signal: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return "Robust Empirical Mode Decomposition (REMD)"

    def __call__(self, signal: np.ndarray, return_all: bool = False):
        return self.fit_transform(signal, return_all=return_all)

    def fit_transform(
        self, signal: np.ndarray, return_all: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float, np.ndarray, np.ndarray]]:
        """
        Decompose ``signal`` into IMFs + residual.

        :param signal: 1-D real array
        :param return_all: if True, also return ``(ort, fvs, iter_num)``
        :return: IMF matrix ``(n_imfs+1, n)`` (last row = residual)
        """
        imf, ort, fvs, iter_num = remd(
            signal,
            max_imfs=self.max_imfs,
            max_iter=self.max_iter,
            ext_ratio=self.ext_ratio,
            ssc=self.ssc,
            smooth_mode=self.smooth_mode,
        )
        self.signal = np.asarray(signal, dtype=float).ravel()
        self.imfs = imf
        self.ort = ort
        self.fvs = fvs
        self.iter_num = iter_num
        if return_all:
            return imf, ort, fvs, iter_num
        return imf
