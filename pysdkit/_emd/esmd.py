# -*- coding: utf-8 -*-
"""
Extreme-Point Symmetric Mode Decomposition (ESMD).

Wang J.L., Li Z.J. Extreme-point symmetric mode decomposition method for
data analysis. Advances in Adaptive Data Analysis, 5(3):1350015, 2013.
arXiv:1303.6540.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


def find_extrema(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local extrema indices / values (strict + flat-peak friendly).

    Matches the paper's local max/min conditions used by DI / sifting.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 3:
        return np.array([], dtype=int), np.array([])

    idx: List[int] = []
    for k in range(1, n - 1):
        left, mid, right = x[k - 1], x[k], x[k + 1]
        is_max = (mid > left and mid >= right) or (mid >= left and mid > right)
        is_min = (mid < left and mid <= right) or (mid <= left and mid < right)
        if is_max or is_min:
            idx.append(k)
    if not idx:
        return np.array([], dtype=int), np.array([])
    ind = np.asarray(idx, dtype=int)
    return ind, x[ind]


def count_extrema(x: np.ndarray) -> int:
    """Number of interior local extrema."""
    ind, _ = find_extrema(x)
    return int(ind.size)


def _interp_curve(
    t_nodes: np.ndarray, y_nodes: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Cubic spline when possible, else linear; always covers ``t``."""
    t_nodes = np.asarray(t_nodes, dtype=float).ravel()
    y_nodes = np.asarray(y_nodes, dtype=float).ravel()
    order = np.argsort(t_nodes)
    t_nodes, y_nodes = t_nodes[order], y_nodes[order]
    # unique times
    uniq_t, uniq_idx = np.unique(t_nodes, return_index=True)
    t_nodes, y_nodes = uniq_t, y_nodes[uniq_idx]
    if t_nodes.size == 0:
        return np.zeros_like(t, dtype=float)
    if t_nodes.size == 1:
        return np.full_like(t, y_nodes[0], dtype=float)
    if t_nodes.size == 2:
        f = interp1d(
            t_nodes,
            y_nodes,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )
        return f(t).astype(float)
    try:
        cs = CubicSpline(t_nodes, y_nodes, bc_type="natural", extrapolate=True)
        return cs(t).astype(float)
    except ValueError:
        f = interp1d(
            t_nodes,
            y_nodes,
            kind="linear",
            fill_value="extrapolate",
            assume_sorted=True,
        )
        return f(t).astype(float)


def boundary_extrema(
    x: np.ndarray, t: np.ndarray, ext_idx: np.ndarray, ext_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Append left / right boundary extrema (Appendix A / Wu–Huang style).

    Returns augmented ``(indices_as_times, values)`` sorted by time.
    """
    x = np.asarray(x, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    if ext_idx.size < 2:
        # Mirror endpoints as extrema
        times = np.r_[t[0], t[ext_idx], t[-1]] if ext_idx.size else np.r_[t[0], t[-1]]
        vals = np.r_[x[0], ext_val, x[-1]] if ext_idx.size else np.r_[x[0], x[-1]]
        order = np.argsort(times)
        return times[order], vals[order]

    # Separate max / min among interior extrema
    is_max = np.zeros(ext_idx.size, dtype=bool)
    for i, k in enumerate(ext_idx):
        is_max[i] = x[k] >= x[k - 1] and x[k] >= x[k + 1]

    max_i = ext_idx[is_max]
    min_i = ext_idx[~is_max]
    max_v = x[max_i] if max_i.size else np.array([])
    min_v = x[min_i] if min_i.size else np.array([])

    def _side(left: bool) -> Tuple[float, float, float, float]:
        """Return (t_max_b, v_max_b, t_min_b, v_min_b) for one boundary."""
        y0 = x[0] if left else x[-1]
        t0 = t[0] if left else t[-1]

        if max_i.size >= 2 and min_i.size >= 2:
            if left:
                im1, im2 = max_i[0], max_i[1]
                jn1, jn2 = min_i[0], min_i[1]
            else:
                im1, im2 = max_i[-1], max_i[-2]
                jn1, jn2 = min_i[-1], min_i[-2]
            # Line through two maxima / minima, evaluate at boundary time
            k1 = (x[im2] - x[im1]) / (t[im2] - t[im1] + 1e-30)
            b1 = x[im1] - k1 * t[im1]
            k2 = (x[jn2] - x[jn1]) / (t[jn2] - t[jn1] + 1e-30)
            b2 = x[jn1] - k2 * t[jn1]
            u = k1 * t0 + b1  # upper intercept at boundary
            d = k2 * t0 + b2  # lower
            if u < d:
                u, d = d, u

            # Appendix A cases
            if d <= y0 <= u:
                return t0, u, t0, d
            hi = (3.0 * u - d) / 2.0
            lo = (3.0 * d - u) / 2.0
            if u < y0 <= hi:
                return t0, y0, t0, d
            if lo <= y0 < d:
                return t0, u, t0, y0
            if y0 > hi:
                # new lower from (t0,y0) and first max
                im = max_i[0] if left else max_i[-1]
                k_star = (x[im] - y0) / (t[im] - t0 + 1e-30)
                b_star = y0 - k_star * t0
                return t0, y0, t0, k_star * t0 + b_star
            if y0 < lo:
                jn = min_i[0] if left else min_i[-1]
                k_star = (x[jn] - y0) / (t[jn] - t0 + 1e-30)
                b_star = y0 - k_star * t0
                return t0, k_star * t0 + b_star, t0, y0

        # Fallback: endpoint as both (degenerate) — mirror first extremum
        return t0, y0, t0, y0

    tl_max, vl_max, tl_min, vl_min = _side(True)
    tr_max, vr_max, tr_min, vr_min = _side(False)

    times = np.concatenate([[tl_max, tl_min], t[ext_idx], [tr_max, tr_min]])
    vals = np.concatenate([[vl_max, vl_min], ext_val, [vr_max, vr_min]])
    order = np.argsort(times)
    times, vals = times[order], vals[order]
    # drop exact duplicate times (keep average)
    uniq_t, inv = np.unique(np.round(times, decimals=12), return_inverse=True)
    uniq_v = np.zeros(uniq_t.size)
    for i in range(uniq_t.size):
        uniq_v[i] = vals[inv == i].mean()
    return uniq_t, uniq_v


def midpoints_from_extrema(
    ext_t: np.ndarray, ext_v: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Midpoints of segments joining consecutive extrema."""
    if ext_t.size < 2:
        return np.array([]), np.array([])
    mt = 0.5 * (ext_t[:-1] + ext_t[1:])
    mv = 0.5 * (ext_v[:-1] + ext_v[1:])
    return mt, mv


def mean_curve(
    x: np.ndarray,
    t: np.ndarray,
    n_curves: int = 2,
) -> np.ndarray:
    """
    Build the sifting mean ``L*`` from ``n_curves`` midpoint interpolants.

    - ``n_curves=1`` → ESMD_I (all midpoints)
    - ``n_curves=2`` → ESMD_II (odd / even midpoints)
    - ``n_curves=3`` → ESMD_III (residue classes mod 3)
    """
    x = np.asarray(x, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    ext_idx, ext_val = find_extrema(x)
    if ext_idx.size < 2:
        return np.zeros_like(x)

    et, ev = boundary_extrema(x, t, ext_idx, ext_val)
    mt, mv = midpoints_from_extrema(et, ev)
    if mt.size < 2:
        return np.zeros_like(x)

    p = max(1, int(n_curves))
    curves = []
    if p == 1:
        curves.append(_interp_curve(mt, mv, t))
    else:
        for r in range(p):
            sel = np.arange(r, mt.size, p)
            if sel.size < 2:
                # not enough nodes — fall back to all midpoints for this curve
                curves.append(_interp_curve(mt, mv, t))
            else:
                curves.append(_interp_curve(mt[sel], mv[sel], t))
    return np.mean(np.vstack(curves), axis=0)


def sift_mode(
    x: np.ndarray,
    t: np.ndarray,
    n_curves: int = 2,
    max_sift: int = 30,
    eps: float = 1e-6,
) -> np.ndarray:
    """Extract one ESMD mode by repeated midpoint-mean sifting."""
    h = np.asarray(x, dtype=float).ravel().copy()
    t = np.asarray(t, dtype=float).ravel()
    for _ in range(int(max_sift)):
        l_star = mean_curve(h, t, n_curves=n_curves)
        if float(np.max(np.abs(l_star))) <= eps:
            break
        h = h - l_star
        if count_extrema(h) < 2:
            break
    return h


def decompose_fixed_sift(
    x: np.ndarray,
    t: np.ndarray,
    max_sift: int,
    n_curves: int = 2,
    extreme_num_r: int = 4,
    eps: float = 1e-6,
    max_imfs: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full ESMD with fixed maximum sifting times ``K``.

    :return: ``(modes, residual)`` — modes shape ``(n_imf, N)``, residual ``(N,)``
    """
    residual = np.asarray(x, dtype=float).ravel().copy()
    t = np.asarray(t, dtype=float).ravel()
    modes: List[np.ndarray] = []

    for _ in range(int(max_imfs)):
        if count_extrema(residual) <= int(extreme_num_r):
            break
        mode = sift_mode(residual, t, n_curves=n_curves, max_sift=max_sift, eps=eps)
        if np.allclose(mode, 0.0) or not np.any(np.isfinite(mode)):
            break
        # Guard against non-progressing sift
        if np.linalg.norm(mode) < 1e-14 * (np.linalg.norm(residual) + 1e-30):
            break
        modes.append(mode)
        residual = residual - mode

    if not modes:
        return np.zeros((0, residual.size)), residual
    return np.vstack(modes), residual


def variance_ratio(
    x: np.ndarray,
    residual: np.ndarray,
) -> float:
    """``ν = σ(Y−R) / σ(Y)`` (paper Step 7)."""
    x = np.asarray(x, dtype=float).ravel()
    r = np.asarray(residual, dtype=float).ravel()
    s0 = float(np.std(x))
    if s0 < 1e-30:
        return 0.0
    return float(np.std(x - r) / s0)


def scan_variance_ratios(
    x: np.ndarray,
    t: np.ndarray,
    min_sift: int = 1,
    max_sift: int = 40,
    n_curves: int = 2,
    extreme_num_r: int = 4,
    eps: float = 1e-6,
    max_imfs: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan ``K ∈ [min_sift, max_sift]`` and return ``(K_grid, ν(K))``.
    """
    ks = np.arange(int(min_sift), int(max_sift) + 1)
    ratios = np.empty(ks.size, dtype=float)
    for i, k in enumerate(ks):
        _, res = decompose_fixed_sift(
            x,
            t,
            max_sift=int(k),
            n_curves=n_curves,
            extreme_num_r=extreme_num_r,
            eps=eps,
            max_imfs=max_imfs,
        )
        ratios[i] = variance_ratio(x, res)
    return ks, ratios


def instantaneous_amplitude(mode: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Instantaneous amplitude via cubic envelope of ``|extrema|`` (paper §6).
    """
    mode = np.asarray(mode, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    abs_m = np.abs(mode)
    ext_idx, _ = find_extrema(mode)
    if ext_idx.size < 2:
        # fallback: endpoints
        nodes_t = np.array([t[0], t[-1]])
        nodes_a = np.array([abs_m[0], abs_m[-1]])
    else:
        nodes_t = np.r_[t[0], t[ext_idx], t[-1]]
        nodes_a = np.r_[abs_m[0], abs_m[ext_idx], abs_m[-1]]
    amp = _interp_curve(nodes_t, nodes_a, t)
    return np.maximum(amp, 0.0)


def instantaneous_frequency(mode: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Direct Interpolating (DI) instantaneous frequency (paper §6.1, simplified).

    At each interior extremum, place a frequency sample
    ``f = 1 / (t_{i+1} - t_{i-1})`` at ``a = (t_{i+1} + t_{i-1}) / 2``,
    with adjacent-equal extrema mapped to ``f = 0``. Boundaries are
    linearly extrapolated; the output is ``max{0, f(t)}``.
    """
    mode = np.asarray(mode, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    ext_idx, ext_val = find_extrema(mode)
    if ext_idx.size < 3:
        return np.zeros_like(mode)

    a_list: List[float] = []
    f_list: List[float] = []
    te = t[ext_idx]
    ye = ext_val

    for i in range(ext_idx.size):
        # adjacent equal → zero frequency
        equal_prev = i > 0 and abs(ye[i] - ye[i - 1]) < 1e-30
        equal_next = i + 1 < ye.size and abs(ye[i] - ye[i + 1]) < 1e-30
        if equal_prev or equal_next:
            a_list.append(float(te[i]))
            f_list.append(0.0)
            continue
        if 0 < i < ext_idx.size - 1:
            dt = te[i + 1] - te[i - 1]
            if dt > 1e-30:
                a_list.append(0.5 * (te[i + 1] + te[i - 1]))
                f_list.append(1.0 / dt)

    if len(a_list) < 2:
        return np.zeros_like(mode)

    a = np.asarray(a_list, dtype=float)
    f = np.asarray(f_list, dtype=float)
    order = np.argsort(a)
    a, f = a[order], f[order]

    # boundary extension (Step 3)
    if abs(f[0]) < 1e-30:
        a_left, f_left = t[0], 0.0
    else:
        if a.size >= 2:
            f_left = (f[1] - f[0]) * (t[0] - a[0]) / (a[1] - a[0] + 1e-30) + f[0]
        else:
            f_left = f[0]
        if f_left <= 0:
            f_left = 1.0 / (2.0 * (te[1] - te[0] + 1e-30))
        a_left = t[0]

    if abs(f[-1]) < 1e-30:
        a_right, f_right = t[-1], 0.0
    else:
        if a.size >= 2:
            f_right = (f[-1] - f[-2]) * (t[-1] - a[-1]) / (a[-1] - a[-2] + 1e-30) + f[
                -1
            ]
        else:
            f_right = f[-1]
        if f_right <= 0:
            f_right = 1.0 / (2.0 * (te[-1] - te[-2] + 1e-30))
        a_right = t[-1]

    a_all = np.r_[a_left, a, a_right]
    f_all = np.r_[f_left, f, f_right]
    freq = _interp_curve(a_all, f_all, t)
    return np.maximum(freq, 0.0)


def total_energy(amplitudes: np.ndarray) -> np.ndarray:
    """``E(t) = 1/2 Σ A_j(t)^2`` (paper Eq. 8)."""
    a = np.atleast_2d(np.asarray(amplitudes, dtype=float))
    return 0.5 * np.sum(a**2, axis=0)


def make_esmd_example3(
    n: int = 400,
) -> Dict[str, np.ndarray]:
    """
    Paper Example 3 (perfectly separable under ESMD_II)::

        Y(t) = -sin(8πt) + 1.5 e^{-0.2 t} sin(1.9π t + π/20) + (t-2)^2
        0 ≤ t ≤ 4
    """
    t = np.linspace(0.0, 4.0, int(n))
    m1 = -np.sin(8 * np.pi * t)
    m2 = 1.5 * np.exp(-0.2 * t) * np.sin(1.9 * np.pi * t + np.pi / 20.0)
    trend = (t - 2.0) ** 2
    y = m1 + m2 + trend
    return {
        "t": t,
        "signal": y,
        "mode1": m1,
        "mode2": m2,
        "trend": trend,
        "dt": float(t[1] - t[0]),
    }


def load_wind_demo(path: str) -> Dict[str, np.ndarray]:
    """
    Load a wind-demo CSV/TXT file (column 0 = wind series, ``dt=0.05`` s).

    :param path: path to a comma-separated file whose first column is the series
    """
    data = np.loadtxt(path, delimiter=",")
    y = data[:, 0]
    dt = 0.05
    t = np.arange(y.size, dtype=float) * dt
    return {"t": t, "signal": y, "dt": dt}


class ESMD(object):
    """
    Extreme-Point Symmetric Mode Decomposition (ESMD).

    Default ``n_curves=2`` corresponds to **ESMD_II** (odd/even midpoints),
    which the paper recommends.  When ``optimize_sift=True``, the optimal
    sifting count ``K0`` is chosen by minimising the variance ratio
    ``ν = σ(Y−R)/σ(Y)``.
    """

    def __init__(
        self,
        n_curves: int = 2,
        min_sift: int = 1,
        max_sift: int = 40,
        extreme_num_r: int = 4,
        eps_ratio: float = 0.001,
        optimize_sift: bool = True,
        max_imfs: int = 20,
    ) -> None:
        """
        :param n_curves: 1 / 2 / 3 → ESMD_I / II / III
        :param min_sift: lower bound of ``K`` scan
        :param max_sift: upper bound of ``K`` scan / fixed ``K`` if not optimising
        :param extreme_num_r: stop when residual extrema ≤ this
        :param eps_ratio: ``ε = eps_ratio * std(Y)`` mean-curve stop tolerance
        :param optimize_sift: if True, pick ``K0`` by variance-ratio scan
        :param max_imfs: safety cap on number of IMFs
        """
        if int(n_curves) < 1:
            raise ValueError("n_curves must be >= 1")
        self.n_curves = int(n_curves)
        self.min_sift = int(min_sift)
        self.max_sift = int(max_sift)
        self.extreme_num_r = int(extreme_num_r)
        self.eps_ratio = float(eps_ratio)
        self.optimize_sift = bool(optimize_sift)
        self.max_imfs = int(max_imfs)

        self.imfs_: Optional[np.ndarray] = None
        self.residual_: Optional[np.ndarray] = None
        self.opt_sift_: Optional[int] = None
        self.variance_ratios_: Optional[np.ndarray] = None
        self.sift_grid_: Optional[np.ndarray] = None
        self.amplitudes_: Optional[np.ndarray] = None
        self.frequencies_: Optional[np.ndarray] = None
        self.energy_: Optional[np.ndarray] = None
        self.t_: Optional[np.ndarray] = None

    def __call__(
        self,
        signal: np.ndarray,
        dt: float = 1.0,
        return_all: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        return self.fit_transform(signal, dt=dt, return_all=return_all)

    def __str__(self) -> str:
        return "Extreme-Point Symmetric Mode Decomposition (ESMD)"

    def fit_transform(
        self,
        signal: np.ndarray,
        dt: float = 1.0,
        return_all: bool = False,
        compute_di: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Decompose a 1-D real signal.

        :param signal: input series ``Y``
        :param dt: sampling period
        :param return_all: also return residual, amplitudes, frequencies, energy
        :param compute_di: compute DI instantaneous amp / freq / energy
        :return: IMFs stacked with residual as last row ``(n_imf+1, N)``,
                 or extended tuple when ``return_all``
        """
        y = np.asarray(signal, dtype=float).ravel()
        if y.size < 8:
            raise ValueError("signal length must be >= 8")
        dt = float(dt)
        if dt <= 0:
            raise ValueError("dt must be positive")

        t = np.arange(y.size, dtype=float) * dt
        eps = self.eps_ratio * float(np.std(y) + 1e-30)

        if self.optimize_sift:
            ks, ratios = scan_variance_ratios(
                y,
                t,
                min_sift=self.min_sift,
                max_sift=self.max_sift,
                n_curves=self.n_curves,
                extreme_num_r=self.extreme_num_r,
                eps=eps,
                max_imfs=self.max_imfs,
            )
            k0 = int(ks[int(np.argmin(ratios))])
            self.sift_grid_ = ks
            self.variance_ratios_ = ratios
            self.opt_sift_ = k0
        else:
            k0 = self.max_sift
            self.opt_sift_ = k0
            self.sift_grid_ = np.array([k0])
            self.variance_ratios_ = None

        modes, residual = decompose_fixed_sift(
            y,
            t,
            max_sift=k0,
            n_curves=self.n_curves,
            extreme_num_r=self.extreme_num_r,
            eps=eps,
            max_imfs=self.max_imfs,
        )

        if modes.size == 0:
            imfs = residual[np.newaxis, :]
        else:
            imfs = np.vstack([modes, residual[np.newaxis, :]])

        self.imfs_ = imfs
        self.residual_ = residual
        self.t_ = t

        if compute_di and modes.size:
            amps = np.vstack([instantaneous_amplitude(m, t) for m in modes])
            freqs = np.vstack([instantaneous_frequency(m, t) for m in modes])
            energy = total_energy(amps)
        else:
            amps = np.zeros((0, y.size))
            freqs = np.zeros((0, y.size))
            energy = np.zeros(y.size)

        self.amplitudes_ = amps
        self.frequencies_ = freqs
        self.energy_ = energy

        if return_all:
            return imfs, residual, amps, freqs, energy
        return imfs

    def get_variance_ratio(
        self, signal: np.ndarray, dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Scan sifting counts and return ``(K_grid, ν)``."""
        y = np.asarray(signal, dtype=float).ravel()
        t = np.arange(y.size, dtype=float) * float(dt)
        eps = self.eps_ratio * float(np.std(y) + 1e-30)
        ks, ratios = scan_variance_ratios(
            y,
            t,
            min_sift=self.min_sift,
            max_sift=self.max_sift,
            n_curves=self.n_curves,
            extreme_num_r=self.extreme_num_r,
            eps=eps,
            max_imfs=self.max_imfs,
        )
        self.sift_grid_ = ks
        self.variance_ratios_ = ratios
        self.opt_sift_ = int(ks[int(np.argmin(ratios))])
        return ks, ratios


def esmd(
    signal: np.ndarray,
    dt: float = 1.0,
    n_curves: int = 2,
    min_sift: int = 1,
    max_sift: int = 40,
    extreme_num_r: int = 4,
    optimize_sift: bool = True,
    return_all: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """Functional wrapper around :class:`ESMD`."""
    return ESMD(
        n_curves=n_curves,
        min_sift=min_sift,
        max_sift=max_sift,
        extreme_num_r=extreme_num_r,
        optimize_sift=optimize_sift,
    ).fit_transform(signal, dt=dt, return_all=return_all)
