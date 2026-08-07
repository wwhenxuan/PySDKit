# -*- coding: utf-8 -*-
"""
Improved Variational Generalized Nonlinear Mode Decomposition (IVGNMD).

IVGNMD improves VGNMD for *crossed* chirp and dispersive modes by replacing
per-cluster ridge picking with a TF-skeleton pipeline:

1. **ATFFC** — adaptive TF fusion → enhanced binary TFD
2. **SE** — improved skeleton extraction (thin / spur / boundary extend)
3. **TFSC** — skeleton cutting at junctions
4. **TFST** — weighted directional skeleton tracking
5. **MTDC** — mode-type discrimination on each tracked path
6. **VOA** — ACMD (chirp) or GDMD (dispersive)

Wang H., Chen S., Zhai W.
Improved variational generalized nonlinear mode decomposition for separating
crossed chirp modes and dispersive modes of non-stationary signals in
mechanical systems. Mechanical Systems and Signal Processing, 2025.

MATLAB toolbox: ``IVGNMD.m``, ``SE.m``, ``TFSC.m``, ``TFST.m``, ``TFPTD.m``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_fill_holes

from pysdkit._gdmd.gdmd import curve_smooth, spectrum_to_time
from pysdkit._gdmd.vgnmd import (
    acmd_single,
    find_ridges_peak,
    gdmd_on_frequency,
    stft_vgnmd,
)


# ---------------------------------------------------------------------------
# Morphology helpers (approximate MATLAB bwmorph / imfill / bwboundaries)
# ---------------------------------------------------------------------------


def _as_bool(img: np.ndarray) -> np.ndarray:
    return np.asarray(img) > 0


def zhang_suen_thin(binary: np.ndarray, max_iter: int = 10000) -> np.ndarray:
    """Zhang–Suen thinning approximating MATLAB ``bwmorph(..., 'thin', Inf)``."""
    img = _as_bool(binary).astype(np.uint8)
    it = 0
    while it < max_iter:
        it += 1
        changed = False
        for step in (0, 1):
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            p1 = img[1:-1, 1:-1]
            neighbours = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(neighbours)
            transitions = np.zeros_like(p1, dtype=np.uint8)
            for k in range(7):
                transitions += ((neighbours[k] == 0) & (neighbours[k + 1] == 1)).astype(
                    np.uint8
                )
            transitions += ((neighbours[7] == 0) & (neighbours[0] == 1)).astype(
                np.uint8
            )
            cond = (p1 == 1) & (b >= 2) & (b <= 6) & (transitions == 1)
            if step == 0:
                cond &= (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond &= (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if np.any(cond):
                changed = True
                img[1:-1, 1:-1][cond] = 0
        if not changed:
            break
    return img.astype(bool)


def bwmorph_remove(binary: np.ndarray) -> np.ndarray:
    """MATLAB ``bwmorph(I, 'remove')`` — keep boundary pixels only."""
    img = _as_bool(binary)
    if not np.any(img):
        return img.copy()
    eroded = binary_erosion(img, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return img & ~eroded


def bwmorph_spur(binary: np.ndarray, n_iter: int = 10) -> np.ndarray:
    """Iteratively remove endpoints (MATLAB ``bwmorph(..., 'spur', n)``)."""
    img = _as_bool(binary).copy()
    for _ in range(int(n_iter)):
        ys, xs = np.nonzero(img)
        endpoints = []
        m, n = img.shape
        for i, j in zip(ys.tolist(), xs.tolist()):
            if i < 1 or j < 1 or i >= m - 1 or j >= n - 1:
                continue
            t, _ = tfptd(img, i, j)
            if t == 1:
                endpoints.append((i, j))
        if not endpoints:
            break
        for i, j in endpoints:
            img[i, j] = False
    return img


def label_components(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected labeling (MATLAB ``bwboundaries`` / ``bwlabel`` style)."""
    structure = np.ones((3, 3), dtype=bool)
    labeled, n_lab = ndimage.label(_as_bool(binary), structure=structure)
    return labeled, int(n_lab)


# ---------------------------------------------------------------------------
# TF point type discrimination
# ---------------------------------------------------------------------------


def tfptd(spec: np.ndarray, i: int, j: int) -> Tuple[int, np.ndarray]:
    """
    Eight-neighbour point-type discrimination (``TFPTD.m``).

    Indices ``i, j`` are **0-based** array coordinates (callers must stay
    interior so ``i±1``, ``j±1`` are valid).

    :return: ``(T, A)`` with ``A`` shape ``(5, 8)``
    """
    a = np.zeros((5, 8), dtype=float)
    # (di, dj, angle, reverse_angle) for 8 neighbours — matches MATLAB order
    neigh = [
        (-1, -1, -135.0, 45.0),
        (-1, 0, -90.0, 90.0),
        (-1, 1, -45.0, 135.0),
        (0, 1, 0.0, 180.0),
        (1, 1, 45.0, -135.0),
        (1, 0, 90.0, -90.0),
        (1, -1, 135.0, -45.0),
        (0, -1, 180.0, 0.0),
    ]
    for k, (di, dj, ang, rev) in enumerate(neigh):
        a[1, k] = di
        a[2, k] = dj
        a[3, k] = ang
        a[4, k] = rev
        a[0, k] = 1.0 if spec[i + di, j + dj] > 0 else 0.0
    t = int(np.sum(a[0, :]))
    return t, a


# ---------------------------------------------------------------------------
# ATFFC (IVGNMD variant — returns binary enhanced TFD)
# ---------------------------------------------------------------------------


def _tfc_ivgnmd(
    spec: np.ndarray, min_frac: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """TF clustering (``TFC`` nested in ``ATFFC.m``)."""
    s = np.asarray(spec, dtype=float)
    labeled, n_lab = label_components(s > 0)
    m, n = s.shape
    min_size = min_frac * m * n
    spec_fc = np.zeros_like(s)
    m_map = np.zeros_like(s)
    k = 0
    for i in range(1, n_lab + 1):
        mask = labeled == i
        if mask.sum() > min_size:
            k += 1
            spec_fc += mask.astype(float) * s
            m_map += mask.astype(float) / float(k)
    return spec_fc, m_map


def _tfdf(spec_fc: np.ndarray, min_frac: float = 0.001) -> np.ndarray:
    """TF distribution filling (``TFDF`` nested in ``ATFFC.m``)."""
    i_bin = (np.asarray(spec_fc, dtype=float) > 0).astype(float)
    filled = binary_fill_holes(i_bin > 0).astype(float)
    holes = filled - i_bin
    labeled, n_lab = label_components(holes > 0)
    m, n = i_bin.shape
    min_size = min_frac * m * n
    out = i_bin.copy()
    for lab in range(1, n_lab + 1):
        mask = labeled == lab
        if mask.sum() < min_size:
            out = out + mask.astype(float)
    return (out > 0).astype(float)


def _energy_ridge_binary(spec: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    """Sparsify a dense fused TF map into ridge-like support before SE."""
    s = np.asarray(spec, dtype=float)
    pos = s[s > 0]
    if pos.size == 0:
        return np.zeros_like(s)
    thr = float(np.percentile(pos, percentile))
    band = s >= thr
    # local maxima along frequency (axis 0) and time (axis 1)
    pad_f = np.pad(s, ((1, 1), (0, 0)), mode="edge")
    peak_f = (pad_f[1:-1, :] >= pad_f[:-2, :]) & (pad_f[1:-1, :] >= pad_f[2:, :])
    pad_t = np.pad(s, ((0, 0), (1, 1)), mode="edge")
    peak_t = (pad_t[:, 1:-1] >= pad_t[:, :-2]) & (pad_t[:, 1:-1] >= pad_t[:, 2:])
    ridge = band & (peak_f | peak_t)
    if int(ridge.sum()) < 50:
        ridge = band
    return ridge.astype(float)


def atffc_ivgnmd(
    signal: np.ndarray,
    samp_freq: float,
    tp: float = 6.0,
    n_windows: int = 5,
    min_frac: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Adaptive TF fusion-clustering for IVGNMD (``ATFFC.m``).

    :return: ``(I_Spec, f)`` — binary enhanced TFD and frequency axis
    """
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    th = np.zeros(n_windows + 1)
    spec_fc = None
    f = None

    for i in range(n_windows):
        win_len = samp_freq / ((i + 1) * tp)
        spec, f = stft_vgnmd(x, samp_freq, n_freq=n, win_len=win_len)
        spec = spec / (spec.max() + 1e-30)
        spec1 = spec.copy()
        if i == 0:
            th[0] = float(np.mean(spec))
        spec_thr = spec.copy()
        spec_thr[spec_thr < th[i]] = 0.0
        spec_c, _ = _tfc_ivgnmd(spec_thr, min_frac=min_frac)

        if i == 0:
            spec_fc = spec1
            spec_f = spec_c
        else:
            both = (spec_fc > 0) & (spec_c > 0)
            spec_f = np.zeros_like(spec_fc)
            spec_f[both] = np.maximum(spec_fc[both], spec_c[both])

        m1 = spec_fc[spec_fc > 0]
        m2 = spec_f[spec_f > 0]
        if m1.size == 0 or m2.size == 0:
            th[i + 1] = 0.0
        else:
            r = float(np.mean(m1) / (np.mean(m2) + 1e-30))
            if r < 0.7:
                th[i + 1] = float(np.mean(m2) - r * 1.5 * np.std(m2))
            else:
                th[i + 1] = 0.0

        spec_fc, _ = _tfc_ivgnmd(spec_f, min_frac=min_frac)

    if f is None or spec_fc is None:
        raise RuntimeError("ATFFC failed to compute a TF representation")
    # Convert dense fused energy into a ridge-like binary map, then fill small holes
    ridge = _energy_ridge_binary(spec_fc, percentile=65.0)
    i_spec = _tfdf(ridge, min_frac=min_frac)
    return i_spec, f


# ---------------------------------------------------------------------------
# Skeleton extraction (SE)
# ---------------------------------------------------------------------------


def _spur_remove(imgo: np.ndarray, h: int) -> np.ndarray:
    """Deburring nested ``spur`` in ``SE.m``."""
    imgo = _as_bool(imgo)
    m, n = imgo.shape
    spur = np.zeros_like(imgo, dtype=bool)
    ys, xs = np.nonzero(imgo)
    for i, j in zip(ys.tolist(), xs.tolist()):
        if i < 1 or j < 1 or i >= m - 1 or j >= n - 1:
            continue
        t, a = tfptd(imgo, i, j)
        if t != 1:
            continue
        ii, jj = i, j
        imgo1 = imgo.copy()
        k = 1
        sp = np.zeros_like(imgo, dtype=bool)
        while t == 1:
            v = np.flatnonzero(a[0, :] == 1)
            if v.size == 0:
                break
            v0 = int(v[0])
            imgo1[ii, jj] = False
            sp[ii, jj] = True
            ii = ii + int(a[1, v0])
            jj = jj + int(a[2, v0])
            if not (1 <= ii < m - 1 and 1 <= jj < n - 1):
                break
            t, a = tfptd(imgo1, ii, jj)
            k += 1
        if k < h:
            spur |= sp
    out = imgo & ~spur
    out = bwmorph_spur(out, n_iter=10)
    out = zhang_suen_thin(out)
    return out


def se(i_spec: np.ndarray, h: int = 60) -> np.ndarray:
    """
    Improved skeleton extraction (``SE.m``).

    :param i_spec: enhanced binary TFD
    :param h: spur length threshold
    :return: complete TF skeleton (bool / 0-1 float)
    """
    i_spec = _as_bool(i_spec)
    imgo = zhang_suen_thin(i_spec)
    imgo = _spur_remove(imgo, int(h))
    outl = bwmorph_remove(i_spec).astype(float) * 3.0
    m, n = imgo.shape
    i_out = np.zeros((m, n), dtype=float)

    ys, xs = np.nonzero(imgo)
    for i, j in zip(ys.tolist(), xs.tolist()):
        if i < 1 or j < 1 or i >= m - 1 or j >= n - 1:
            continue
        t, a = tfptd(imgo, i, j)
        if t != 1:
            continue
        ii, jj = i, j
        imgo1 = imgo.copy()
        k = 1
        rv: List[float] = []
        while t == 1:
            v = np.flatnonzero(a[0, :] == 1)
            if v.size == 0:
                break
            v0 = int(v[0])
            rv.append(float(a[4, v0]))
            imgo1[ii, jj] = False
            ii = ii + int(a[1, v0])
            jj = jj + int(a[2, v0])
            if not (1 <= ii < m - 1 and 1 <= jj < n - 1):
                break
            t, a = tfptd(imgo1, ii, jj)
            k += 1
        if k < 2 or not rv:
            continue
        ang_p = np.asarray(rv[::-1], dtype=float)
        coff = np.logspace(0, 10, k - 1)
        if coff.size != ang_p.size:
            n_use = min(coff.size, ang_p.size)
            coff = coff[:n_use]
            ang_p = ang_p[:n_use]
        w = coff / (coff.sum() + 1e-30)
        ang = np.zeros(8)
        for g in range(8):
            va = np.flatnonzero(np.isclose(ang_p, a[3, g]))
            ang[g] = float(np.sum(w[va])) if va.size else 0.0
        pm = float(np.max(ang))
        ev = int(np.argmax(ang))
        if pm <= 0:
            continue
        iii, jjj = i, j
        t1 = 0
        steps = 0
        while t1 < 2 and steps < max(m, n):
            iii = iii + int(a[1, ev])
            jjj = jjj + int(a[2, ev])
            if not (0 <= iii < m and 0 <= jjj < n):
                break
            i_out[iii, jjj] = 1.0
            if 1 <= iii < m - 1 and 1 <= jjj < n - 1:
                t1, _ = tfptd(outl, iii, jjj)
            else:
                break
            steps += 1
        ni = iii + int(a[1, ev])
        nj = jjj + int(a[2, ev])
        if 0 <= ni < m and 0 <= nj < n:
            i_out[ni, nj] = 1.0

    combined = (i_out > 0) | imgo
    return zhang_suen_thin(combined).astype(float)


# ---------------------------------------------------------------------------
# TF skeleton cutting (TFSC)
# ---------------------------------------------------------------------------


def _bwboundaries_g(tfc: np.ndarray, min_len: int = 80) -> np.ndarray:
    """Nested ``bwboundaries_g`` in ``TFSC.m``."""
    labeled, n_lab = label_components(tfc > 0)
    m, n = tfc.shape
    spec1 = np.zeros((m, n), dtype=float)
    for lab in range(1, n_lab + 1):
        mask = labeled == lab
        # MATLAB length(B{i}) traces the object contour (~2x thin-skeleton pixels).
        # Use max(pixel_count, 2*boundary) so thin ridges of length >= ~40 survive.
        boundary = bwmorph_remove(mask)
        n_pix = int(mask.sum())
        n_bound = int(boundary.sum()) if boundary.any() else n_pix
        contour_len = max(n_pix, 2 * n_bound if n_bound < n_pix else n_bound)
        if contour_len > min_len or n_pix >= min_len:
            spec1 += mask.astype(float)
    keep = spec1 > 0
    return np.asarray(tfc, dtype=float) * keep.astype(float)


def tfsc(spec_r: np.ndarray) -> np.ndarray:
    """TF skeleton cutting (``TFSC.m``) — uncrossed skeleton with markers."""
    spec_r = np.asarray(spec_r, dtype=float)
    tfc = spec_r.copy()
    m, n = spec_r.shape
    ys, xs = np.nonzero(spec_r > 0)
    for i, j in zip(ys.tolist(), xs.tolist()):
        if i < 1 or j < 1 or i >= m - 1 or j >= n - 1:
            continue
        t, a = tfptd(spec_r, i, j)
        v = np.flatnonzero(a[0, :] == 1)
        if t <= 2:
            continue
        # zero junction and short branches (up to 5 steps) on the shared TFC map
        tfc[i, j] = 0.0
        for kk in range(min(t, v.size)):
            vk = int(v[kk])
            ii = i + int(a[1, vk])
            jj = j + int(a[2, vk])
            c = 1
            while 0 <= ii < m and 0 <= jj < n and spec_r[ii, jj] > 0 and c < 5:
                tfc[ii, jj] = 0.0
                ii = ii + int(a[1, vk])
                jj = jj + int(a[2, vk])
                c += 1

    tfcb = tfc.copy()
    ys2, xs2 = np.nonzero(tfc > 0)
    for i, j in zip(ys2.tolist(), xs2.tolist()):
        if i < 1 or j < 1 or i >= m - 1 or j >= n - 1:
            continue
        t1, _ = tfptd(spec_r, i, j)
        t2, _ = tfptd(tfc, i, j)
        if t1 == 2 and t2 == 1:
            tfcb[i, j] = 2.0
    return _bwboundaries_g(tfcb)


# ---------------------------------------------------------------------------
# TF skeleton tracking (TFST)
# ---------------------------------------------------------------------------


def _angs(spec_r: np.ndarray, h: int, g: int) -> float:
    """Nested ``ANGS`` — weighted local direction at a pseudo-boundary."""
    ang_p = []
    hh, gg = int(h), int(g)
    m, n = spec_r.shape
    work = spec_r.copy()
    for _ in range(10):
        if not (1 <= hh < m - 1 and 1 <= gg < n - 1):
            break
        work[hh, gg] = 0.0
        _, a = tfptd(work, hh, gg)
        v = np.flatnonzero(a[0, :] == 1)
        if v.size == 0:
            break
        v0 = int(v[0])
        hh = hh + int(a[1, v0])
        gg = gg + int(a[2, v0])
        ang_p.append(float(a[3, v0]))
    if not ang_p:
        return 0.0
    ang_p_arr = np.asarray(ang_p, dtype=float).reshape(1, -1)
    coff = np.logspace(0, 1, ang_p_arr.size)
    ang_c = coff / coff.sum()
    sym = 1.0
    for idx in range(ang_p_arr.size):
        val = ang_p_arr[0, idx]
        if val != 180 and val != 0:
            sym = val / abs(val)
        elif val == 0:
            sym = 1.0
        else:
            if idx == 0:
                sym = 1.0
            ang_p_arr[0, idx] = sym * val
    return float(np.sum(ang_p_arr * ang_c))


def _dpf(i: float, j: float, pi: float, pj: float) -> np.ndarray:
    """Discontinuous point filling (``DPF``). Returns ``(3, L)`` path."""
    xl = float(pi - i)
    yl = float(pj - j)
    pof: List[List[float]] = [[], [], []]
    ii, jj = float(i), float(j)
    guard = 0
    while guard < 10000:
        guard += 1
        if xl == 0 and yl == 0:
            break
        if xl == 0:
            k = int(abs(yl))
            sj = yl / abs(yl) if yl != 0 else 0.0
            for _ in range(k):
                jj = jj + sj
                pof[0].append(ii)
                pof[1].append(jj)
                pof[2].append(sj * 90.0)
            break
        if yl == 0:
            k = int(abs(xl))
            si = xl / abs(xl) if xl != 0 else 0.0
            for _ in range(k):
                ii = ii + si
                pof[0].append(ii)
                pof[1].append(jj)
                pof[2].append((1.0 - si) * 90.0)
            break
        # diagonal steps
        si = xl / abs(xl)
        sj = yl / abs(yl)
        steps = int(min(abs(xl), abs(yl)))
        g0 = len(pof[0])
        for h in range(1, steps + 1):
            ii = ii + si
            jj = jj + sj
            pof[0].append(ii)
            pof[1].append(jj)
            pof[2].append(-si * 45.0 + sj * 90.0)
        h = steps
        xl = si * (abs(xl) - h)
        yl = sj * (abs(yl) - h)
        if xl == 0 and yl == 0:
            break
    if not pof[0]:
        return np.zeros((3, 0))
    return np.vstack([pof[0], pof[1], pof[2]])


def _tfss(spec_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One skeleton separation pass (``TFSS``)."""
    spec_r = np.asarray(spec_r, dtype=float)
    m1, n1 = spec_r.shape
    pad_c = np.zeros((m1, 1))
    spec = np.hstack([pad_c, spec_r, pad_c])
    pad_r = np.zeros((1, n1 + 2))
    spec = np.vstack([pad_r, spec, pad_r])
    m, n = spec.shape
    rv_list: List[List[float]] = [[], [], []]
    found = False

    def _append(ii: float, jj: float, ang: float = 0.0) -> None:
        rv_list[0].append(float(ii))
        rv_list[1].append(float(jj))
        rv_list[2].append(float(ang))

    for i0 in range(1, m - 1):
        if found:
            break
        for j0 in range(1, n - 1):
            if spec[i0, j0] != 1:
                continue
            t, a = tfptd(spec, i0, j0)
            if t != 1:
                continue

            i, j = i0, j0
            kk = 0
            steps = 0
            max_steps = m * n
            while steps < max_steps:
                steps += 1
                if not (1 <= i < m - 1 and 1 <= j < n - 1):
                    break
                t, a = tfptd(spec, i, j)
                cur_val = spec[i, j]

                if t >= 1 and cur_val == 1:
                    v = np.flatnonzero(a[0, :] == 1)
                    if v.size == 0:
                        _append(i, j, 0.0)
                        spec[i, j] = 0.0
                        break
                    v0 = int(v[0])
                    ang = float(a[3, v0])
                    _append(i, j, ang)
                    spec[i, j] = 0.0
                    i = i + int(a[1, v0])
                    j = j + int(a[2, v0])
                    continue

                if t == 0 and cur_val == 2:
                    # pseudo-boundary jump (MATLAB)
                    spec[i, j] = 0.0
                    ang_hist = np.asarray(rv_list[2][kk:], dtype=float).copy()
                    if ang_hist.size < 1:
                        break
                    sym = 1.0
                    for g_idx in range(ang_hist.size):
                        val = ang_hist[g_idx]
                        if val != 180 and val != 0:
                            sym = val / abs(val)
                        elif val == 0:
                            sym = 1.0
                        else:
                            if g_idx == 0:
                                sym = 1.0
                            ang_hist[g_idx] = sym * val
                    coff = np.logspace(0, 50, ang_hist.size)
                    w = coff / coff.sum()
                    ang_w = float(np.sum(w * ang_hist))
                    l_rad = 50
                    ang_b = []
                    for h in range(max(1, i - l_rad), min(m - 1, i + l_rad + 1)):
                        for g in range(max(1, j - l_rad), min(n - 1, j + l_rad + 1)):
                            if spec[h, g] == 2:
                                ang_b.append((_angs(spec, h, g), float(h), float(g)))
                    if not ang_b:
                        break
                    diffs = [abs(ab[0] - ang_w) for ab in ang_b]
                    evv = int(np.argmin(diffs))
                    pi_v, pj_v = ang_b[evv][1], ang_b[evv][2]
                    pof = _dpf(float(i), float(j), pi_v, pj_v)
                    for p in range(pof.shape[1]):
                        _append(pof[0, p], pof[1, p], pof[2, p])
                    i = int(round(pi_v))
                    j = int(round(pj_v))
                    _append(i, j, ang_b[evv][0])
                    kk = len(rv_list[0])
                    continue

                if t == 0 and cur_val == 1:
                    _append(i, j, 0.0)
                    spec[i, j] = 0.0
                    break

                # isolated / exhausted
                if cur_val > 0:
                    _append(i, j, 0.0)
                    spec[i, j] = 0.0
                break

            found = True
            break

    out_spec = spec[1:-1, 1:-1].copy()
    if not rv_list[0]:
        ys, xs = np.nonzero(out_spec > 0)
        if ys.size == 0:
            return np.zeros((2, 0)), out_spec
        out_spec[int(ys[0]), int(xs[0])] = 0.0
        return np.array([[float(xs[0])], [float(ys[0])]]), out_spec

    rv = np.vstack(
        [
            np.asarray(rv_list[0], dtype=float),
            np.asarray(rv_list[1], dtype=float),
            np.asarray(rv_list[2], dtype=float),
        ]
    )
    # remove pad offset and swap axes (MATLAB TFSS ending)
    rv1 = np.vstack([rv[1, :] - 1.0, rv[0, :] - 1.0])
    return rv1, out_spec


def _merge_similar_paths(
    paths: List[np.ndarray], freq_tol: float = 25.0
) -> List[np.ndarray]:
    """Merge TFST fragments that share a similar frequency band (chirp pieces)."""
    if not paths:
        return paths
    used = [False] * len(paths)
    merged: List[np.ndarray] = []
    for i, p in enumerate(paths):
        if used[i]:
            continue
        f_mean = float(np.mean(p[1, :]))
        bundle = [p]
        used[i] = True
        for j in range(i + 1, len(paths)):
            if used[j]:
                continue
            if abs(float(np.mean(paths[j][1, :])) - f_mean) <= freq_tol:
                bundle.append(paths[j])
                used[j] = True
        if len(bundle) == 1:
            merged.append(bundle[0])
            continue
        tv = np.concatenate([b[0, :] for b in bundle])
        fv = np.concatenate([b[1, :] for b in bundle])
        order = np.argsort(tv)
        merged.append(np.vstack([tv[order], fv[order]]))
    return merged


def tfst(
    tfc: np.ndarray, min_pixels: int = 100, min_path_len: int = 20
) -> Tuple[List[np.ndarray], int]:
    """
    TF skeleton tracking (``TFST.m``).

    :return: ``(R, K)`` — list of paths ``Rv`` with shape ``(2, P)`` =
             ``[time_idx; freq_idx]`` (0-based), and mode count ``K``
    """
    work = np.asarray(tfc, dtype=float).copy()
    paths: List[np.ndarray] = []
    idle = 0
    while np.count_nonzero(work > 0) > min_pixels and idle < 30:
        before = int(np.count_nonzero(work > 0))
        rv, work = _tfss(work)
        after = int(np.count_nonzero(work > 0))
        if after >= before:
            idle += 1
        else:
            idle = 0
        if rv.size == 0 or rv.shape[1] == 0:
            break
        m, n = work.shape
        tv = np.clip(np.round(rv[0, :]).astype(int), 0, n - 1)
        fv = np.clip(np.round(rv[1, :]).astype(int), 0, m - 1)
        if tv.size >= min_path_len:
            paths.append(np.vstack([tv.astype(float), fv.astype(float)]))
        if len(paths) > 50:
            break
    paths = _merge_similar_paths(paths)
    return paths, len(paths)


# ---------------------------------------------------------------------------
# MTDC on tracked ridges
# ---------------------------------------------------------------------------


def _avg_repeat_spacing(coord: np.ndarray) -> float:
    vals, counts = np.unique(coord, return_counts=True)
    repeated = vals[counts > 1]
    if repeated.size == 0:
        return np.inf
    total = 0.0
    n_rep = 0
    for v in repeated:
        pos = np.flatnonzero(coord == v)
        if pos.size < 2:
            continue
        total += float(np.sum(np.diff(pos)) / (pos.size - 1))
        n_rep += 1
    if n_rep == 0:
        return np.inf
    return total / n_rep


def mtdc_ridge(
    rv: np.ndarray, spec: np.ndarray, ad_thresh: float = 10.0
) -> Tuple[int, np.ndarray]:
    """
    Mode-type discrimination on a tracked skeleton (nested ``MTDC`` in ``IVGNMD.m``).

    ``rv`` is ``(2, P)`` with rows ``[time_idx; freq_idx]`` (0-based).

    :return: ``(type, IIF_GD)`` with ``IIF_GD`` shape ``(P, 2)``
             chirp: ``(time, freq)``; dispersive: ``(freq, time)``
    """
    rv = np.asarray(rv, dtype=float)
    if rv.ndim != 2 or rv.shape[0] < 2 or rv.shape[1] == 0:
        return 1, np.zeros((0, 2), dtype=int)

    tv = np.round(rv[0, :]).astype(int)
    fv = np.round(rv[1, :]).astype(int)
    m, n = spec.shape
    tv = np.clip(tv, 0, n - 1)
    fv = np.clip(fv, 0, m - 1)

    i_mask = np.zeros((m, n), dtype=float)
    for k in range(fv.size):
        i_mask[fv[k], tv[k]] = 1.0

    def _chirp_ridge() -> np.ndarray:
        order = np.argsort(tv)
        t_s, f_s = tv[order], fv[order]
        _, uniq = np.unique(t_s, return_index=True)
        uniq = np.sort(uniq)
        ridge = np.column_stack([t_s[uniq], f_s[uniq]])
        if ridge.shape[0] >= 4:
            return ridge.astype(int)
        indext, indexf = find_ridges_peak(i_mask)
        return np.column_stack([indext, indexf]).astype(int)

    def _disp_ridge() -> np.ndarray:
        order = np.argsort(fv)
        f_s, t_s = fv[order], tv[order]
        _, uniq = np.unique(f_s, return_index=True)
        uniq = np.sort(uniq)
        ridge = np.column_stack([f_s[uniq], t_s[uniq]])
        if ridge.shape[0] >= 4:
            return ridge.astype(int)
        indext, indexf = find_ridges_peak(i_mask.T)
        return np.column_stack([indext, indexf]).astype(int)

    # Geometric prior (normalised spans): horizontal → chirp, vertical → dispersive.
    # Stabilises MTDC on short TFST fragments where the classical CR test is noisy.
    dt_n = abs(float(tv.max() - tv.min())) / max(n - 1, 1)
    df_n = abs(float(fv.max() - fv.min())) / max(m - 1, 1)
    if df_n <= dt_n:
        return 1, _chirp_ridge()
    return 2, _disp_ridge()


# ---------------------------------------------------------------------------
# VOA (IVGNMD defaults)
# ---------------------------------------------------------------------------


def voa_ivgnmd(
    signal: np.ndarray,
    mode_type: int,
    ridge_index: np.ndarray,
    t: np.ndarray,
    f: np.ndarray,
    alpha: float = 5e-7,
    beta: float = 0.5e-5,
    tol: float = 1e-30,
    max_iter: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Variational optimisation (``VOA.m``) with IVGNMD padding conventions.

    :return: ``(Modet, Modef, EIF_GD)``
    """
    x = np.asarray(signal, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    f = np.asarray(f, dtype=float).ravel()
    n = x.size
    nf = f.size
    ridge_index = np.atleast_2d(np.asarray(ridge_index, dtype=int))
    if ridge_index.size == 0:
        return np.zeros(n), np.zeros(nf), np.zeros(n if mode_type == 1 else nf)

    if mode_type == 1:
        # columns (time, freq) — keep order along ridge (MATLAB uses as-is)
        indext = ridge_index[:, 0].astype(int)
        indexf = ridge_index[:, 1].astype(int)
        # unique sorted time support like practical ACMD use
        order = np.argsort(indext)
        indext = indext[order]
        indexf = indexf[order]
        # drop duplicate times (keep first)
        _, uniq = np.unique(indext, return_index=True)
        uniq = np.sort(uniq)
        indext = indext[uniq]
        indexf = indexf[uniq]
        indext = np.clip(indext, 0, n - 1)
        indexf = np.clip(indexf, 0, nf - 1)
        if indext.size < 4:
            return np.zeros(n), np.zeros(nf), np.zeros(n)
        sig1 = x[indext]
        iif = curve_smooth(f[indexf], beta).ravel()
        sest, if_est, _ = acmd_single(
            sig1, t[indext], iif, alpha=alpha, beta=beta, tol=tol, max_iter=max_iter
        )
        # pad like MATLAB linspace zeros
        left = np.zeros(int(indext[0]), dtype=float)
        right = np.zeros(int(n - indext[-1] - 1), dtype=float)
        mode_t = np.concatenate([left, sest, right])
        if mode_t.size != n:
            mode_t = np.zeros(n)
            mode_t[indext] = sest
        eif = np.concatenate([left, if_est, right])
        if eif.size != n:
            eif = np.zeros(n)
            eif[indext] = if_est
        mode_f_full = np.abs(np.fft.fft(mode_t))
        mode_f = mode_f_full[:nf]
        return np.real(mode_t), mode_f, eif

    if mode_type == 2:
        indexf = ridge_index[:, 0].astype(int)
        indext = ridge_index[:, 1].astype(int)
        order = np.argsort(indexf)
        indexf = indexf[order]
        indext = indext[order]
        _, uniq = np.unique(indexf, return_index=True)
        uniq = np.sort(uniq)
        indexf = indexf[uniq]
        indext = indext[uniq]
        indexf = np.clip(indexf, 0, nf - 1)
        indext = np.clip(indext, 0, n - 1)
        if indexf.size < 4:
            return np.zeros(n), np.zeros(nf), np.zeros(nf)

        fft_full = np.fft.fft(x)
        dsn = fft_full[:nf]
        dsn1 = dsn[indexf]
        ini_gd = curve_smooth(t[indext], beta).ravel()
        gd_est, des_est = gdmd_on_frequency(
            dsn1,
            f[indexf],
            ini_gd,
            alpha=alpha,
            beta=beta,
            tol=tol,
            max_iter=max_iter,
        )
        left = np.zeros(int(indexf[0]), dtype=complex)
        right_len = int(nf - indexf[-1])
        right = np.zeros(max(right_len, 0), dtype=complex)
        mode_f2 = np.concatenate([left, np.asarray(des_est, dtype=complex), right])
        if mode_f2.size > nf:
            mode_f2 = mode_f2[:nf]
        elif mode_f2.size < nf:
            mode_f2 = np.pad(mode_f2, (0, nf - mode_f2.size))
        eif = np.zeros(nf, dtype=float)
        eif_left = np.zeros(int(indexf[0]))
        eif_right = np.zeros(max(int(nf - indexf[-1]), 0))
        eif_mid = np.asarray(gd_est, dtype=float).ravel()
        eif_cat = np.concatenate([eif_left, eif_mid, eif_right])
        eif[: min(nf, eif_cat.size)] = eif_cat[:nf]

        # mirror for IFFT (MATLAB VOA)
        n_half = int(np.ceil(n / 2.0))
        mid = mode_f2[: min(mode_f2.size, n_half)]
        if mid.size < 2:
            return np.zeros(n), np.abs(mode_f2[:nf]), eif
        mirrored = np.concatenate([mid, np.conj(mid[1:n_half][::-1])])
        if mirrored.size < n:
            mirrored = np.pad(mirrored, (0, n - mirrored.size))
        elif mirrored.size > n:
            mirrored = mirrored[:n]
        mode_t = np.real(np.fft.ifft(mirrored))
        return mode_t, np.abs(mode_f2[:nf]), eif

    raise ValueError(f"unknown mode type {mode_type}")


# ---------------------------------------------------------------------------
# Demo signal (Test.m)
# ---------------------------------------------------------------------------


def make_ivgnmd_demo_signal(
    samp_freq: float = 1000.0,
    duration: float = 1.0,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Crossed GNS from the IVGNMD MATLAB ``Test.m`` demo.

    Two chirps + two dispersive modes on ``t ∈ [0, T)``.
    """
    fs = float(samp_freq)
    t = np.arange(0.0, duration, 1.0 / fs)
    nt = t.size
    nf = nt // 2 + 1
    f1 = np.arange(nf) / duration

    sig1 = np.cos(2 * np.pi * (300 * t + 15 * np.sin(2 * np.pi * t)))
    if1 = 300 + 30 * np.pi * np.cos(2 * np.pi * t)

    sig2 = np.cos(2 * np.pi * (200 * t - 15 * np.sin(2 * np.pi * t)))
    if2 = 200 - 30 * np.pi * np.cos(2 * np.pi * t)

    ds31 = 20 * np.exp(
        -1j * 2 * np.pi * (0.4 * f1 + 15 * np.sin(2 * np.pi * 0.002 * f1))
    )
    gd31 = 0.4 + 15 * 0.004 * np.pi * np.cos(2 * np.pi * 0.002 * f1)
    sig3 = spectrum_to_time(ds31, nt)

    ds41 = 20 * np.exp(
        -1j * 2 * np.pi * (0.6 * f1 - 15 * np.sin(2 * np.pi * 0.002 * f1))
    )
    gd41 = 0.6 - 15 * 0.004 * np.pi * np.cos(2 * np.pi * 0.002 * f1)
    sig4 = spectrum_to_time(ds41, nt)

    clean = sig1 + sig2 + sig3 + sig4
    if noise_std > 0:
        rng = np.random.default_rng() if rng is None else rng
        noise = rng.standard_normal(nt)
        noise = (noise - noise.mean()) / (noise.std() + 1e-30)
        noise = float(noise_std) * noise
        observed = clean + noise
    else:
        noise = np.zeros(nt)
        observed = clean.copy()

    return {
        "t": t,
        "fs": np.array([fs]),
        "signal": observed,
        "clean": clean,
        "noise": noise,
        "modes_true": np.vstack([sig1, sig2, sig3, sig4]),
        "if1": if1,
        "if2": if2,
        "f_axis": f1,
        "gd3": gd31,
        "gd4": gd41,
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class IVGNMD(object):
    """
    Improved Variational Generalized Nonlinear Mode Decomposition (IVGNMD).

    Separates crossed chirp and dispersive modes via TF-skeleton extraction
    and tracking, then reconstructs each mode with ACMD or GDMD (VOA).
    """

    def __init__(
        self,
        alpha: float = 5e-7,
        beta: float = 0.5e-5,
        tol: float = 1e-30,
        max_iter: int = 300,
        tp: float = 6.0,
        spur_len: int = 40,
        min_skeleton_pixels: int = 50,
        min_path_len: int = 40,
        n_windows: int = 5,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.tp = float(tp)
        self.spur_len = int(spur_len)
        self.min_skeleton_pixels = int(min_skeleton_pixels)
        self.min_path_len = int(min_path_len)
        self.n_windows = int(n_windows)

        self.modes_time_: Optional[np.ndarray] = None
        self.modes_freq_: Optional[np.ndarray] = None
        self.types_: Optional[np.ndarray] = None
        self.init_ridges_: Optional[List[np.ndarray]] = None
        self.features_: Optional[List[np.ndarray]] = None
        self.i_spec_: Optional[np.ndarray] = None
        self.skeleton_: Optional[np.ndarray] = None
        self.f_: Optional[np.ndarray] = None
        self.t_: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return (
            "Improved Variational Generalized Nonlinear Mode Decomposition " "(IVGNMD)"
        )

    def __call__(
        self, signal: np.ndarray, fs: float, return_all: bool = False
    ) -> Union[np.ndarray, Tuple]:
        return self.fit_transform(signal, fs, return_all=return_all)

    def fit_transform(
        self,
        signal: np.ndarray,
        fs: float,
        return_all: bool = False,
    ) -> Union[np.ndarray, Tuple]:
        """
        Decompose a generalized nonlinear signal with crossed modes.

        :param signal: 1-D real signal
        :param fs: sampling frequency (Hz)
        :param return_all: if True, also return frequency-domain modes, types,
                           initial ridges, refined IF/GD, axes, and TF maps
        :return: time-domain modes ``(K, N)``, or a rich tuple when ``return_all``
        """
        x = np.asarray(signal, dtype=float).ravel()
        if x.size < 16:
            raise ValueError("signal length must be >= 16")
        fs = float(fs)
        if fs <= 0:
            raise ValueError("fs must be positive")

        n = x.size
        t = np.arange(n, dtype=float) / fs

        # 1) enhanced binary TFD
        i_spec, f = atffc_ivgnmd(x, fs, tp=self.tp, n_windows=self.n_windows)
        # 2) skeleton extraction
        skeleton = se(i_spec, h=self.spur_len)
        # 3) cut crossings
        tfc = tfsc(skeleton)
        # 4) track independent paths
        paths, k = tfst(
            tfc,
            min_pixels=self.min_skeleton_pixels,
            min_path_len=self.min_path_len,
        )
        if k == 0:
            raise RuntimeError(
                "IVGNMD found no TF skeletons; try a higher SNR or different fs"
            )

        modes_t = np.zeros((k, n), dtype=float)
        modes_f = np.zeros((k, f.size), dtype=float)
        types = np.zeros(k, dtype=int)
        init_ridges: List[np.ndarray] = []
        features: List[np.ndarray] = []

        for i, rv in enumerate(paths):
            mtype, ridge = mtdc_ridge(rv, i_spec)
            init_ridges.append(ridge)
            mt, mf, feat = voa_ivgnmd(
                x,
                mtype,
                ridge,
                t,
                f,
                alpha=self.alpha,
                beta=self.beta,
                tol=self.tol,
                max_iter=self.max_iter,
            )
            modes_t[i] = np.real(mt)
            modes_f[i] = np.asarray(mf, dtype=float).ravel()[: f.size]
            types[i] = int(mtype)
            features.append(np.asarray(feat, dtype=float).ravel())

        self.modes_time_ = modes_t
        self.modes_freq_ = modes_f
        self.types_ = types
        self.init_ridges_ = init_ridges
        self.features_ = features
        self.i_spec_ = i_spec
        self.skeleton_ = skeleton
        self.f_ = f
        self.t_ = t

        if return_all:
            return (
                modes_t,
                modes_f,
                types,
                init_ridges,
                features,
                f,
                t,
                i_spec,
                skeleton,
            )
        return modes_t


def ivgnmd(
    signal: np.ndarray,
    fs: float,
    alpha: float = 5e-7,
    beta: float = 0.5e-5,
    tol: float = 1e-30,
    max_iter: int = 300,
    return_all: bool = False,
) -> Union[np.ndarray, Tuple]:
    """Functional wrapper around :class:`IVGNMD`."""
    return IVGNMD(alpha=alpha, beta=beta, tol=tol, max_iter=max_iter).fit_transform(
        signal, fs, return_all=return_all
    )
