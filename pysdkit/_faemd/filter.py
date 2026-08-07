# -*- coding: utf-8 -*-
"""
Shared order-statistics filtering and adaptive window sizing for FA-MVEMD.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import Delaunay
from scipy.stats import mode as scipy_mode


def immse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two arrays (MATLAB ``immse``)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("immse inputs must share the same shape")
    return float(np.mean((a - b) ** 2))


def _force_odd_windows(windows: np.ndarray, type_index: int) -> np.ndarray:
    """Force odd integers and clamp the selected type to at least 3."""
    windows = 2.0 * np.floor(np.asarray(windows, dtype=float) / 2.0) + 1.0
    if windows[type_index] < 3:
        windows[type_index] = 3.0
    return windows


def _seven_windows_from_spacings(
    edge_max: np.ndarray, edge_min: np.ndarray, type_index: int
) -> np.ndarray:
    """Bhuiyan / FA-MVEMD d1..d7 window candidates from extrema spacings."""
    edge_max = np.asarray(edge_max, dtype=float).ravel()
    edge_min = np.asarray(edge_min, dtype=float).ravel()
    if edge_max.size == 0 or edge_min.size == 0:
        return np.full(7, 3.0)

    d1 = min(np.min(edge_max), np.min(edge_min))
    d2 = max(np.min(edge_max), np.min(edge_min))
    d3 = min(np.max(edge_max), np.max(edge_min))
    d4 = max(np.max(edge_max), np.max(edge_min))
    d5 = (d1 + d2 + d3 + d4) / 4.0
    concat = np.concatenate([edge_min, edge_max])
    d6 = float(np.median(concat))
    d7 = float(scipy_mode(concat, keepdims=False).mode)
    windows = np.array([d1, d2, d3, d4, d5, d6, d7], dtype=float)
    return _force_odd_windows(windows, type_index)


def filter_size1D(
    imax: np.ndarray, imin: np.ndarray, window_type: int = 0
) -> np.ndarray:
    """
    Adaptive 1-D window sizes from extrema index spacings.

    :param imax: Indices of maxima
    :param imin: Indices of minima
    :param window_type: Selected type in ``0..6`` (MATLAB ``type`` 1..7)
    :return: Length-7 array of odd window sizes
    """
    imax = np.asarray(imax, dtype=float).ravel()
    imin = np.asarray(imin, dtype=float).ravel()
    if imax.size < 2 or imin.size < 2:
        return np.full(7, 3.0)

    edge_max = np.diff(np.sort(imax))
    edge_min = np.diff(np.sort(imin))
    # Drop zero spacings (duplicate indices)
    edge_max = edge_max[edge_max > 0]
    edge_min = edge_min[edge_min > 0]
    if edge_max.size == 0 or edge_min.size == 0:
        return np.full(7, 3.0)
    return _seven_windows_from_spacings(edge_max, edge_min, window_type)


def ord_filt1(signal: np.ndarray, order: str, window_size: int) -> np.ndarray:
    """1-D rank-order filter with symmetric (reflected) padding."""
    signal = np.asarray(signal, dtype=float)
    shape = signal.shape
    x = np.squeeze(signal).astype(float)
    if x.ndim != 1:
        raise ValueError("ord_filt1 expects a 1-D signal")

    w = int(window_size)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    r = (w - 1) // 2

    padded = np.concatenate([np.flip(x[:r]), x, np.flip(x[-r:])])
    out = np.empty_like(padded)
    if order == "max":
        for m in range(r, len(padded) - r):
            out[m] = np.max(padded[m - r : m + r + 1])
    elif order == "min":
        for m in range(r, len(padded) - r):
            out[m] = np.min(padded[m - r : m + r + 1])
    else:
        raise ValueError("order must be 'max' or 'min'")

    return np.reshape(out[r:-r], shape)


def pad_smooth_1d(env_max: np.ndarray, env_min: np.ndarray, w_sz: int) -> np.ndarray:
    """MATLAB ``Pad_smooth`` for 1-D envelopes → mean envelope."""
    w_sz = int(w_sz)
    h = w_sz // 2
    max_p = np.pad(np.asarray(env_max, dtype=float), (h, h), mode="symmetric")
    min_p = np.pad(np.asarray(env_min, dtype=float), (h, h), mode="symmetric")
    # Moving average with endpoint discard (same length as original)
    kernel = np.ones(w_sz) / w_sz
    max_s = np.convolve(max_p, kernel, mode="valid")
    min_s = np.convolve(min_p, kernel, mode="valid")
    return 0.5 * (max_s + min_s)


def mean_envelope_1d(signal: np.ndarray, w_sz: int) -> np.ndarray:
    """OSF max/min envelopes + pad-smooth mean for a 1-D channel."""
    env_max = ord_filt1(signal, "max", w_sz)
    env_min = ord_filt1(signal, "min", w_sz)
    return pad_smooth_1d(env_max, env_min, w_sz)


def identify_max_min_2d(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    2-D extrema maps via 8-neighbour ring comparison (MATLAB ``Identify_max_min``).
    """
    signal = np.asarray(signal, dtype=float)
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    neigh_max = ndimage.maximum_filter(signal, footprint=mask, mode="nearest")
    neigh_min = ndimage.minimum_filter(signal, footprint=mask, mode="nearest")
    maxima = signal >= neigh_max
    minima = signal <= neigh_min
    return maxima, minima


def _nearest_edge_lengths_2d(points_yx: np.ndarray) -> Optional[np.ndarray]:
    """Per-vertex shortest Delaunay edge length; ``None`` if triangulation fails."""
    if points_yx.shape[0] < 3:
        return None
    # Delaunay expects (x, y)
    coords = np.column_stack([points_yx[:, 1], points_yx[:, 0]])
    try:
        tri = Delaunay(coords)
    except Exception:
        return None

    simplices = tri.simplices
    nearest = np.zeros(points_yx.shape[0], dtype=float)
    for simplex in simplices:
        pts = coords[simplex]
        e01 = np.linalg.norm(pts[1] - pts[0])
        e02 = np.linalg.norm(pts[2] - pts[0])
        e12 = np.linalg.norm(pts[2] - pts[1])
        # Vertex → min of its two incident edges (MATLAB EMD2D2V)
        vertex_edges = np.array(
            [
                min(e01, e02),
                min(e01, e12),
                min(e02, e12),
            ]
        )
        for local_i, global_i in enumerate(simplex):
            e = vertex_edges[local_i]
            if nearest[global_i] == 0.0 or e < nearest[global_i]:
                nearest[global_i] = e
    return nearest


def filter_size_2d(
    maxima_map: np.ndarray, minima_map: np.ndarray, window_type: int = 5
) -> np.ndarray:
    """Adaptive windows from 2-D Delaunay nearest-neighbour distances."""
    max_yx = np.column_stack(np.nonzero(maxima_map))
    min_yx = np.column_stack(np.nonzero(minima_map))
    max_nearest = _nearest_edge_lengths_2d(max_yx)
    min_nearest = _nearest_edge_lengths_2d(min_yx)
    if max_nearest is None or min_nearest is None:
        return np.zeros(7)
    return _seven_windows_from_spacings(max_nearest, min_nearest, window_type)


def ord_filt2(signal: np.ndarray, order: str, window_size: int) -> np.ndarray:
    """2-D order-statistics filter (MATLAB ``ordfilt2`` with symmetric padding)."""
    w = int(window_size)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    signal = np.asarray(signal, dtype=float)
    if order == "max":
        return ndimage.maximum_filter(signal, size=w, mode="mirror")
    if order == "min":
        return ndimage.minimum_filter(signal, size=w, mode="mirror")
    raise ValueError("order must be 'max' or 'min'")


def pad_smooth_2d(env_max: np.ndarray, env_min: np.ndarray, w_sz: int) -> np.ndarray:
    """Separable moving-average smoothing of 2-D envelopes (replicate pad)."""
    w_sz = int(w_sz)
    h = w_sz // 2
    max_p = np.pad(env_max, ((h, h), (h, h)), mode="edge")
    min_p = np.pad(env_min, ((h, h), (h, h)), mode="edge")
    # uniform_filter with mode nearest on the padded array, then crop
    max_s = ndimage.uniform_filter(max_p, size=w_sz, mode="nearest")[h:-h, h:-h]
    min_s = ndimage.uniform_filter(min_p, size=w_sz, mode="nearest")[h:-h, h:-h]
    return 0.5 * (max_s + min_s)


def mean_envelope_2d(signal: np.ndarray, w_sz: int) -> np.ndarray:
    env_max = ord_filt2(signal, "max", w_sz)
    env_min = ord_filt2(signal, "min", w_sz)
    return pad_smooth_2d(env_max, env_min, w_sz)


def identify_max_min_3d(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    3-D extrema via 26-neighbour comparison (Robust=1 style).
    """
    signal = np.asarray(signal, dtype=float)
    footprint = np.ones((3, 3, 3), dtype=bool)
    footprint[1, 1, 1] = False
    neigh_max = ndimage.maximum_filter(signal, footprint=footprint, mode="nearest")
    neigh_min = ndimage.minimum_filter(signal, footprint=footprint, mode="nearest")
    return signal >= neigh_max, signal <= neigh_min


def _nearest_edge_lengths_3d(points: np.ndarray) -> Optional[np.ndarray]:
    """Per-vertex shortest tetrahedral edge; ``None`` if Delaunay fails."""
    if points.shape[0] < 4:
        return None
    try:
        tri = Delaunay(points.astype(float))
    except Exception:
        return None

    nearest = np.zeros(points.shape[0], dtype=float)
    for simplex in tri.simplices:
        pts = points[simplex]
        # 6 edges of a tetrahedron
        edges = []
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for a, b in pairs:
            edges.append(((a, b), np.linalg.norm(pts[a] - pts[b])))
        # For each vertex, min over incident edges
        for v in range(4):
            incident = [length for (i, j), length in edges if i == v or j == v]
            e = min(incident)
            g = simplex[v]
            if nearest[g] == 0.0 or e < nearest[g]:
                nearest[g] = e
    return nearest


def filter_size_3d(
    maxima_map: np.ndarray, minima_map: np.ndarray, window_type: int = 5
) -> np.ndarray:
    """Adaptive windows from 3-D Delaunay nearest-neighbour distances."""
    max_pts = np.column_stack(np.nonzero(maxima_map)).astype(float)
    min_pts = np.column_stack(np.nonzero(minima_map)).astype(float)
    max_nearest = _nearest_edge_lengths_3d(max_pts)
    min_nearest = _nearest_edge_lengths_3d(min_pts)
    if max_nearest is None or min_nearest is None:
        return np.zeros(7)
    return _seven_windows_from_spacings(max_nearest, min_nearest, window_type)


def ord_filt3_separable(signal: np.ndarray, order: str, window_size: int) -> np.ndarray:
    """Separable 3-D OSF: apply 1-D rank filter along each axis."""
    w = int(window_size)
    if w < 3:
        w = 3
    if w % 2 == 0:
        w += 1
    out = np.asarray(signal, dtype=float).copy()
    filt = ndimage.maximum_filter if order == "max" else ndimage.minimum_filter
    if order not in ("max", "min"):
        raise ValueError("order must be 'max' or 'min'")
    for axis in range(3):
        size = [1, 1, 1]
        size[axis] = w
        out = filt(out, size=size, mode="mirror")
    return out


def pad_smooth_3d(env_max: np.ndarray, env_min: np.ndarray, w_sz: int) -> np.ndarray:
    """Separable moving-average smoothing of 3-D envelopes."""
    w_sz = int(w_sz)
    h = w_sz // 2
    max_p = np.pad(env_max, h, mode="edge")
    min_p = np.pad(env_min, h, mode="edge")
    max_s = ndimage.uniform_filter(max_p, size=w_sz, mode="nearest")
    min_s = ndimage.uniform_filter(min_p, size=w_sz, mode="nearest")
    sl = tuple(slice(h, -h) for _ in range(3))
    return 0.5 * (max_s[sl] + min_s[sl])


def mean_envelope_3d(signal: np.ndarray, w_sz: int) -> np.ndarray:
    env_max = ord_filt3_separable(signal, "max", w_sz)
    env_min = ord_filt3_separable(signal, "min", w_sz)
    return pad_smooth_3d(env_max, env_min, w_sz)


# Backward-compatible alias used by older imports
def filter_size1D_legacy(imax: np.ndarray, imin: np.ndarray) -> int:
    """Legacy single-window helper (returns type-4 / d4 style size)."""
    windows = filter_size1D(imax, imin, window_type=3)
    return int(windows[3])
