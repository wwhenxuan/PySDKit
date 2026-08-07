# -*- coding: utf-8 -*-
"""
Bidimensional Multivariate Empirical Mode Decomposition (BMEMD)

Xia, Y., Zhang, B., Pei, W., and Mandic, D. P. (2019).
Bidimensional Multivariate Empirical Mode Decomposition with Applications
in Multi-Scale Image Fusion. IEEE Access, 7:114261–114270.

MATLAB reference:
https://github.com/z-bingo/Bidimensional-Multivariate-Empirical-Mode-Decomposition
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

from pysdkit._emd.memd import hamm, nth_prime


class BMEMD(object):
    """
    Bidimensional Multivariate Empirical Mode Decomposition (BMEMD)

    Projects a multi-channel image onto direction vectors on the
    ``(n-1)``-sphere, finds 2-D extrema of each projected surface, builds
    multivariate envelopes by Delaunay-based surface interpolation, and
    sifts until a MEMD-style stop criterion is met.  All channels share the
    same BIMF count, which enables multi-scale image fusion.
    """

    def __init__(
        self,
        n_dir: int = 8,
        max_imfs: int = 4,
        stop_crit: str = "stop",
        stop_vec: Optional[Sequence[float]] = None,
        stop_cnt: int = 2,
        max_iter: int = 1000,
        max_sift: int = 50,
    ) -> None:
        """
        :param n_dir: Number of projection directions (``>= 6``; MATLAB default 8)
        :param max_imfs: Maximum number of oscillatory BIMFs before the residue
        :param stop_crit: ``"stop"`` (sd / sd2 / tol) or ``"fix_h"``
        :param stop_vec: ``[sd, sd2, tol]`` when ``stop_crit="stop"``;
            default ``[0.01, 0.1, 0.01]`` (MATLAB BMEMD default)
        :param stop_cnt: Fixed sifting count when ``stop_crit="fix_h"``
        :param max_iter: Hard cap on outer BIMF extraction iterations
        :param max_sift: Hard cap on inner sifting iterations per BIMF
        """
        if not isinstance(n_dir, (int, np.integer)) or n_dir < 6:
            raise ValueError("n_dir must be an integer >= 6")
        if not isinstance(max_imfs, (int, np.integer)) or max_imfs < 1:
            raise ValueError("max_imfs must be a positive integer")
        if stop_crit not in ("stop", "fix_h"):
            raise ValueError("stop_crit must be 'stop' or 'fix_h'")

        self.n_dir = int(n_dir)
        self.max_imfs = int(max_imfs)
        self.stop_crit = stop_crit
        if stop_vec is None:
            stop_vec = (0.01, 0.1, 0.01)
        if len(stop_vec) != 3:
            raise ValueError("stop_vec must contain three elements [sd, sd2, tol]")
        self.sd, self.sd2, self.tol = map(float, stop_vec)
        self.stop_cnt = int(stop_cnt)
        self.max_iter = int(max_iter)
        self.max_sift = int(max_sift)

        self.imfs: Optional[np.ndarray] = None
        self.residue: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return "Bidimensional Multivariate Empirical Mode Decomposition (BMEMD)"

    def __call__(
        self, images: np.ndarray, max_imfs: Optional[int] = None
    ) -> np.ndarray:
        return self.fit_transform(images, max_imfs=max_imfs)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit_transform(
        self, images: np.ndarray, max_imfs: Optional[int] = None
    ) -> np.ndarray:
        """
        Decompose a multi-channel image stack.

        :param images: Array of shape ``(n_channels, H, W)`` with
            ``2 <= n_channels <= 16``
        :param max_imfs: Optional override for the number of oscillatory BIMFs
        :return: ``(K, n_channels, H, W)`` where the last slice is the residue
        """
        x = self._check_input(images)
        n_ch, height, width = x.shape
        max_imfs = self.max_imfs if max_imfs is None else int(max_imfs)

        directions = self._direction_vectors(n_ch)
        residue = x.astype(float, copy=True)
        modes: List[np.ndarray] = []

        for _ in range(max_imfs):
            if self._stop_emd(residue, directions):
                break

            mode = residue.copy()
            sift_i = 0
            n_h = 0  # consecutive OK counts for fix_h

            while sift_i < self.max_sift:
                sift_i += 1
                try:
                    env_mean, nem, amp = self._envelope_mean(mode, directions)
                except Exception:
                    env_mean = np.zeros_like(mode)
                    break

                if self.stop_crit == "stop":
                    if self._stop_sifting(env_mean, amp, nem):
                        break
                else:  # fix_h
                    # MEMD-style: count successive siftings with enough extrema
                    if nem > 9:
                        n_h += 1
                    else:
                        n_h = 0
                    if n_h >= self.stop_cnt:
                        break

                mode = mode - env_mean

            modes.append(mode)
            residue = residue - mode
            if sift_i >= self.max_sift:
                break

        modes.append(residue)
        imfs = np.stack(modes, axis=0)
        self.imfs = imfs[:-1]
        self.residue = residue
        return imfs

    def fuse(
        self,
        images: np.ndarray,
        imfs: Optional[np.ndarray] = None,
        var_window: int = 5,
    ) -> np.ndarray:
        """
        Multi-scale image fusion via local-variance weights (Xia et al.).

        For each oscillatory BIMF, channel weights are proportional to the
        local squared deviation from a moving mean (MATLAB ``local_var_img``).
        The residue is fused by intensity proportions.

        :param images: ``(n_channels, H, W)`` input stack (used if ``imfs`` is None)
        :param imfs: Optional precomputed BIMFs ``(K, n_channels, H, W)``
        :param var_window: Odd window size for local variance (default 5)
        :return: Fused grayscale image ``(H, W)``
        """
        if imfs is None:
            imfs = self.fit_transform(images)
        imfs = np.asarray(imfs, dtype=float)
        if imfs.ndim != 4:
            raise ValueError("imfs must have shape (K, n_channels, H, W)")

        k_modes, n_ch, height, width = imfs.shape
        fused = np.zeros((height, width), dtype=float)

        for q in range(k_modes):
            bimf = imfs[q]  # (C, H, W)
            if q < k_modes - 1:
                var = local_var_img(bimf, var_window)  # (C, H, W)
                denom = np.sum(var, axis=0, keepdims=True) + 1e-12
                weights = var / denom
            else:
                denom = np.sum(bimf, axis=0, keepdims=True)
                # Avoid division by zero on flat residue
                denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
                weights = bimf / denom
            fused = fused + np.sum(bimf * weights, axis=0)

        return fused

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_input(images: np.ndarray) -> np.ndarray:
        x = np.asarray(images, dtype=float)
        if x.ndim != 3:
            raise ValueError("BMEMD expects a 3-D array of shape (n_channels, H, W)")
        n_ch, height, width = x.shape
        if n_ch < 2 or n_ch > 16:
            raise ValueError("n_channels must satisfy 2 <= n_channels <= 16")
        if height < 3 or width < 3:
            raise ValueError("Each spatial dimension must be >= 3")
        return x

    def _direction_vectors(self, n_dim: int) -> np.ndarray:
        """
        Unit directions of shape ``(n_dir, n_dim)``.

        Matches MATLAB ``get_dir`` / Hammersley construction in ``bmemd.m``.
        """
        ndir = self.n_dir
        dirs = np.zeros((ndir, n_dim), dtype=float)

        if n_dim == 2:
            # Uniform samples on the circle (MATLAB 1-based index)
            for it in range(1, ndir + 1):
                dirs[it - 1, 0] = np.cos(2.0 * np.pi * it / ndir)
                dirs[it - 1, 1] = np.sin(2.0 * np.pi * it / ndir)
            return dirs

        # Hammersley low-discrepancy sequence
        if n_dim == 3:
            base = [-ndir, 2]
            seq = np.zeros((2, ndir))
            for it in range(2):
                seq[it, :] = np.asarray(hamm(ndir, base[it])).ravel()
            for it in range(ndir):
                tt = float(np.clip(2.0 * seq[0, it] - 1.0, -1.0, 1.0))
                phirad = float(seq[1, it] * 2.0 * np.pi)
                st = np.sqrt(max(1.0 - tt * tt, 0.0))
                dirs[it, 0] = st * np.cos(phirad)
                dirs[it, 1] = st * np.sin(phirad)
                dirs[it, 2] = tt
            return dirs

        # n_dim > 3 (same construction as MEMD / MATLAB bmemd.m)
        primes = nth_prime(n_dim - 1)
        base = [-ndir] + list(primes[: n_dim - 1])
        seq = np.zeros((n_dim, ndir))
        for it in range(n_dim):
            seq[it, :] = np.asarray(hamm(ndir, base[it])).ravel()

        for it in range(ndir):
            b = 2.0 * seq[:, it] - 1.0
            # atan2(sqrt(flipud(cumsum(b(end:-1:2).^2))), b(1:end-1))
            tht = np.arctan2(
                np.sqrt(np.flipud(np.cumsum(b[:0:-1] ** 2))), b[: n_dim - 1]
            )
            dir_t = np.cumprod(np.concatenate(([1.0], np.sin(tht))))
            dir_t = np.asarray(dir_t[:n_dim], dtype=float)
            dir_t[: n_dim - 1] = np.cos(tht) * dir_t[: n_dim - 1]
            dirs[it] = dir_t
        return dirs

    @staticmethod
    def _project(images: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Pixel-wise projection ``Σ_c I_c * u_c`` → shape ``(H, W)``."""
        return np.tensordot(direction, images, axes=(0, 0))

    @staticmethod
    def _regional_extrema(surface: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Boolean maps of regional maxima / minima (MATLAB ``imregional*``)."""
        maxima = maximum_filter(surface, size=3) == surface
        minima = minimum_filter(surface, size=3) == surface
        return maxima, minima

    def _stop_emd(self, residue: np.ndarray, directions: np.ndarray) -> bool:
        """Stop if any projection has fewer than 3 maxima or minima."""
        for d in directions:
            y = self._project(residue, d)
            maxima, minima = self._regional_extrema(y)
            if maxima.sum() < 3 or minima.sum() < 3:
                return True
        return False

    def _envelope_mean(
        self, mode: np.ndarray, directions: np.ndarray
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Average multivariate envelopes over all projection directions.

        :return: ``(env_mean, nem_last, amp)`` where ``amp`` is the summed
            envelope amplitude map used by the stop criterion.
        """
        n_ch, height, width = mode.shape
        env_mean = np.zeros_like(mode)
        amp = np.zeros((height, width), dtype=float)
        nem = 0

        yy_grid, xx_grid = np.mgrid[0:height, 0:width]

        for d in directions:
            y = self._project(mode, d)
            maxima, minima = self._regional_extrema(y)
            nem = int(maxima.sum() + minima.sum())

            max_r, max_c = np.nonzero(maxima)
            min_r, min_c = np.nonzero(minima)
            if max_r.size < 3 or min_r.size < 3:
                raise RuntimeError("Insufficient extrema for envelope fitting")

            env_max = np.zeros_like(mode)
            env_min = np.zeros_like(mode)
            for c in range(n_ch):
                env_max[c] = _surface_from_points(
                    max_c, max_r, mode[c, max_r, max_c], xx_grid, yy_grid
                )
                env_min[c] = _surface_from_points(
                    min_c, min_r, mode[c, min_r, min_c], xx_grid, yy_grid
                )

            amp = amp + np.sqrt(np.sum((env_max - env_min) ** 2, axis=0))
            env_mean = env_mean + 0.5 * (env_max + env_min)

        env_mean = env_mean / float(len(directions))
        return env_mean, nem, amp

    def _stop_sifting(self, env_mean: np.ndarray, amp: np.ndarray, nem: int) -> bool:
        """Return True if sifting should stop (MATLAB ``stop_sifting``)."""
        sx = np.sqrt(np.sum(env_mean**2, axis=0))
        if np.any(amp):
            sx = sx / (amp + 1e-12)
        continue_sift = (
            (np.mean(sx > self.sd) > self.tol) or np.any(sx > self.sd2)
        ) and (nem > 9)
        return not continue_sift


def _surface_from_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xx_grid: np.ndarray,
    yy_grid: np.ndarray,
) -> np.ndarray:
    """
    Interpolate scattered extrema onto the full image grid.

    Uses Clough–Tocher (Delaunay-based) interpolation as recommended in the
    BMEMD paper, with nearest-neighbour fill for exterior / degenerate regions.
    """
    points = np.column_stack([x.astype(float), y.astype(float)])
    values = z.astype(float)

    # Deduplicate coincident extrema (keep mean value)
    _, unique_idx = np.unique(points, axis=0, return_index=True)
    points = points[unique_idx]
    values = values[unique_idx]

    if points.shape[0] < 3:
        return np.full(xx_grid.shape, float(np.mean(values)))

    try:
        interp = CloughTocher2DInterpolator(points, values, fill_value=np.nan)
        surface = interp(xx_grid, yy_grid)
    except Exception:
        surface = np.full(xx_grid.shape, np.nan, dtype=float)

    if np.any(~np.isfinite(surface)):
        nearest = NearestNDInterpolator(points, values)
        nan_mask = ~np.isfinite(surface)
        surface[nan_mask] = nearest(xx_grid[nan_mask], yy_grid[nan_mask])

    return surface


def local_var_img(images: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Local squared deviation from a moving mean (MATLAB ``local_var_img``).

    :param images: ``(H, W)`` or ``(C, H, W)``
    :param window: Odd filter size
    :return: Same shape as ``images``
    """
    x = np.asarray(images, dtype=float)
    w = int(window)
    if w < 1:
        raise ValueError("window must be a positive integer")

    if x.ndim == 2:
        mean = uniform_filter(x, size=w, mode="nearest")
        return (x - mean) ** 2

    if x.ndim == 3:
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            mean = uniform_filter(x[i], size=w, mode="nearest")
            out[i] = (x[i] - mean) ** 2
        return out

    raise ValueError("images must be 2-D or 3-D")


def fuse_images(
    images: np.ndarray,
    n_dir: int = 8,
    max_imfs: int = 4,
    var_window: int = 5,
    **bmemd_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: BMEMD decomposition + variance-weighted fusion.

    :return: ``(fused_image, imfs)``
    """
    bmemd = BMEMD(n_dir=n_dir, max_imfs=max_imfs, **bmemd_kwargs)
    imfs = bmemd.fit_transform(images)
    fused = bmemd.fuse(images, imfs=imfs, var_window=var_window)
    return fused, imfs
