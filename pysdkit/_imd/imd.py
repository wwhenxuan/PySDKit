# -*- coding: utf-8 -*-
"""
Impulsive Mode Decomposition (IMD).

Hou B., Xie M., Yan H., Wang D. Impulsive mode decomposition.
Mechanical Systems and Signal Processing, 211:111227, 2024.
https://doi.org/10.1016/j.ymssp.2024.111227
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import hilbert

from pysdkit.data._loaders import load_imd_gearbox_snippet, load_imd_input_sig


def fft_bandpass(
    signal: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
) -> np.ndarray:
    """
    Ideal FFT band-pass filter (MATLAB ``FFTbandpass``).

    Keep bins in ``[f_low, f_high]`` and the corresponding negative-frequency
    image ``[Fs-f_high, Fs-f_low]``, zero the rest, then IFFT.
    """
    x = np.asarray(signal, dtype=float).ravel()
    n = x.size
    fs = float(fs)
    lo, hi = sorted((float(f_low), float(f_high)))

    spectrum = np.fft.fft(x)
    freqs = np.arange(n, dtype=float) * (fs / n)
    keep = ((freqs >= lo) & (freqs <= hi)) | ((freqs >= fs - hi) & (freqs <= fs - lo))
    spectrum = spectrum.copy()
    spectrum[~keep] = 0.0
    y = np.fft.ifft(spectrum)
    return np.real(y)


def fre_am(data: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Single-sided amplitude spectrum (MATLAB ``FreAm``)."""
    x = np.asarray(data, dtype=float).ravel()
    n = x.size
    amp = np.abs(np.fft.fft(x)) / n
    f1 = np.arange(n // 2, dtype=float) * (float(fs) / n)
    out = amp[: f1.size].copy()
    if out.size > 1:
        out[1:] = 2.0 * out[1:]
    return f1, out


def pq_mean(seg: np.ndarray, p: float, q: float) -> float:
    """Ratio of generalised means ``M_p / M_q``."""
    x = np.asarray(seg, dtype=float).ravel()
    x = np.maximum(x, 1e-30)

    def _mean_order(a: np.ndarray, order: float) -> float:
        if abs(order) < 1e-15:
            return float(np.exp(np.mean(np.log(a))))
        return float(np.mean(a**order) ** (1.0 / order))

    return _mean_order(x, p) / _mean_order(x, q)


def cesm_pq_mean(
    signal: np.ndarray,
    n_seg: int,
    a: float = -10.0,
    p: float = 2.0,
    q: float = 1.0,
) -> float:
    """
    Cycle-embedded sparsity measure (MATLAB ``CESMpqMean``).

    Segments the absolute signal into ``n_seg`` blocks, computes the
    ``p/q`` mean ratio on each block, then aggregates with order-``a`` mean
    (``a=0`` → geometric mean).
    """
    x = np.abs(np.asarray(signal, dtype=float).ravel())
    n_seg = max(int(n_seg), 1)
    seg_len = int(np.floor(x.size / n_seg))
    if seg_len < 1:
        return 0.0

    vals = np.empty(n_seg, dtype=float)
    for i in range(n_seg):
        vals[i] = pq_mean(x[i * seg_len : (i + 1) * seg_len], p, q)
    vals = np.maximum(vals, 1e-30)

    if abs(a) < 1e-15:
        return float(np.exp(np.mean(np.log(vals))))
    return float(np.mean(vals**a) ** (1.0 / a))


def segment_sparsity(signal: np.ndarray, n_seg: int = 10) -> float:
    """
    IMD objective: GMSM / CESM on the squared envelope
    (MATLAB ``segmentSparsityM`` with ``p=2, q=1, a=-10``).
    """
    se = np.abs(hilbert(np.asarray(signal, dtype=float).ravel())) ** 2
    return cesm_pq_mean(se, n_seg=n_seg, a=-10.0, p=2.0, q=1.0)


def _minmax_check(
    minimum: np.ndarray, maximum: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Clip ``values`` into ``[minimum, maximum]`` (MATLAB ``MinMaxCheck``)."""
    return np.minimum(np.maximum(values, minimum), maximum)


def particle_swarm_optimize(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_particles: int = 45,
    max_iter: int = 20,
    maximize: bool = True,
    velocity_clamp: float = 2.0,
    cognitive: float = 2.0,
    social: float = 2.0,
    w_min: float = 0.4,
    w_max: float = 0.9,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compact PSO matching the MATLAB toolbox used by IMD.

    :return: ``(best_position, gbest_fitness_history)``
    """
    rng = np.random.default_rng() if rng is None else rng
    bounds = np.asarray(bounds, dtype=float)
    n_dim = bounds.shape[0]
    lo = bounds[:, 0].copy()
    hi = bounds[:, 1].copy()

    pos = lo + (hi - lo) * rng.random((n_particles, n_dim))
    vmax = hi * velocity_clamp
    vmin = -vmax
    vel = vmin + (vmax - vmin) * rng.random((n_particles, n_dim))

    pbest = pos.copy()
    pbest_fit = np.full(n_particles, -np.inf if maximize else np.inf)
    gbest = pos[0].copy()
    gbest_fit = -np.inf if maximize else np.inf
    history: List[float] = []

    for it in range(max_iter):
        for p in range(n_particles):
            fit = float(objective(pos[p]))
            if maximize:
                if fit > pbest_fit[p]:
                    pbest_fit[p] = fit
                    pbest[p] = pos[p].copy()
                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest = pos[p].copy()
            else:
                if fit < pbest_fit[p]:
                    pbest_fit[p] = fit
                    pbest[p] = pos[p].copy()
                if fit < gbest_fit:
                    gbest_fit = fit
                    gbest = pos[p].copy()

            w = ((max_iter - (it + 1)) * (w_max - w_min)) / max(max_iter - 1, 1) + w_min
            r1 = rng.random(n_dim)
            r2 = rng.random(n_dim)
            vel[p] = (
                w * vel[p]
                + social * r2 * (gbest - pos[p])
                + cognitive * r1 * (pbest[p] - pos[p])
            )
            vel[p] = _minmax_check(vmin, vmax, vel[p])
            pos[p] = _minmax_check(lo, hi, pos[p] + vel[p])

        history.append(gbest_fit)

    return gbest, np.asarray(history, dtype=float)


def band_split(
    previous_band: np.ndarray,
    selected_band: np.ndarray,
    min_band: float,
) -> np.ndarray:
    """Generate new candidate bands (MATLAB ``BandSplited``)."""
    prev = np.asarray(previous_band, dtype=float).ravel()
    sel = np.asarray(selected_band, dtype=float).ravel()
    parts: List[List[float]] = []
    if (sel[0] - prev[0]) > min_band:
        parts.append([prev[0], sel[0]])
    if (prev[1] - sel[1]) > min_band:
        parts.append([sel[1], prev[1]])
    if not parts:
        return np.zeros((0, 2), dtype=float)
    return np.asarray(parts, dtype=float)


class IMD(object):
    """
    Impulsive Mode Decomposition (IMD).

    IMD searches informative frequency bands with particle-swarm optimisation
    by maximising a cycle-embedded sparsity measure (GMSM / CESM) of the
    squared envelope. Each accepted band is turned into an impulsive mode by
    ideal FFT band-pass filtering.

    Hou et al., Mech. Syst. Signal Process., 211:111227, 2024.
    """

    def __init__(
        self,
        n_particles: int = 45,
        max_iter: int = 20,
        threshold: float = 1.60,
        min_band: float = 0.0,
        max_modes: int = 4,
        seg_num: int = 10,
        fs: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        :param n_particles: PSO swarm size (MATLAB ``BirdNum``)
        :param max_iter: PSO iterations (MATLAB ``maxIter``)
        :param threshold: sparsity threshold ``PreSetThres``
        :param min_band: minimum candidate bandwidth (Hz)
        :param max_modes: maximum number of impulsive modes
        :param seg_num: number of segments in the CESM objective
        :param fs: default sampling frequency
        :param seed: RNG seed for reproducible PSO
        """
        self.n_particles = int(n_particles)
        self.max_iter = int(max_iter)
        self.threshold = float(threshold)
        self.min_band = float(min_band)
        self.max_modes = int(max_modes)
        self.seg_num = int(seg_num)
        self.fs = fs
        self.seed = seed

        self.modes: Optional[np.ndarray] = None
        self.residual: Optional[np.ndarray] = None
        self.selected_bands: Optional[np.ndarray] = None
        self.residual_bands: Optional[np.ndarray] = None
        self.fitness_history: Optional[List[np.ndarray]] = None

    def __call__(
        self,
        signal: np.ndarray,
        fs: Optional[float] = None,
        return_all: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        return self.fit_transform(signal, fs=fs, return_all=return_all)

    def __str__(self) -> str:
        return "Impulsive Mode Decomposition (IMD)"

    def _pso_search_band(
        self,
        signal: np.ndarray,
        fs: float,
        band_range: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = float(band_range[0]), float(band_range[1])
        bounds = np.array([[lo, hi], [lo, hi]], dtype=float)

        def objective(para: np.ndarray) -> float:
            f1, f2 = sorted((float(para[0]), float(para[1])))
            if f2 - f1 <= 1e-12:
                return 0.0
            filtered = fft_bandpass(signal, fs, f1, f2)
            return segment_sparsity(filtered, n_seg=self.seg_num)

        best, hist = particle_swarm_optimize(
            objective,
            bounds=bounds,
            n_particles=self.n_particles,
            max_iter=self.max_iter,
            maximize=True,
            rng=rng,
        )
        return np.sort(best), hist

    def fit_transform(
        self,
        signal: np.ndarray,
        fs: Optional[float] = None,
        return_all: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Decompose a signal into impulsive modes.

        :param signal: 1-D real signal
        :param fs: sampling frequency (Hz)
        :param return_all: also return bands / residual / fitness histories
        :return: modes ``(n_modes, N)`` sorted by descending fitness, or a dict
        """
        y = np.asarray(signal, dtype=float).ravel()
        if y.size < 16:
            raise ValueError("signal length must be at least 16 samples")

        fs = self.fs if fs is None else fs
        if fs is None:
            raise ValueError("fs must be provided either in __init__ or fit_transform")
        fs = float(fs)
        if fs <= 0:
            raise ValueError("fs must be positive")

        rng = np.random.default_rng(self.seed)
        nyquist = fs / 2.0
        initial = np.array([0.0, nyquist], dtype=float)

        histories: List[np.ndarray] = []
        selected: List[List[float]] = []

        opt1, hist1 = self._pso_search_band(y, fs, initial, rng)
        histories.append(hist1)
        fit1 = float(hist1[-1])
        selected.append([float(opt1[0]), float(opt1[1]), fit1])

        if fit1 >= self.threshold and self.max_modes > 1:
            candidates = band_split(initial, opt1, self.min_band)
            mode_num = 1
            while candidates.size and mode_num < self.max_modes:
                cur = candidates[0]
                opt, hist = self._pso_search_band(y, fs, cur, rng)
                histories.append(hist)
                fit = float(hist[-1])
                if fit >= self.threshold:
                    selected.append([float(opt[0]), float(opt[1]), fit])
                    new_c = band_split(cur, opt, self.min_band)
                    if new_c.size:
                        candidates = np.vstack([candidates, new_c])
                    mode_num += 1
                candidates = candidates[1:]

        selected_arr = np.asarray(selected, dtype=float)
        # residual bands between selected informative bands
        residual_bands = self._residual_bands(selected_arr[:, :2], nyquist)

        # sort informative bands by fitness (desc), matching MATLAB sortrows(...,-3)
        order = np.argsort(-selected_arr[:, 2])
        selected_arr = selected_arr[order]

        modes = np.vstack([fft_bandpass(y, fs, row[0], row[1]) for row in selected_arr])
        residual = y - np.sum(modes, axis=0)

        self.modes = modes
        self.residual = residual
        self.selected_bands = selected_arr
        self.residual_bands = residual_bands
        self.fitness_history = histories

        if return_all:
            return {
                "modes": modes,
                "residual": residual,
                "selected_bands": selected_arr,
                "residual_bands": residual_bands,
                "fitness_history": histories,
            }
        return modes

    @staticmethod
    def _residual_bands(selected_f: np.ndarray, nyquist: float) -> np.ndarray:
        """Build residual (non-informative) frequency intervals on ``[0, Fs/2]``."""
        if selected_f.size == 0:
            return np.array([[0.0, nyquist]], dtype=float)

        temp = np.asarray(selected_f, dtype=float).copy()
        residual: List[List[float]] = []
        idx = int(np.argmin(temp[:, 0]))
        residual.append([0.0, float(temp[idx, 0])])
        next_l = float(temp[idx, 1])
        temp = np.delete(temp, idx, axis=0)
        while temp.size:
            idx = int(np.argmin(temp[:, 0]))
            residual.append([next_l, float(temp[idx, 0])])
            next_l = float(temp[idx, 1])
            temp = np.delete(temp, idx, axis=0)
        residual.append([next_l, float(nyquist)])

        out = np.asarray(residual, dtype=float)
        keep = out[:, 0] != out[:, 1]
        return out[keep] if np.any(keep) else np.zeros((0, 2), dtype=float)


def imd(
    signal: np.ndarray,
    fs: float,
    n_particles: int = 45,
    max_iter: int = 20,
    threshold: float = 1.60,
    min_band: float = 0.0,
    max_modes: int = 4,
    seg_num: int = 10,
    seed: Optional[int] = None,
    return_all: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Functional wrapper around :class:`IMD`."""
    return IMD(
        n_particles=n_particles,
        max_iter=max_iter,
        threshold=threshold,
        min_band=min_band,
        max_modes=max_modes,
        seg_num=seg_num,
        seed=seed,
    ).fit_transform(signal, fs=fs, return_all=return_all)
