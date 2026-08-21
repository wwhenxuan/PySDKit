# -*- coding: utf-8 -*-
"""
Synchrosqueezing Transform (SST).

Faithful Python port of Ahrabian, Looney, Stanković and Mandic's MATLAB
``cwavelet_transform.m`` / ``sst_wavelet_linear.m`` / ``multi_sst_TF_main.m``
and the three 2015 papers:

* Ahrabian, Looney, Stanković, Mandic — *Synchrosqueezing-based
  time-frequency analysis of multivariate data*, Signal Processing 106
  (2015) 331–341.
* Ahrabian, Mandic — *A class of multivariate denoising algorithms based
  on synchrosqueezing*, IEEE TSP 63(9) (2015) 2196–2208.
* Ahrabian, Mandic — *Selective time-frequency reassignment based on
  synchrosqueezing*, IEEE SPL 22(11) (2015) 2039–2043.

The classical (univariate) SST of Daubechies, Lu and Wu is the inner
reassignment step: CWT coefficients are moved from scale to their
instantaneous frequency.  Adaptive multivariate bandwidth tiling then
turns those coefficients into AM–FM modes (inverse SST over each band)
or a joint time-frequency image.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

_WAVELET_MAP = {
    0: 0,
    "0": 0,
    "morlet": 0,
    1: 1,
    "1": 1,
    "bump": 1,
}


def parse_wavelet(wavelet: Union[int, str]) -> int:
    """
    Map a mother-wavelet selector onto MATLAB ``wavelet`` (0 Morlet, 1 bump).

    :param wavelet: int or str,
        ``0`` / ``'morlet'`` or ``1`` / ``'bump'``.
    :return: int in ``{0, 1}``.
    :raises ValueError: If the selector is unknown.
    """
    if isinstance(wavelet, str):
        key: Union[int, str] = wavelet.strip().lower()
    else:
        key = int(wavelet)
    if key not in _WAVELET_MAP:
        raise ValueError(
            "wavelet must be 0/'morlet' or 1/'bump'; got {!r}".format(wavelet)
        )
    return _WAVELET_MAP[key]


def as_channels(signal: np.ndarray) -> np.ndarray:
    """
    Orient a record as ``(n_channels, n_samples)``.

    1-D input becomes a single channel.  A 2-D array with more columns
    than rows is treated as ``(C, T)`` (MATLAB example layout and the
    PySDKit multivariate convention).  A tall ``(T, C)`` array is
    transposed, matching ``multi_sst_TF_main.m`` (``if m<n, x=x'`` is
    the opposite test on a row-wise example; after that test the code
    stores samples in rows, channels in columns).

    :param signal: array_like,
        Univariate or multivariate real record.
    :return: float array of shape ``(C, T)`` with ``T >= 3``, ``C >= 1``.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim == 2:
        n0, n1 = x.shape
        if n0 > n1:
            x = x.T
    else:
        raise ValueError(
            "SST expects a 1-D signal or a 2-D array; got shape {}".format(
                np.asarray(signal).shape
            )
        )
    if x.shape[1] < 3:
        raise ValueError("SST requires at least 3 samples")
    if x.shape[0] < 1:
        raise ValueError("SST requires at least one channel")
    return np.ascontiguousarray(x, dtype=float)


def frequency_axis_linear(n_samples: int) -> np.ndarray:
    """
    Normalised frequency grid of MATLAB ``sst_wavelet_linear`` / ``multi_sst``.

    ``freq = linspace(0, 0.5, floor(N/2)+1)`` (cycles per sample).

    :param n_samples: int,
        Length of the analysed record.
    :return: 1-D frequency axis.
    """
    n_freq = int(np.floor(n_samples / 2.0)) + 1
    return np.linspace(0.0, 0.5, n_freq)


def morlet_transfer(omega: np.ndarray, mu: float = 2.0 * np.pi) -> np.ndarray:
    """
    Frequency response of the Morlet wavelet used in ``cwavelet_transform.m``.

    :param omega: array_like,
        Radial frequencies ``a * 2π k / N``.
    :param mu: float,
        Centre frequency (MATLAB ``mu = 2*pi``).
    :return: real-valued transfer (same shape as ``omega``).
    """
    omega = np.asarray(omega, dtype=float)
    cmu = (1.0 + np.exp(-(mu**2)) - 2.0 * np.exp(-0.75 * mu**2)) ** (-0.5)
    kmu = np.exp(-0.5 * mu**2)
    return (
        cmu
        * np.pi ** (-0.25)
        * (np.exp(-0.5 * (mu - omega) ** 2) - kmu * np.exp(-0.5 * omega**2))
    )


def bump_transfer(omega: np.ndarray, mu: float = 5.0, sigma: float = 1.0) -> np.ndarray:
    """
    Frequency response of the bump wavelet (MATLAB ``mu=5``, ``si=1``).

    :param omega: array_like,
        Radial frequencies ``a * 2π k / N``.
    :param mu: float,
        Centre frequency.
    :param sigma: float,
        Half-support (MATLAB ``si``).
    :return: transfer, zero outside ``(mu-sigma, mu+sigma)``.
    """
    omega = np.asarray(omega, dtype=float)
    u = (omega - mu) / float(sigma)
    support = (np.abs(omega) > (mu - sigma)) & (np.abs(omega) < (mu + sigma))
    inside = 1.0 - u**2
    bump = np.zeros(omega.shape, dtype=float)
    safe = support & (inside > 0.0)
    bump[safe] = np.exp(-1.0 / inside[safe])
    bump[~np.isfinite(bump)] = 0.0
    return bump


def reflect_pad(signal: np.ndarray) -> np.ndarray:
    """
    Three-copy boundary used by MATLAB ``cwavelet_transform.m``.

    ``xn = [fliplr(x), x, x]``.

    :param signal: 1-D array of length ``n``.
    :return: padded array of length ``3 n``.
    """
    x = np.asarray(signal, dtype=float).ravel()
    return np.concatenate([x[::-1], x, x])


def cwavelet_transform(
    signal: np.ndarray,
    n_voices: int = 32,
    wavelet: Union[int, str] = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Continuous wavelet transform (MATLAB ``cwavelet_transform``).

    Returns the CWT, a phase-difference instantaneous frequency,
    the scale vector and the scale-derivative CWT (the last is computed
    but unused by ``multi_sst_TF_main``).

    :param signal: 1-D real samples.
    :param n_voices: int,
        Voices per octave (MATLAB ``nv``, typically 32).
    :param wavelet: int or str,
        Morlet (0) or bump (1).
    :return: ``(Wt, inst_freq, scales, dWt)`` with ``Wt`` of shape
        ``(n_scales, n_samples)``.
    """
    x = np.asarray(signal, dtype=float).ravel()
    n_samples = int(x.size)
    if n_samples < 3:
        raise ValueError("cwavelet_transform requires at least 3 samples")
    n_voices = int(n_voices)
    if n_voices < 1:
        raise ValueError("n_voices must be a positive integer")
    kind = parse_wavelet(wavelet)

    padded = reflect_pad(x)
    n_pad = int(padded.size)
    n_octaves = int(np.floor(np.log2(n_pad))) - 1
    if n_octaves < 1:
        n_octaves = 1
    n_scales = n_octaves * n_voices
    scales = 2.0 ** (np.arange(1, n_scales + 1, dtype=float) / n_voices)

    spectrum = np.fft.fft(padded)
    fft_bins = np.arange(n_pad, dtype=float)
    omega = scales[:, None] * fft_bins[None, :] * (2.0 * np.pi / n_pad)

    if kind == 0:
        transfer = morlet_transfer(omega)
    else:
        transfer = bump_transfer(omega)
    transfer = transfer * np.sqrt(scales)[:, None]
    # MATLAB uses abs(psi) for the analysis filter.
    wt_pad = np.fft.ifft(np.abs(transfer) * spectrum[None, :], axis=1)
    d_transfer = (2.0 * np.pi * (1j * omega)) * transfer
    dwt_pad = np.fft.ifft(d_transfer * spectrum[None, :], axis=1)

    angle = np.unwrap(np.angle(wt_pad), axis=1)
    inst_pad = np.diff(angle, axis=1) / (2.0 * np.pi)

    wt = wt_pad[:, n_samples : 2 * n_samples]
    dwt = dwt_pad[:, n_samples : 2 * n_samples]
    inst_freq = inst_pad[:, n_samples : 2 * n_samples]
    return wt, inst_freq, scales, dwt


def sst_wavelet_linear(
    cwt: np.ndarray,
    inst_freq: np.ndarray,
    scales: np.ndarray,
    signal: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear-frequency synchrosqueezing (MATLAB ``sst_wavelet_linear``).

    Each CWT coefficient at scale ``a`` is reassigned to the frequency
    bin ``ceil(ω/Δf + 0.5)`` of its instantaneous frequency, weighted by
    ``W / sqrt(a)``.  The map is then scaled so that
    ``var(real(sum_f Tf)) = var(x)``.

    :param cwt: complex CWT, shape ``(n_scales, n_samples)``.
    :param inst_freq: IF from :func:`cwavelet_transform`, same shape.
    :param scales: 1-D scale vector.
    :param signal: 1-D original samples (for variance normalisation).
    :return: ``(Tf, freq, Tw)`` — squeezed map, normalised frequency axis,
        and the IF written into each occupied bin.
    """
    wt = np.asarray(cwt, dtype=np.complex128)
    omega = np.asarray(inst_freq, dtype=float)
    scales = np.asarray(scales, dtype=float).ravel()
    x = np.asarray(signal, dtype=float).ravel()
    n_scales, n_samples = wt.shape
    if scales.size != n_scales:
        raise ValueError("scales length must match the CWT scale axis")
    if omega.shape != wt.shape:
        raise ValueError("inst_freq must have the same shape as the CWT")

    freq = frequency_axis_linear(n_samples)
    n_freq = int(freq.size)
    df = float(freq[1] - freq[0]) if n_freq > 1 else 1.0
    squeezed = np.zeros((n_freq, n_samples), dtype=np.complex128)
    assigned_if = np.zeros((n_freq, n_samples), dtype=float)
    weight = wt / np.sqrt(scales)[:, None]

    # Match MATLAB loop order (scale then time); later writes overwrite Tw.
    for scale_idx in range(n_scales):
        omega_row = omega[scale_idx]
        contrib = weight[scale_idx]
        bins = np.ceil(omega_row / df + 0.5).astype(np.int64) - 1
        valid = (
            (omega_row >= 0.0) & np.isfinite(omega_row) & (bins >= 0) & (bins < n_freq)
        )
        times = np.nonzero(valid)[0]
        if times.size == 0:
            continue
        dest = bins[valid]
        squeezed[dest, times] += contrib[valid]
        assigned_if[dest, times] = omega_row[valid]

    recon = np.real(np.sum(squeezed, axis=0))
    var_x = float(np.var(x))
    var_r = float(np.var(recon))
    if var_x > 0.0 and var_r > 0.0:
        squeezed = squeezed / np.sqrt(var_r / var_x)
    return squeezed, freq, assigned_if


def multivariate_bandwidth(record: np.ndarray) -> Tuple[float, float]:
    """
    Joint bandwidth and energy (MATLAB ``multi_bandwidth``).

    :param record: array_like,
        Time-major ``(n_samples, n_channels)`` as in the MATLAB file.
    :return: ``(band, power)`` — second central moment of the joint
        analytic spectrum, and ``sum |x|^2``.
    """
    x = np.asarray(record, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n_samples, n_channels = x.shape
    if n_samples < 2 or n_channels < 1:
        return 0.0, 0.0
    spectrum = np.fft.fft(x, axis=0)
    n_half = int(np.floor(n_samples / 2.0))
    if n_half < 1:
        return 0.0, float(np.sum(x * x))
    half = spectrum[:n_half, :]
    energy_density = np.sum(half * np.conj(half), axis=1).real
    total = float(np.sum(energy_density) / (2.0 * np.pi))
    power = float(np.sum(x * x))
    if total <= 0.0:
        return 0.0, power
    spec = energy_density / total
    omega = 2.0 * np.pi * np.linspace(0.0, 0.5, n_half)
    mean_freq = float(np.sum(omega * spec) / (2.0 * np.pi))
    band = float(np.sum(((omega - mean_freq) ** 2) * spec) / (2.0 * np.pi))
    return band, power


def map_scale_to_bins(edges: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Nearest-bin indices of normalised-frequency edges on ``freq``."""
    edges = np.asarray(edges, dtype=float).ravel()
    freq = np.asarray(freq, dtype=float).ravel()
    idx = np.empty(edges.size, dtype=np.int64)
    for i, value in enumerate(edges):
        idx[i] = int(np.argmin(np.abs(freq - value)))
    return idx


def invert_sst_band(squeezed: np.ndarray, start: int, stop: int) -> np.ndarray:
    """
    Time-domain mode from one frequency slice (MATLAB ``real(sum(sst))``).

    :param squeezed: complex SST, shape ``(n_freq, n_samples)``.
    :param start: int,
        Inclusive low-frequency index.
    :param stop: int,
        Exclusive high-frequency index (MATLAB ``fw(i+1):fw(i)-1``).
    :return: real 1-D waveform of length ``n_samples``.
    """
    lo = int(min(start, stop))
    hi = int(max(start, stop))
    if hi <= lo:
        return np.zeros(squeezed.shape[1], dtype=float)
    return np.real(np.sum(squeezed[lo:hi, :], axis=0))


def bandwidth_of_bands(
    sst_stack: np.ndarray,
    assigned_if: np.ndarray,
    edges: np.ndarray,
    freq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Multivariate bandwidth per tile (MATLAB ``multi_bandwidth_check``).

    :param sst_stack: complex array ``(C, F, T)``.
    :param assigned_if: unused IF map (kept for MATLAB signature parity).
    :param edges: 1-D normalised-frequency edges (will be flipped).
    :param freq: SST frequency axis.
    :return: ``(band, power, x_scale)`` with ``x_scale`` of shape
        ``(C, n_bands, T)``.
    """
    _ = assigned_if
    edges = np.asarray(edges, dtype=float).ravel()[::-1]
    bins = map_scale_to_bins(edges, freq)
    n_channels, _, n_times = sst_stack.shape
    n_bands = int(bins.size) - 1
    x_scale = np.zeros((n_channels, n_bands, n_times), dtype=float)
    for band_idx in range(n_bands):
        start = int(bins[band_idx + 1])
        stop = int(bins[band_idx])
        lo, hi = min(start, stop), max(start, stop)
        if hi > lo:
            x_scale[:, band_idx, :] = np.real(np.sum(sst_stack[:, lo:hi, :], axis=1))
    band = np.zeros(n_bands, dtype=float)
    power = np.zeros(n_bands, dtype=float)
    for band_idx in range(n_bands):
        # MATLAB passes x1' so time is rows.
        tile = np.transpose(x_scale[:, band_idx, :], (1, 0))
        band[band_idx], power[band_idx] = multivariate_bandwidth(tile)
    return band, power, x_scale


def adaptive_frequency_edges(
    sst_stack: np.ndarray,
    assigned_if: np.ndarray,
    freq: np.ndarray,
    n_levels: int = 5,
) -> np.ndarray:
    """
    Adaptive frequency tiles from multivariate bandwidth (paper §4.1).

    Direct port of the binary-mask construction in ``multi_sst_TF_main.m``.

    :param sst_stack: complex ``(C, F, T)``.
    :param assigned_if: IF written by SST, ``(C, F, T)``.
    :param freq: normalised frequency axis.
    :param n_levels: int,
        MATLAB ``V`` (typically 5).
    :return: descending edges including a final ``0``, length ``K+1``.
    """
    n_levels = int(n_levels)
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")

    max_bins = 2**n_levels
    band_f = np.zeros((n_levels, max_bins), dtype=float)
    power_f = np.zeros((n_levels, max_bins), dtype=float)
    for level in range(1, n_levels + 1):
        edges = np.linspace(0.0, 0.5, (2**level) + 1)
        band, power, _ = bandwidth_of_bands(sst_stack, assigned_if, edges, freq)
        n_band = min(band.size, 2**level)
        band_f[level - 1, :n_band] = np.sqrt(np.maximum(band[:n_band], 0.0))
        power_f[level - 1, :n_band] = power[:n_band]

    band_power = np.zeros((n_levels, max_bins), dtype=float)
    for level in range(1, n_levels + 1):
        n_half = (2**level) // 2
        for h in range(n_half):
            sl = slice(2 * h, 2 * h + 2)
            denom = float(np.sum(power_f[level - 1, sl]))
            if denom <= 0.0:
                continue
            band_power[level - 1, h] = float(
                np.sum(band_f[level - 1, sl] * power_f[level - 1, sl]) / denom
            )

    bin_mask = np.zeros((n_levels, max_bins), dtype=float)
    for level in range(2, n_levels + 1):
        n_half = (2**level) // 2
        power_row = power_f[level - 1, : 2**level]
        power_sum = float(np.sum(power_row))
        for h in range(n_half):
            parent = band_f[level - 2, h]
            child = band_power[level - 1, h]
            if parent > child * 1.0:
                bin_mask[level - 1, h] = 1.0
            else:
                bin_mask[level - 1, h] = 0.0
            if power_sum > 0.0 and power_row[-1] / power_sum > 0.4:
                if child > 0.0 and abs((parent - child) / child) * 100.0 < 9.0:
                    bin_mask[level - 1, h] = 1.0

    for level in range(2, n_levels):
        n_half = (2**level) // 2
        for h in range(n_half):
            if bin_mask[level - 1, h] == 0.0:
                bin_mask[level, 2 * h : 2 * h + 2] = 0.0

    scale_temp = np.zeros((n_levels, max_bins), dtype=float)
    scale_temp[0, 0:2] = np.array([0.25, 0.5])
    for level in range(2, n_levels + 1):
        n_half = (2**level) // 2
        for h in range(n_half):
            if bin_mask[level - 1, h] == 1.0:
                h1 = n_half - h
                scale_temp[level - 1, h] = (2 * h1 - 1) / float(2 ** (level + 1))

    flat = np.sort(scale_temp.ravel())[::-1]
    edges = flat[flat > 0.0]
    if edges.size == 0:
        edges = np.array([0.5], dtype=float)
    return np.concatenate([edges, np.array([0.0])])


def joint_instantaneous_parameters(
    sst_stack: np.ndarray,
    assigned_if: np.ndarray,
    edges: np.ndarray,
    freq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-band channel IF / amplitude and joint IF / amplitude (paper eqs. 20–23).

    :return: ``(joint_if, joint_amp, x_scale)``.
        ``joint_if`` and ``joint_amp`` have shape ``(n_bands, T)``.
        ``x_scale`` has shape ``(C, n_bands, T)``.
    """
    bins = map_scale_to_bins(edges, freq)
    n_channels, _, n_times = sst_stack.shape
    n_bands = int(bins.size) - 1
    inst_freq = np.zeros((n_channels, n_bands, n_times), dtype=float)
    inst_amp = np.zeros((n_channels, n_bands, n_times), dtype=float)
    x_scale = np.zeros((n_channels, n_bands, n_times), dtype=float)

    for channel in range(n_channels):
        sst_c = sst_stack[channel]
        if_c = assigned_if[channel]
        for band_idx in range(n_bands):
            start = int(bins[band_idx + 1])
            stop = int(bins[band_idx])
            lo, hi = min(start, stop), max(start, stop)
            if hi <= lo:
                continue
            power = np.abs(sst_c[lo:hi, :]) ** 2
            denom = np.sum(power, axis=0)
            numer = np.sum(power * if_c[lo:hi, :], axis=0)
            good = denom > 0.0
            inst_freq[channel, band_idx, good] = numer[good] / denom[good]
            inst_amp[channel, band_idx, :] = denom
            x_scale[channel, band_idx, :] = np.real(np.sum(sst_c[lo:hi, :], axis=0))

    joint_if = np.zeros((n_bands, n_times), dtype=float)
    joint_amp = np.zeros((n_bands, n_times), dtype=float)
    for band_idx in range(n_bands):
        amp = inst_amp[:, band_idx, :]
        frq = inst_freq[:, band_idx, :]
        denom = np.sum(amp, axis=0)
        numer = np.sum(amp * frq, axis=0)
        good = denom > 0.0
        joint_if[band_idx, good] = numer[good] / denom[good]
        joint_amp[band_idx, :] = np.sqrt(np.maximum(np.sum(amp, axis=0), 0.0))
    return joint_if, joint_amp, x_scale


def paint_joint_tfr(
    joint_if: np.ndarray,
    joint_amp: np.ndarray,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scatter joint amplitude onto the linear frequency grid (MATLAB Tf).

    Uses ``round(ω/Δf + 1)`` as in ``multi_sst_TF_main``, not the
    ``ceil`` rule of ``sst_wavelet_linear``.
    """
    freq = frequency_axis_linear(n_samples)
    n_freq = int(freq.size)
    df = float(freq[1] - freq[0]) if n_freq > 1 else 1.0
    tfr = np.zeros((n_freq, n_samples), dtype=float)
    n_bands = int(joint_if.shape[0])
    times = np.arange(n_samples)
    for band_idx in range(n_bands):
        omega = joint_if[band_idx]
        amp = joint_amp[band_idx]
        bins = np.round(omega / df + 1.0).astype(np.int64) - 1
        ok = (omega >= 0.0) & np.isfinite(omega) & (bins >= 0) & (bins < n_freq)
        tfr[bins[ok], times[ok]] = amp[ok]
    return tfr, freq


def n_scale_components(mask: np.ndarray) -> int:
    """Number of contiguous True runs along a 1-D boolean mask."""
    flag = np.asarray(mask, dtype=bool).ravel()
    if flag.size == 0:
        return 0
    jumps = np.diff(flag.astype(np.int8))
    return int(flag[0]) + int(np.sum(jumps == 1))


def statistical_mode_int(values: np.ndarray) -> int:
    """Mode of an integer array (smallest value on ties)."""
    values = np.asarray(values, dtype=int).ravel()
    if values.size == 0:
        return 0
    counts = np.bincount(np.clip(values, 0, None))
    return int(np.argmax(counts))


def rcm_thresholds(magnitude: np.ndarray, n_gamma: int = 40) -> np.ndarray:
    """Log-spaced thresholds from the finest-scale floor up to ``max |W|``."""
    mag = np.asarray(magnitude, dtype=float)
    peak = float(np.max(mag))
    floor = float(np.median(mag[0])) if mag.ndim == 2 else float(np.min(mag[mag > 0]))
    if not np.isfinite(floor) or floor <= 0.0:
        floor = peak * 1e-6 if peak > 0.0 else 1e-12
    if peak <= floor:
        return np.array([peak], dtype=float)
    return np.geomspace(floor, peak, int(n_gamma))


def selective_cwt_mask(
    cwt: np.ndarray,
    n_window: int = 32,
    n_gamma: int = 40,
) -> Tuple[np.ndarray, float, int]:
    """
    Hard mask of signal-like CWT coefficients (IEEE SPL 2015, §IV).

    Time is split into non-overlapping windows; ``|W|`` is averaged in
    each window; the RCM mode-count of those profiles yields a global
    number of oscillations ``K`` and a threshold ``γ``.  Coefficients
    below ``γ`` are zeroed before synchrosqueezing.

    :return: ``(masked_cwt, gamma, n_modes)``.
    """
    wt = np.asarray(cwt, dtype=np.complex128)
    n_scales, n_times = wt.shape
    n_window = max(int(n_window), 1)
    n_pad = int(np.ceil(n_times / n_window) * n_window)
    mag = np.abs(wt)
    if n_pad > n_times:
        mag = np.pad(mag, ((0, 0), (0, n_pad - n_times)))
    n_blocks = n_pad // n_window
    profiles = mag.reshape(n_scales, n_blocks, n_window).mean(axis=2)

    gammas = rcm_thresholds(mag[:, :n_times], n_gamma=n_gamma)
    k_blocks = np.zeros(n_blocks, dtype=int)
    for block in range(n_blocks):
        counts = np.array(
            [n_scale_components(profiles[:, block] > g) for g in gammas],
            dtype=int,
        )
        k_blocks[block] = statistical_mode_int(counts)
    n_modes = max(statistical_mode_int(k_blocks), 1)

    matching = []
    for block in range(n_blocks):
        for gamma in gammas:
            if n_scale_components(profiles[:, block] > gamma) == n_modes:
                matching.append(float(gamma))
    gamma = float(np.mean(matching)) if matching else float(gammas[len(gammas) // 2])
    masked = wt.copy()
    masked[np.abs(masked) < gamma] = 0.0
    return masked, gamma, n_modes


def universal_threshold(n_samples: int, sigma: float, gain: float = 0.2) -> float:
    """
    Modified universal threshold of the TSP 2015 denoising paper.

    Typical ``gain`` (paper ``c``) is ``0.1–0.3``.
    """
    n_samples = max(int(n_samples), 2)
    return float(gain) * float(sigma) * np.sqrt(2.0 * np.log(n_samples))


class SST(object):
    """
    Synchrosqueezing Transform for time-frequency analysis and modes.

    One class exposes both faces of SST:

    * :meth:`transform` — univariate squeezed CWT, or the **joint**
      multivariate TFR of Ahrabian et al., Signal Processing 2015.
    * :meth:`fit_transform` — invert SST over adaptive frequency tiles
      to recover AM–FM modes (the EMD-like reconstruction of Daubechies
      et al. / the ``real(sum(sst))`` step of the MATLAB companion).
    * :meth:`denoise` — multivariate SST thresholding (IEEE TSP 2015).
    * :meth:`selective_transform` — reassign only signal-like CWT
      coefficients (IEEE SPL 2015).

    Input layout is ``(n_samples,)`` or ``(n_channels, n_samples)``.
    Multivariate IMFs have shape ``(n_imfs, n_samples, n_channels)``;
    univariate IMFs have shape ``(n_imfs, n_samples)``.  The last IMF
    is the reconstruction residual.

    :param n_voices: int,
        Voices per octave (MATLAB ``nv``, default 32).
    :param wavelet: int or str,
        ``'bump'`` (default, MATLAB ``1``) or ``'morlet'``.
    :param n_levels: int,
        Depth of the bandwidth quadtree (MATLAB ``V``, default 5).
    :param n_window: int,
        Selective-SST averaging window (SPL 2015).
    :param denoise_gain: float,
        ``c`` in the modified universal threshold (default 0.2).
    """

    def __init__(
        self,
        n_voices: int = 32,
        wavelet: Union[int, str] = "bump",
        n_levels: int = 5,
        n_window: int = 32,
        denoise_gain: float = 0.2,
    ) -> None:
        if int(n_voices) < 1:
            raise ValueError("n_voices must be >= 1")
        if int(n_levels) < 1:
            raise ValueError("n_levels must be >= 1")
        if int(n_window) < 1:
            raise ValueError("n_window must be >= 1")
        if float(denoise_gain) <= 0.0:
            raise ValueError("denoise_gain must be positive")
        self.n_voices = int(n_voices)
        self.wavelet = parse_wavelet(wavelet)
        self.n_levels = int(n_levels)
        self.n_window = int(n_window)
        self.denoise_gain = float(denoise_gain)

        self.signal: Optional[np.ndarray] = None
        self.cwt_: Optional[np.ndarray] = None
        self.inst_freq_: Optional[np.ndarray] = None
        self.scales_: Optional[np.ndarray] = None
        self.sst_: Optional[np.ndarray] = None
        self.sst_if_: Optional[np.ndarray] = None
        self.freq_: Optional[np.ndarray] = None
        self.tfr_: Optional[np.ndarray] = None
        self.edges_: Optional[np.ndarray] = None
        self.imfs: Optional[np.ndarray] = None
        self.residue: Optional[np.ndarray] = None
        self.joint_if_: Optional[np.ndarray] = None
        self.joint_amp_: Optional[np.ndarray] = None

    def __str__(self) -> str:
        """Return the full algorithm name and abbreviation."""
        return "Synchrosqueezing Transform (SST)"

    def __call__(
        self, signal: np.ndarray, return_tfr: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Allow instances to be called like functions (modal decomposition)."""
        return self.fit_transform(signal, return_tfr=return_tfr)

    def _channel_sst(
        self, record: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run CWT + linear SST on every channel of ``(C, T)``."""
        n_channels, n_samples = record.shape
        cwt_list: List[np.ndarray] = []
        if_list: List[np.ndarray] = []
        sst_list: List[np.ndarray] = []
        tw_list: List[np.ndarray] = []
        scales = None
        freq = None
        for channel in range(n_channels):
            wt, inst, scales, _dwt = cwavelet_transform(
                record[channel], n_voices=self.n_voices, wavelet=self.wavelet
            )
            squeezed, freq, tw = sst_wavelet_linear(wt, inst, scales, record[channel])
            cwt_list.append(wt)
            if_list.append(inst)
            sst_list.append(squeezed)
            tw_list.append(tw)
        cwt = np.stack(cwt_list, axis=0)
        inst = np.stack(if_list, axis=0)
        sst = np.stack(sst_list, axis=0)
        tw = np.stack(tw_list, axis=0)
        assert scales is not None and freq is not None
        return cwt, inst, np.asarray(scales), sst, tw, np.asarray(freq)

    def _store_channel_maps(
        self,
        record: np.ndarray,
        cwt: np.ndarray,
        inst: np.ndarray,
        scales: np.ndarray,
        sst: np.ndarray,
        tw: np.ndarray,
        freq: np.ndarray,
    ) -> None:
        self.signal = record
        self.cwt_ = cwt
        self.inst_freq_ = inst
        self.scales_ = scales
        self.sst_ = sst
        self.sst_if_ = tw
        self.freq_ = freq

    def transform(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Time-frequency representation.

        Univariate records return the linear SST map (complex, then
        plotted via ``abs``).  Multivariate records return the **joint**
        amplitude image of ``multi_sst_TF_main.m``.

        :param signal: array_like,
            ``(T,)`` or ``(C, T)``.
        :return: ``(tfr, freq)`` with ``tfr`` of shape ``(n_freq, T)``.
        """
        record = as_channels(signal)
        cwt, inst, scales, sst, tw, freq = self._channel_sst(record)
        self._store_channel_maps(record, cwt, inst, scales, sst, tw, freq)
        if record.shape[0] == 1:
            self.tfr_ = sst[0]
            self.edges_ = None
            self.joint_if_ = None
            self.joint_amp_ = None
            return sst[0], freq

        edges = adaptive_frequency_edges(sst, tw, freq, n_levels=self.n_levels)
        joint_if, joint_amp, _modes = joint_instantaneous_parameters(
            sst, tw, edges, freq
        )
        tfr, freq = paint_joint_tfr(joint_if, joint_amp, record.shape[1])
        self.edges_ = edges
        self.joint_if_ = joint_if
        self.joint_amp_ = joint_amp
        self.tfr_ = tfr
        return tfr, freq

    def fit_transform(
        self,
        signal: np.ndarray,
        return_tfr: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Invert SST over adaptive bands to recover AM–FM modes.

        Each tile of the bandwidth partition is summed in frequency
        (MATLAB ``x_scale = real(sum(sst))``).  The last slice is the
        residual ``x - sum(modes)``.

        :param signal: array_like,
            Univariate ``(T,)`` or multivariate ``(C, T)``.
        :param return_tfr: bool,
            If True, also return ``(tfr, freq)``.
        :return: IMF array, or ``(imfs, tfr, freq)``.
        """
        record = as_channels(signal)
        cwt, inst, scales, sst, tw, freq = self._channel_sst(record)
        self._store_channel_maps(record, cwt, inst, scales, sst, tw, freq)
        edges = adaptive_frequency_edges(sst, tw, freq, n_levels=self.n_levels)
        joint_if, joint_amp, x_scale = joint_instantaneous_parameters(
            sst, tw, edges, freq
        )
        self.edges_ = edges
        self.joint_if_ = joint_if
        self.joint_amp_ = joint_amp

        n_channels, n_samples = record.shape
        n_bands = int(x_scale.shape[1])
        modes = np.transpose(x_scale, (1, 2, 0))  # (K, T, C)
        recon = np.sum(modes, axis=0)
        residual = record.T - recon
        imfs = np.concatenate([modes, residual[None, :, :]], axis=0)

        if n_channels == 1:
            tfr = sst[0]
            imf_out = imfs[:, :, 0]
        else:
            tfr, freq = paint_joint_tfr(joint_if, joint_amp, n_samples)
            imf_out = imfs
        self.tfr_ = tfr
        self.imfs = imf_out
        self.residue = imf_out[-1]
        if return_tfr:
            return imf_out, tfr, freq
        return imf_out

    def denoise(self, signal: np.ndarray) -> np.ndarray:
        """
        Multivariate (or univariate) SST thresholding (IEEE TSP 2015).

        Adaptive bands are formed, the joint instantaneous amplitude is
        compared to a modified universal threshold, and silenced tiles
        are dropped before inverse SST.

        :param signal: array_like,
            ``(T,)`` or ``(C, T)``.
        :return: denoised array, same layout as the oriented ``(C, T)``
            record (1-D if the input was univariate).
        """
        record = as_channels(signal)
        cwt, inst, scales, sst, tw, freq = self._channel_sst(record)
        self._store_channel_maps(record, cwt, inst, scales, sst, tw, freq)
        edges = adaptive_frequency_edges(sst, tw, freq, n_levels=self.n_levels)
        _joint_if, joint_amp, x_scale = joint_instantaneous_parameters(
            sst, tw, edges, freq
        )
        self.edges_ = edges
        self.joint_amp_ = joint_amp

        finest = np.abs(cwt[:, 0, :]).ravel()
        sigma = float(np.median(finest) / 0.6745) if finest.size else 0.0
        thresh = universal_threshold(record.shape[1], sigma, gain=self.denoise_gain)
        keep = joint_amp >= thresh
        cleaned = x_scale * keep[None, :, :]
        denoised = np.sum(cleaned, axis=1)
        if record.shape[0] == 1:
            return denoised[0]
        return denoised

    def selective_transform(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selective synchrosqueezing (IEEE SPL 2015).

        CWT coefficients that fail the windowed RCM threshold are zeroed
        *before* reassignment, so noise is not squeezed onto the TFR.

        :param signal: 1-D or single-channel array.  Multivariate input
            is squeezed channel-wise and the mean ``|Tf|`` is returned.
        :return: ``(tfr, freq)``.
        """
        record = as_channels(signal)
        n_channels, n_samples = record.shape
        maps = []
        freq = frequency_axis_linear(n_samples)
        cwt_list = []
        inst_list = []
        sst_list = []
        tw_list = []
        scales = None
        for channel in range(n_channels):
            wt, inst, scales, _dwt = cwavelet_transform(
                record[channel], n_voices=self.n_voices, wavelet=self.wavelet
            )
            masked, _gamma, _k = selective_cwt_mask(wt, n_window=self.n_window)
            squeezed, freq, tw = sst_wavelet_linear(
                masked, inst, scales, record[channel]
            )
            maps.append(squeezed)
            cwt_list.append(wt)
            inst_list.append(inst)
            sst_list.append(squeezed)
            tw_list.append(tw)
        self.signal = record
        self.cwt_ = np.stack(cwt_list, axis=0)
        self.inst_freq_ = np.stack(inst_list, axis=0)
        self.scales_ = np.asarray(scales)
        self.sst_ = np.stack(sst_list, axis=0)
        self.sst_if_ = np.stack(tw_list, axis=0)
        self.freq_ = freq
        if n_channels == 1:
            tfr = maps[0]
        else:
            tfr = np.mean(np.abs(np.stack(maps, axis=0)), axis=0)
        self.tfr_ = tfr
        return tfr, freq

    def instantaneous_frequency(self) -> np.ndarray:
        """
        Return joint IF trajectories ``(n_bands, T)`` after ``transform``
        / ``fit_transform`` on multivariate data, or the SST-bin IF
        ``(F, T)`` for a univariate map.
        """
        if self.joint_if_ is not None:
            return np.asarray(self.joint_if_, dtype=float)
        if self.sst_if_ is None:
            raise ValueError("Call transform or fit_transform first.")
        if self.sst_if_.ndim == 3:
            return self.sst_if_[0]
        return np.asarray(self.sst_if_, dtype=float)


def sst(signal: np.ndarray, **kwargs) -> np.ndarray:
    """Functional modal-decomposition interface (``SST(...)(signal)``)."""
    return SST(**kwargs).fit_transform(signal)
