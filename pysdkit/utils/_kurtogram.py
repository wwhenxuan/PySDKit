# -*- coding: utf-8 -*-
"""
Fast Kurtogram (Antoni, MSSP 2007).

Faithful Python port of MATLAB ``Fast_kurtogram.m`` (Pack Kurtogram V4).
A 1/2-binary + 1/3-ternary analytic filter-bank tree samples the
``(f_c, Δf)`` plane in ``O(N log N)`` and scores the kurtosis of each
complex envelope.  The resulting 2-D map (the kurtogram) locates the
band where impulsive / transient faults are most detectable.

The MATLAB driver always opens a figure and then blocks on ``input()``
to optionally filter.  This module is non-interactive: :func:`fast_kurtogram`
returns the map and the peak ``(K_max, level, f_c, Δf)``; an explicit
call to :func:`find_wav_kurt` extracts the complex envelope.

Antoni, J. Fast computation of the kurtogram for the detection of
transient faults. Mechanical Systems and Signal Processing 21 (2007)
108–124.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import solve_toeplitz
from scipy.signal import firwin, lfilter

_EPS = float(np.finfo(float).eps)


def analytic_filters() -> (
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """
    Analytic FIR pair ``(h, g)`` and ternary trio ``(h1, h2, h3)``.

    Direct port of MATLAB ``Fast_kurtogram.m`` lines 35–43
    (``fir1`` + complex modulation).
    """
    order = 16
    cutoff = 0.4
    taps = np.arange(order + 1, dtype=float)
    lowpass = firwin(order + 1, cutoff, window="hamming", pass_zero=True)
    filt_h = lowpass * np.exp(2j * np.pi * taps * 0.125)
    n_seq = np.arange(2, order + 2)
    # MATLAB: g = h(1+mod(1-n,N)).*(-1).^(1-n) with N=16, length 16
    idx = np.mod(1 - n_seq, order)
    signs = np.where(n_seq % 2 == 0, -1.0, 1.0)
    filt_g = filt_h[idx] * signs

    order_t = int(np.fix(1.5 * order))
    taps_t = np.arange(order_t + 1, dtype=float)
    h1_lp = firwin(order_t + 1, (2.0 / 3.0) * cutoff, window="hamming", pass_zero=True)
    filt_h1 = h1_lp * np.exp(2j * np.pi * taps_t * 0.25 / 3.0)
    filt_h2 = filt_h1 * np.exp(2j * np.pi * taps_t / 6.0)
    filt_h3 = filt_h1 * np.exp(2j * np.pi * taps_t / 3.0)
    return filt_h, filt_g, filt_h1, filt_h2, filt_h3


def kurt(x: np.ndarray, opt: str = "kurt2") -> float:
    """
    Envelope kurtosis of ``x`` (MATLAB ``kurt``).

    ``kurt2`` is the classical kurtosis of the (complex) envelope:
    ``E[|x|^4]/E[|x|^2]^2`` minus 3 (real) or 2 (complex).
    ``kurt1`` is the L1 variant used as an option in the MATLAB file.

    :param x: array_like,
        Real or complex 1-D samples.
    :param opt: str,
        ``'kurt2'`` (default) or ``'kurt1'``.
    :return: float kurtosis (0 if the record is degenerate).
    """
    samples = np.asarray(x).ravel()
    if samples.size == 0 or np.all(samples == 0):
        return 0.0
    samples = samples - np.mean(samples)
    is_real = np.max(np.abs(np.imag(samples))) <= 1e-15 * (
        1.0 + np.max(np.abs(samples))
    )

    if opt == "kurt2":
        energy = float(np.mean(np.abs(samples) ** 2))
        if energy < _EPS:
            return 0.0
        value = float(np.mean(np.abs(samples) ** 4) / energy**2)
        return value - 3.0 if is_real else value - 2.0
    if opt == "kurt1":
        energy = float(np.mean(np.abs(samples)))
        if energy < _EPS:
            return 0.0
        value = float(np.mean(np.abs(samples) ** 2) / energy**2)
        return value - 1.57 if is_real else value - 1.27
    raise ValueError("opt must be 'kurt2' or 'kurt1'; got {!r}".format(opt))


def max_ij(matrix: np.ndarray) -> Tuple[int, int, float]:
    """
    Row/column of the maximum (MATLAB ``max_IJ``, 0-based).

    MATLAB maximises each column first, then the row of column-maxima,
    so ties are broken by the left-most column, then the top-most row
    in that column.

    :return: ``(row, col, value)``.
    """
    arr = np.asarray(matrix)
    if arr.size == 0:
        raise ValueError("max_ij() of an empty array")
    col_max = np.max(arr, axis=0)
    col = int(np.argmax(col_max))
    row = int(np.argmax(arr[:, col]))
    return row, col, float(arr[row, col])


def binary(index: int, bits: int) -> np.ndarray:
    """
    Binary expansion of ``index`` with ``bits`` digits (MATLAB ``binary``).

    ``index = a[0]*2^(k-1) + ... + a[k-1]``.
    """
    if bits < 0:
        raise ValueError("bits must be non-negative")
    if index >= 2**bits or index < 0:
        raise ValueError("i must be such that 0 <= i < 2^k")
    out = np.zeros(bits, dtype=int)
    temp = int(index)
    for pos, power in enumerate(range(bits - 1, -1, -1)):
        out[pos] = temp // (2**power)
        temp -= int(out[pos]) * (2**power)
    return out


def raylinv(
    prob: Union[float, np.ndarray], scale: Union[float, np.ndarray] = 1.0
) -> np.ndarray:
    """Inverse Rayleigh CDF (MATLAB ``raylinv``), used as an envelope threshold."""
    p = np.asarray(prob, dtype=float)
    b = np.asarray(scale, dtype=float)
    p, b = np.broadcast_arrays(p, b)
    out = np.zeros(p.shape, dtype=float)
    bad = (b <= 0.0) | (p < 0.0) | (p > 1.0)
    out[bad] = np.nan
    out[p == 1.0] = np.inf
    ok = (b > 0.0) & (p > 0.0) & (p < 1.0)
    out[ok] = np.sqrt((-2.0 * b[ok] ** 2) * np.log(1.0 - p[ok]))
    return out if out.ndim else float(out)


def dbfb(
    x: np.ndarray, filt_h: np.ndarray, filt_g: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-band analysis filter bank (MATLAB ``DBFB``).

    Decimation keeps MATLAB even samples ``a(2:2:N)`` → ``a[1::2]``.
    """
    samples = np.asarray(x).ravel()
    n_samples = int(samples.size)
    approx = lfilter(filt_h, [1.0], samples)[1:n_samples:2]
    detail = lfilter(filt_g, [1.0], samples)[1:n_samples:2]
    n_dec = int(approx.size)
    if n_dec:
        detail = detail * ((-1.0) ** np.arange(1, n_dec + 1))
    return np.asarray(approx).ravel(), np.asarray(detail).ravel()


def tbfb(
    x: np.ndarray,
    filt_h1: np.ndarray,
    filt_h2: np.ndarray,
    filt_h3: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-band analysis filter bank (MATLAB ``TBFB``), stride 3."""
    samples = np.asarray(x).ravel()
    n_samples = int(samples.size)
    a1 = lfilter(filt_h1, [1.0], samples)[2:n_samples:3]
    a2 = lfilter(filt_h2, [1.0], samples)[2:n_samples:3]
    a3 = lfilter(filt_h3, [1.0], samples)[2:n_samples:3]
    return np.asarray(a1).ravel(), np.asarray(a2).ravel(), np.asarray(a3).ravel()


def _kurt_trim(x: np.ndarray, n_drop: int, opt: str) -> float:
    """Kurtosis after dropping the FIR transient ``x[n_drop-1:]`` (1-based ``Lh``)."""
    start = max(int(n_drop) - 1, 0)
    if start >= np.asarray(x).size:
        return 0.0
    return kurt(np.asarray(x).ravel()[start:], opt=opt)


def _k_wpq_local(
    x: np.ndarray,
    filt_h: np.ndarray,
    filt_g: np.ndarray,
    filt_h1: np.ndarray,
    filt_h2: np.ndarray,
    filt_h3: np.ndarray,
    nlevel: int,
    opt: str,
    level: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """One node of the binary–ternary packet tree (MATLAB ``K_wpQ_local``)."""
    approx, detail = dbfb(x, filt_h, filt_g)
    len_h = int(np.asarray(filt_h).size)
    len_g = int(np.asarray(filt_g).size)

    k1 = _kurt_trim(approx, len_h, opt)
    k2 = _kurt_trim(detail, len_g, opt)

    if level > 2:
        a1, a2, a3 = tbfb(approx, filt_h1, filt_h2, filt_h3)
        d1, d2, d3 = tbfb(detail, filt_h1, filt_h2, filt_h3)
        ka1 = _kurt_trim(a1, len_h, opt)
        ka2 = _kurt_trim(a2, len_h, opt)
        ka3 = _kurt_trim(a3, len_h, opt)
        kd1 = _kurt_trim(d1, len_h, opt)
        kd2 = _kurt_trim(d2, len_h, opt)
        kd3 = _kurt_trim(d3, len_h, opt)
    else:
        ka1 = ka2 = ka3 = kd1 = kd2 = kd3 = 0.0

    if level == 1:
        k_mat = np.concatenate([np.full(3, k1), np.full(3, k2)]).reshape(1, -1)
        kq = np.array([ka1, ka2, ka3, kd1, kd2, kd3], dtype=float).reshape(1, -1)
    else:
        ka, kaq = _k_wpq_local(
            approx, filt_h, filt_g, filt_h1, filt_h2, filt_h3, nlevel, opt, level - 1
        )
        kd, kdq = _k_wpq_local(
            detail, filt_h, filt_g, filt_h1, filt_h2, filt_h3, nlevel, opt, level - 1
        )
        row = np.concatenate([np.full(ka.shape[1], k1), np.full(kd.shape[1], k2)])
        k_mat = np.vstack([row, np.hstack([ka, kd])])
        width = int(round(kaq.shape[1] / 3.0))
        q_row = np.concatenate(
            [
                np.full(width, ka1),
                np.full(width, ka2),
                np.full(width, ka3),
                np.full(width, kd1),
                np.full(width, kd2),
                np.full(width, kd3),
            ]
        )
        kq = np.vstack([q_row, np.hstack([kaq, kdq])])

    if level == nlevel:
        k_raw = kurt(x, opt=opt)
        k_mat = np.vstack([np.full(k_mat.shape[1], k_raw), k_mat])
        t1, t2, t3 = tbfb(x, filt_h1, filt_h2, filt_h3)
        q_width = int(round(kq.shape[1] / 3.0))
        q_top = np.concatenate(
            [
                np.full(q_width, _kurt_trim(t1, len_h, opt)),
                np.full(q_width, _kurt_trim(t2, len_h, opt)),
                np.full(q_width, _kurt_trim(t3, len_h, opt)),
            ]
        )
        if kq.shape[0] > 2:
            kq = np.vstack([q_top, kq[: kq.shape[0] - 2, :]])
        else:
            kq = q_top.reshape(1, -1)
    return k_mat, kq


def k_wpq(
    x: np.ndarray,
    filt_h: np.ndarray,
    filt_g: np.ndarray,
    filt_h1: np.ndarray,
    filt_h2: np.ndarray,
    filt_h3: np.ndarray,
    nlevel: int,
    opt: str = "kurt2",
) -> np.ndarray:
    """
    Kurtosis of the full binary–ternary packet tree (MATLAB ``K_wpQ``).

    Output shape is ``(2*nlevel, 3*2**nlevel)``.
    """
    samples = np.asarray(x).ravel()
    nlevel = int(nlevel)
    max_level = int(np.floor(np.log2(max(samples.size, 1))))
    if nlevel >= max_level:
        raise ValueError("nlevel must be smaller than log2(length(x))")
    kd, kq = _k_wpq_local(
        samples, filt_h, filt_g, filt_h1, filt_h2, filt_h3, nlevel, opt, nlevel
    )
    n_cols = 3 * (2**nlevel)
    k_map = np.zeros((2 * nlevel, n_cols), dtype=float)
    k_map[0, :] = np.resize(kd[0], n_cols)
    for i in range(1, nlevel):
        k_map[2 * i - 1, :] = np.resize(kd[i], n_cols)
        k_map[2 * i, :] = np.resize(kq[i - 1], n_cols)
    k_map[2 * nlevel - 1, :] = np.resize(kd[nlevel], n_cols)
    return k_map


def level_frequency_axes(nlevel: int, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vertical (level) and horizontal (Hz) axes of the kurtogram image.

    MATLAB::

        Level_w = [0, interleaved 1..nlevel and 1..nlevel+log2(3)-1]
        freq_w  = Fs * ((0:3*2^nlevel-1)/(3*2^(nlevel+1)) + 1/(3*2^(nlevel+2)))
    """
    nlevel = int(nlevel)
    delta = np.log2(3.0) - 1.0
    stacked = np.empty((2, nlevel), dtype=float)
    stacked[0, :] = np.arange(1, nlevel + 1, dtype=float)
    stacked[1, :] = stacked[0, :] + delta
    # MATLAB Level_w(:) is column-major
    interleaved = np.ravel(stacked, order="F")
    level_w = np.concatenate([[0.0], interleaved[: 2 * nlevel - 1]])
    n_freq = 3 * (2**nlevel)
    freq_w = float(fs) * (
        np.arange(n_freq, dtype=float) / (3.0 * 2 ** (nlevel + 1))
        + 1.0 / (3.0 * 2 ** (nlevel + 2))
    )
    return level_w, freq_w


def _k_wpq_filt_local(
    x: np.ndarray,
    filt_h: np.ndarray,
    filt_g: np.ndarray,
    filt_h1: np.ndarray,
    filt_h2: np.ndarray,
    filt_h3: np.ndarray,
    acoeff: np.ndarray,
    bcoeff: Optional[int],
    level: int,
) -> np.ndarray:
    """Walk one path of the packet tree (MATLAB ``K_wpQ_filt_local``)."""
    approx, detail = dbfb(x, filt_h, filt_g)
    if level == 1:
        branch = approx if int(acoeff[level - 1]) == 0 else detail
        if bcoeff is None:
            filt_len = int(
                np.asarray(filt_h).size
                if int(acoeff[level - 1]) == 0
                else np.asarray(filt_g).size
            )
            return np.asarray(branch).ravel()[filt_len - 1 :]
        c1, c2, c3 = tbfb(branch, filt_h1, filt_h2, filt_h3)
        chosen = (c1, c2, c3)[int(bcoeff)]
        return np.asarray(chosen).ravel()[int(np.asarray(filt_h1).size) - 1 :]
    child = approx if int(acoeff[level - 1]) == 0 else detail
    return _k_wpq_filt_local(
        child, filt_h, filt_g, filt_h1, filt_h2, filt_h3, acoeff, bcoeff, level - 1
    )


def k_wpq_filt(
    x: np.ndarray,
    filt_h: np.ndarray,
    filt_g: np.ndarray,
    filt_h1: np.ndarray,
    filt_h2: np.ndarray,
    filt_h3: np.ndarray,
    acoeff: np.ndarray,
    bcoeff: Optional[int],
    level: Optional[int] = None,
) -> np.ndarray:
    """Packet coefficients along ``acoeff`` / ``bcoeff`` (MATLAB ``K_wpQ_filt``)."""
    samples = np.asarray(x).ravel()
    acoeff = np.asarray(acoeff, dtype=int).ravel()
    nlevel = int(acoeff.size)
    if level is None:
        level = nlevel
        max_level = int(np.floor(np.log2(max(samples.size, 1))))
        if nlevel >= max_level:
            raise ValueError("nlevel must be smaller than log2(length(x))")
    if nlevel == 0:
        if bcoeff is None:
            return samples.copy()
        c1, c2, c3 = tbfb(samples, filt_h1, filt_h2, filt_h3)
        chosen = (c1, c2, c3)[int(bcoeff)]
        return np.asarray(chosen).ravel()[int(np.asarray(filt_h1).size) - 1 :]
    return _k_wpq_filt_local(
        samples, filt_h, filt_g, filt_h1, filt_h2, filt_h3, acoeff, bcoeff, int(level)
    )


def find_wav_kurt(
    x: np.ndarray,
    sc: float,
    fr: float,
    fs: float = 1.0,
    opt: str = "kurt2",
    filters: Optional[Tuple[np.ndarray, ...]] = None,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Extract the complex envelope at kurtogram coordinates (MATLAB ``Find_wav_kurt``).

    :param x: 1-D real record (already demeaned, as in ``fast_kurtogram``).
    :param sc: float,
        Decomposition level (``Level_w`` of the peak, MATLAB ``Sc``).
    :param fr: float,
        Carrier frequency **normalised by** ``fs`` (MATLAB ``fi/Fs``, in ``[0, 0.5]``).
    :param fs: float,
        Sampling frequency.
    :param opt: str,
        Kurtosis flavour stored with the envelope.
    :param filters: optional precomputed ``(h, g, h1, h2, h3)``.
    :return: dict with ``c`` (complex envelope), ``bw``, ``fc``, ``kurtosis``.
    """
    samples = np.asarray(x).ravel()
    if filters is None:
        filt_h, filt_g, filt_h1, filt_h2, filt_h3 = analytic_filters()
    else:
        filt_h, filt_g, filt_h1, filt_h2, filt_h3 = filters

    sc = float(sc)
    level = float(np.fix(sc)) + (np.log2(3.0) - 1.0) * float(
        np.remainder(sc, 1.0) >= 0.5
    )
    bandwidth = 2.0 ** (-level - 1.0)
    # MATLAB: freq_w = (0:2^level-1)/(2^(level+1)) + Bw/2  (colon stops at floor)
    n_axis = int(np.round(2.0**level))
    freq_w = np.arange(n_axis, dtype=float) / (2.0 ** (level + 1.0)) + bandwidth / 2.0
    bin_idx = int(np.argmin(np.abs(freq_w - float(fr))))
    fc = float(freq_w[bin_idx])
    node = int(np.round(fc / bandwidth - 0.5))

    if float(level) == np.floor(level):
        acoeff = binary(node, int(level))
        bcoeff = None
        temp_level = int(level)
    else:
        i2 = int(np.fix(node / 3.0))
        temp_level = int(np.fix(level)) - 1
        acoeff = binary(i2, max(temp_level, 0))
        bcoeff = int(node - i2 * 3)
    acoeff = acoeff[::-1]

    envelope = k_wpq_filt(
        samples,
        filt_h,
        filt_g,
        filt_h1,
        filt_h2,
        filt_h3,
        acoeff,
        bcoeff,
        temp_level,
    )
    return {
        "c": np.asarray(envelope),
        "bw": float(bandwidth) * float(fs),
        "fc": float(fc) * float(fs),
        "kurtosis": kurt(envelope, opt=opt),
        "level": float(level),
    }


def prewhiten_ar(x: np.ndarray, order: int = 100) -> np.ndarray:
    """
    AR inverse-filter pre-whitening used in ``demo_Fast_Kurtogram.m``.

    MATLAB ``lpc`` + ``fftfilt``, then drop the first ``order`` samples of
    the filter transient.

    :param x: 1-D real record.
    :param order: int,
        AR order (demo uses 100).
    :return: whitened 1-D array of length ``len(x) - order``.
    """
    samples = np.asarray(x, dtype=float).ravel()
    order = int(order)
    if order < 1:
        raise ValueError("order must be >= 1")
    if samples.size <= order + 1:
        raise ValueError("signal is shorter than the AR order")
    samples = samples - np.mean(samples)
    # Biased autocorrelation, Levinson–Durbin (MATLAB lpc)
    corr = np.correlate(samples, samples, mode="full")
    mid = samples.size - 1
    r = corr[mid : mid + order + 1] / float(samples.size)
    ar = solve_toeplitz(r[:-1], r[1:])
    coeff = np.concatenate([[1.0], -np.asarray(ar, dtype=float)])
    whitened = lfilter(coeff, [1.0], samples)
    return np.asarray(whitened[order:], dtype=float)


def fast_kurtogram(
    x: np.ndarray,
    nlevel: int,
    fs: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Union[float, int, np.ndarray]]]:
    """
    Fast kurtogram of a 1-D record (MATLAB ``Fast_Kurtogram`` without prompts).

    :param x: array_like,
        Real 1-D signal.
    :param nlevel: int,
        Decomposition depth.  Must satisfy ``nlevel <= log2(N) - 7``.
    :param fs: float,
        Sampling frequency (MATLAB default 1).
    :return: ``(Kwav, info)`` where ``Kwav`` has shape
        ``(2*nlevel, 3*2**nlevel)`` (negative kurtosis clipped to 0) and
        ``info`` holds ``Kmax``, ``level``, ``fc``, ``bw``, ``freq_w``,
        ``level_w``, and 0-based peak indices ``row``, ``col``.
    :raises ValueError: If ``nlevel`` is too large for ``len(x)``.
    """
    samples = np.asarray(x, dtype=float).ravel()
    n_samples = int(samples.size)
    nlevel = int(nlevel)
    if nlevel < 1:
        raise ValueError("nlevel must be a positive integer")
    max_level = np.log2(n_samples) - 7.0
    if nlevel > max_level:
        raise ValueError(
            "Please enter a smaller number of decomposition levels "
            "(nlevel <= log2(N)-7 = {:.3f})".format(max_level)
        )
    samples = samples - np.mean(samples)
    filt_h, filt_g, filt_h1, filt_h2, filt_h3 = analytic_filters()
    k_map = k_wpq(
        samples, filt_h, filt_g, filt_h1, filt_h2, filt_h3, nlevel, opt="kurt2"
    )
    k_wav = k_map * (k_map > 0.0)
    level_w, freq_w = level_frequency_axes(nlevel, fs)
    row, col, k_max = max_ij(k_wav)
    fi = col / 3.0 / 2 ** (nlevel + 1) + 2.0 ** (-2.0 - level_w[row])
    bandwidth = float(fs) * 2.0 ** (-(level_w[row] + 1.0))
    info: Dict[str, Union[float, int, np.ndarray]] = {
        "Kmax": float(k_max),
        "level": float(level_w[row]),
        "fc": float(fs) * float(fi),
        "bw": float(bandwidth),
        "freq_w": freq_w,
        "level_w": level_w,
        "row": row,
        "col": col,
        "fs": float(fs),
        "nlevel": nlevel,
        "filters": (filt_h, filt_g, filt_h1, filt_h2, filt_h3),
        "x": samples,
    }
    return k_wav, info


def plot_kurtogram(
    k_wav: np.ndarray,
    info: Dict[str, Union[float, int, np.ndarray]],
    ax: Optional[plt.Axes] = None,
):
    """
    Display a kurtogram the way MATLAB ``imagesc`` does.

    :param k_wav: kurtogram matrix from :func:`fast_kurtogram`.
    :param info: companion dict (needs ``freq_w``, ``level_w``, peak fields).
    :param ax: matplotlib axes; a new figure is created when omitted.
    :return: the axes.
    """
    freq_w = np.asarray(info["freq_w"], dtype=float)
    level_w = np.asarray(info["level_w"], dtype=float)
    nlevel = int(info["nlevel"])
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 4.8))
    mesh = ax.imshow(
        np.asarray(k_wav, dtype=float),
        aspect="auto",
        origin="upper",
        extent=[freq_w[0], freq_w[-1], 2 * nlevel, 1],
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(1, 2 * nlevel + 1))
    ax.set_yticklabels(["{:.1f}".format(v) for v in np.round(level_w * 10.0) / 10.0])
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("level k")
    k_max = float(info["Kmax"])
    level = float(info["level"])
    bw = float(info["bw"])
    fc = float(info["fc"])
    ax.set_title(
        "fb-kurt.2 - $K_{{max}}$={:.1f} @ level {:.1f}, Bw={:g} Hz, $f_c$={:g} Hz".format(
            np.round(10.0 * k_max) / 10.0,
            np.fix(10.0 * level) / 10.0,
            bw,
            fc,
        )
    )
    plt.colorbar(mesh, ax=ax)
    return ax
