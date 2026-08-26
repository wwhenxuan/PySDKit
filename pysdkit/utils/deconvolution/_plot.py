# -*- coding: utf-8 -*-
"""
Paper-style harmonic markers for envelope spectra.

The IMCKD / ACYCBD / SMHD figures label the fault line and its
integer multiples on the Hilbert envelope spectrum, typically

    f_o,  2 f_o,  3 f_o,  ...

with an arrow on each local peak (Miao et al., MSSP 92, 2017, Fig. 8).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.text import Annotation

from ._common import peak_frequency


def harmonic_label(order: int, symbol: str = r"f_o") -> str:
    """
    Mathtext label for harmonic ``order`` of ``symbol``.

    ``order=1, symbol='f_o'`` → ``$f_o$``;
    ``order=3`` → ``$3f_o$``.  Pass ``f_i``, ``f_g``, ``\\mathrm{BPFI}``
    for inner-race / gear-mesh / literal names.
    """
    order = int(order)
    if order < 1:
        raise ValueError("order must be >= 1")
    body = symbol.strip()
    if body.startswith("$") and body.endswith("$"):
        body = body[1:-1]
    if order == 1:
        return r"${}$".format(body)
    return r"${}{}$".format(order, body)


def harmonic_peaks(
    freq: np.ndarray,
    magnitude: np.ndarray,
    fundamental: float,
    n_harmonics: int = 6,
    f_max: Optional[float] = None,
    window: Optional[float] = None,
) -> np.ndarray:
    """
    Locate envelope-spectrum peaks at ``k * fundamental``.

    Each harmonic is the **argmax** of ``magnitude`` inside a window
    of half-width ``window`` (default ``0.3 f_0``, capped at
    ``0.49 f_0`` so neighbours do not steal each other).

    :return: structured array with fields ``order``, ``frequency``,
        ``amplitude``.
    """
    freq = np.asarray(freq, dtype=float).ravel()
    magnitude = np.asarray(magnitude, dtype=float).ravel()
    if freq.size != magnitude.size or freq.size == 0:
        raise ValueError("freq and magnitude must be non-empty and aligned")
    fund = float(fundamental)
    if fund <= 0.0:
        raise ValueError("fundamental must be > 0")
    n_harm = int(n_harmonics)
    if n_harm < 1:
        raise ValueError("n_harmonics must be >= 1")
    limit = float(np.max(freq)) if f_max is None else float(f_max)
    half = 0.3 * fund if window is None else float(window)
    if half <= 0.0:
        raise ValueError("window must be > 0")
    half = min(half, 0.49 * fund)

    orders: List[int] = []
    frequencies: List[float] = []
    amplitudes: List[float] = []
    for order in range(1, n_harm + 1):
        target = order * fund
        if target - half > limit:
            break
        mask = (freq >= target - half) & (freq <= min(target + half, limit))
        mask &= freq > 0.0
        if not np.any(mask):
            continue
        local = np.where(mask)[0]
        index = int(local[int(np.argmax(magnitude[local]))])
        orders.append(order)
        frequencies.append(float(freq[index]))
        amplitudes.append(float(magnitude[index]))

    return np.array(
        list(zip(orders, frequencies, amplitudes)),
        dtype=[
            ("order", np.int32),
            ("frequency", np.float64),
            ("amplitude", np.float64),
        ],
    )


def annotate_harmonics(
    freq: np.ndarray,
    magnitude: np.ndarray,
    fundamental: float,
    n_harmonics: int = 6,
    *,
    ax: Optional[Axes] = None,
    symbol: str = r"f_o",
    f_max: Optional[float] = None,
    window: Optional[float] = None,
    plot_spectrum: Optional[bool] = None,
    color: str = "black",
    fontsize: float = 12,
    x_offset: float = 8.0,
    y_offset: float = 14.0,
) -> Tuple[Axes, np.ndarray, List[Annotation]]:
    """
    Draw ``f_o, 2f_o, …`` arrows on an envelope spectrum.

    Matches the bearing-diagnosis figures: a thin black arrow from the
    mathtext label (upper right of the peak) onto the local maximum
    nearest each multiple of ``fundamental``.

    :param freq: frequency axis in hertz.
    :param magnitude: spectrum samples aligned with ``freq``.
    :param fundamental: characteristic frequency $$f_o$$ (or BPFI, …).
    :param n_harmonics: highest integer multiple to mark (default 6).
    :param ax: existing axes; a new figure is created when omitted.
    :param symbol: mathtext body, default ``f_o``.  Use ``f_i`` for
        inner race, ``\\mathrm{BPFO}`` for an upright name.
    :param f_max: stop marking above this frequency (default: last bin
        or the axes x-limit when already zoomed).
    :param window: half-width in hertz around each multiple.
    :param plot_spectrum: draw the spectrum line.  Defaults to True
        only when ``ax`` is created here.
    :param color: label and arrow colour.
    :param fontsize: mathtext size.
    :param x_offset: label offset in *points* (right of the peak).
    :param y_offset: label offset in *points* (above the peak).
    :return: ``(ax, peaks, annotations)``.
    """
    created = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 4.0))
    if plot_spectrum is None:
        plot_spectrum = created
    if plot_spectrum:
        ax.plot(
            np.asarray(freq, dtype=float),
            np.asarray(magnitude, dtype=float),
            color="C0",
            lw=0.8,
        )
        if ax.get_xlabel() == "":
            ax.set_xlabel("Frequency [Hz]")
        if ax.get_ylabel() == "":
            ax.set_ylabel("Amplitude")

    if f_max is None:
        x_left, x_right = ax.get_xlim()
        data_max = float(np.max(np.asarray(freq, dtype=float)))
        # Respect a zoomed x-axis (e.g. [0, 200] in the MATLAB demos)
        if np.isfinite(x_right) and x_right > x_left:
            f_max = min(data_max, float(x_right))
        else:
            f_max = data_max

    peaks = harmonic_peaks(
        freq,
        magnitude,
        fundamental,
        n_harmonics=n_harmonics,
        f_max=f_max,
        window=window,
    )
    annotations: List[Annotation] = []
    if peaks.size == 0:
        return ax, peaks, annotations

    y_low, y_high = ax.get_ylim()
    amp_max = float(np.max(peaks["amplitude"]))
    ax.set_ylim(min(y_low, 0.0), max(y_high, amp_max * 1.28))

    for row in peaks:
        order = int(row["order"])
        site = (float(row["frequency"]), float(row["amplitude"]))
        text = harmonic_label(order, symbol=symbol)
        handle = ax.annotate(
            text,
            xy=site,
            xytext=(float(x_offset), float(y_offset)),
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 0.9,
                "shrinkA": 0,
                "shrinkB": 1.5,
            },
            fontsize=fontsize,
            color=color,
            ha="left",
            va="bottom",
            clip_on=False,
        )
        annotations.append(handle)
    return ax, peaks, annotations


def marked_envelope_spectrum(
    freq: np.ndarray,
    magnitude: np.ndarray,
    fundamental: Optional[float] = None,
    n_harmonics: int = 6,
    *,
    ax: Optional[Axes] = None,
    symbol: str = r"f_o",
    f_max: Optional[float] = None,
    **kwargs,
) -> Tuple[Axes, np.ndarray]:
    """
    Plot an envelope spectrum and optionally mark its harmonic series.

    If ``fundamental`` is omitted, the largest bin in ``(0, f_max]`` is
    used (via :func:`peak_frequency`).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 4.0))
    freq = np.asarray(freq, dtype=float).ravel()
    magnitude = np.asarray(magnitude, dtype=float).ravel()
    ax.plot(freq, magnitude, color="C0", lw=0.8)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Amplitude")
    if f_max is not None:
        ax.set_xlim(0.0, float(f_max))
    fund = fundamental
    if fund is None:
        fund = peak_frequency(freq, magnitude, f_max=f_max)
    ax, peaks, _ann = annotate_harmonics(
        freq,
        magnitude,
        fund,
        n_harmonics=n_harmonics,
        ax=ax,
        symbol=symbol,
        f_max=f_max,
        plot_spectrum=False,
        **kwargs,
    )
    return ax, peaks
