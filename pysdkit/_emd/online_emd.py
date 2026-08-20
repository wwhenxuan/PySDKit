# -*- coding: utf-8 -*-
"""
Online Empirical Mode Decomposition (Online EMD / OEMD).

Faithful Python port of Fontugne, Borgnat and Flandrin's MATLAB
``oemd_init.m`` / ``oemd_iter.m``.  Classical EMD revisits the whole record
at every sifting step, so it cannot follow a stream and its cost grows with
the accumulated length.  Online EMD instead:

1. slides a window that holds ``n_extrema`` consecutive extrema of the
   current residual;
2. extracts **one** IMF of that short window with a standard EMD sifter;
3. stitches overlapping local IMFs with a truncated-Gaussian window
   (paper :math:`\\tau = 3`) and commits the samples that leave the
   window, pushing their residual to the next IMF stage.

Fontugne, R., Borgnat, P. and Flandrin, P.
"Online Empirical Mode Decomposition."
IEEE ICASSP, New Orleans, 2017.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from ._find_extrema import find_extrema_simple
from .emd import EMD

_MIN_N_EXTREMA = 4
_BOUND = 3.0
_FIXE_SIFTINGS = 10
_FIXE_H = 4

_ALGO_NAMES = {
    0: 0,
    "0": 0,
    "rilling": 0,
    "stop": 0,
    1: 1,
    "1": 1,
    "fixe": 1,
    "fix": 1,
    "siftings": 1,
    2: 2,
    "2": 2,
    "fix_h": 2,
    "fixh": 2,
    "huang": 2,
}


def parse_emd_algo(emd_algo: Union[int, str]) -> int:
    """
    Map an ``emdAlgo`` flag onto ``0`` (Rilling-like), ``1`` (10 siftings)
    or ``2`` (Huang ``FIX_H=4``).

    MATLAB ``oemd_init`` uses integer codes.  The C mex paths ``emdc`` /
    ``emdc_fix`` are replaced by :class:`pysdkit._emd.emd.EMD` with the
    matching stopping rule.

    :param emd_algo: int or str,
        Algorithm selector (see ``_ALGO_NAMES``).
    :return: int,
        Canonical code in ``{0, 1, 2}``.
    :raises ValueError: If the selector is unknown.
    """
    if isinstance(emd_algo, str):
        key: Union[int, str] = emd_algo.strip().lower()
    else:
        key = int(emd_algo)
    if key not in _ALGO_NAMES:
        raise ValueError(
            "emd_algo must be 0/'rilling', 1/'fixe' or 2/'fix_h'; got "
            "{!r}".format(emd_algo)
        )
    return _ALGO_NAMES[key]


def matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """
    MATLAB ``start:step:stop`` for a strictly positive step.

    :param start: float,
        First value.
    :param step: float,
        Increment (must be ``> 0``).
    :param stop: float,
        Last value not to be exceeded.
    :return: 1-D float array.
    """
    start = float(start)
    step = float(step)
    stop = float(stop)
    if step <= 0.0:
        raise ValueError("matlab_colon requires a positive step")
    n_pts = int(np.floor((stop - start) / step + 1e-12)) + 1
    if n_pts <= 0:
        return np.zeros(0, dtype=float)
    return start + np.arange(n_pts, dtype=float) * step


def _colon_or_linspace(
    start: float, step: float, stop: float, length: int
) -> np.ndarray:
    """MATLAB colon with a linspace fallback when rounding drops a sample."""
    if length <= 0:
        return np.zeros(0, dtype=float)
    grid = matlab_colon(start, step, stop)
    if grid.size != length:
        grid = np.linspace(start, stop, length)
    return grid


def truncated_gaussian(ind: np.ndarray, bound: float = _BOUND) -> np.ndarray:
    """
    Truncated standard-normal window used for IMF stitching.

    MATLAB::

        w = (1/sqrt(2*pi))*exp(-ind.^2/2) - (1/sqrt(2*pi))*exp(-bound^2/2)

    which is :math:`\\varphi(s)-\\varphi(\\tau)` with :math:`\\tau=` ``bound``
    (paper: :math:`\\tau = 3`).  The value is zero at ``|s|=bound``.

    :param ind: array_like,
        Warped sample coordinates.
    :param bound: float,
        Truncation abscissa ``tau``.
    :return: weights, same shape as ``ind``.
    """
    ind = np.asarray(ind, dtype=float)
    scale = 1.0 / np.sqrt(2.0 * np.pi)
    return scale * (np.exp(-0.5 * ind**2) - np.exp(-0.5 * float(bound) ** 2))


def extrema_indices(signal: np.ndarray) -> np.ndarray:
    """
    0-based indices of local maxima and minima (Flandrin ``extr``).

    Uses :func:`find_extrema_simple`, the same first-difference / plateau
    rule as the Flandrin toolbox ``extr.m`` that ``oemd_iter`` calls.

    :param signal: array_like,
        Real 1-D samples.
    :return: sorted unique integer indices.
    """
    samples = np.asarray(signal, dtype=float).ravel()
    n_samples = samples.size
    if n_samples < 3:
        return np.zeros(0, dtype=int)
    time = np.arange(n_samples, dtype=float)
    max_pos, _, min_pos, _, _ = find_extrema_simple(time, samples)
    idx = np.concatenate(
        [np.atleast_1d(np.asarray(max_pos)), np.atleast_1d(np.asarray(min_pos))]
    )
    if idx.size == 0:
        return np.zeros(0, dtype=int)
    return np.unique(np.round(idx).astype(int))


def extract_first_imf(window: np.ndarray, emd_algo: int) -> np.ndarray:
    """
    Extract the fastest IMF of a short window (MATLAB ``MAXMODES=1``).

    Residue appended by :meth:`EMD.fit_transform` is discarded; only the
    first row is returned, matching ``acumIMF(1,:)``.

    :param window: array_like,
        Real 1-D window ``data(windowTail:newWindowHead+1)``.
    :param emd_algo: int,
        ``0`` default / Rilling-like stop, ``1`` ``FIXE=10``,
        ``2`` ``FIXE_H=4``.
    :return: 1-D IMF, same length as ``window``.
    """
    samples = np.asarray(window, dtype=float).ravel()
    if samples.size == 0:
        return np.zeros(0, dtype=float)
    if samples.size < 3:
        return samples.copy()

    if emd_algo == 1:
        sifter = EMD(max_imfs=1, FIXE=_FIXE_SIFTINGS)
    elif emd_algo == 2:
        sifter = EMD(max_imfs=1, FIXE_H=_FIXE_H)
    elif emd_algo == 0:
        sifter = EMD(max_imfs=1)
    else:
        raise ValueError("unknown emd_algo {}".format(emd_algo))

    imfs = np.asarray(sifter.fit_transform(samples, max_imfs=1), dtype=float)
    if imfs.ndim == 1:
        first = imfs
    else:
        first = imfs[0]
    if first.size != samples.size:
        out = np.zeros(samples.size, dtype=float)
        n_copy = min(first.size, samples.size)
        out[:n_copy] = first[:n_copy]
        return out
    return np.asarray(first, dtype=float)


def first_window_weights(
    extr_ind: np.ndarray,
    n_extrema: int,
    bound: float = _BOUND,
) -> np.ndarray:
    """
    Stitching weights for the first window of a stage (``windowTail==1``).

    Ports the ``else`` branch of MATLAB ``oemd_iter.m``.  The array grows
    to the last extremum (MATLAB assignment past ``zeros`` length).

    :param extr_ind: array_like,
        0-based extrema indices, length at least ``n_extrema``.
    :param n_extrema: int,
        Window length ``l`` (MATLAB ``nbExtrema``).
    :param bound: float,
        Gaussian truncation ``tau``.
    :return: 1-D weights of length ``extr_ind[n_extrema-1] + 1``.
    """
    extr = np.asarray(extr_ind[:n_extrema], dtype=int).ravel()
    n_head = int(extr[n_extrema - 1])
    weights = np.zeros(n_head + 1, dtype=float)
    weights[: int(extr[1]) + 1] = 1.0
    scale = bound / ((n_extrema - 1) / 2.0)
    for i_m in range(2, n_extrema):
        i0 = i_m - 1
        start = int(extr[i0]) + 1
        stop_incl = int(extr[i0 + 1])
        data_length = stop_incl - start + 1
        if data_length <= 0:
            continue
        acc = np.zeros(data_length, dtype=float)
        for j_m in range(i_m, n_extrema):
            last = j_m + 1.0 - (n_extrema + 1) / 2.0
            colon_start = (1.0 / data_length) + (last - 1.0)
            grid = _colon_or_linspace(colon_start, 1.0 / data_length, last, data_length)
            acc = acc + truncated_gaussian(grid * scale, bound=bound)
        weights[start : stop_incl + 1] = acc
    return weights


def sliding_window_weights(
    extr_ind: np.ndarray,
    window_tail: int,
    n_extrema: int,
    bound: float = _BOUND,
) -> np.ndarray:
    """
    Stitching weights for every window after the first (``windowTail~=1``).

    Warps each inter-extremum segment onto a truncated Gaussian, then
    forces the first sample to ``-bound`` (weight 0) as in MATLAB.

    :param extr_ind: array_like,
        0-based extrema indices.
    :param window_tail: int,
        0-based MATLAB ``windowTail-1``.
    :param n_extrema: int,
        Window length ``l``.
    :param bound: float,
        Gaussian truncation ``tau``.
    :return: 1-D weights of length ``extr[l-1] - extr[0] + 1``.
    """
    extr = np.asarray(extr_ind[:n_extrema], dtype=int).ravel()
    n_weights = int(extr[n_extrema - 1] - extr[0] + 1)
    ind = np.zeros(n_weights, dtype=float)
    last_extrema = np.arange(1, n_extrema, dtype=float) + 1.0 - (n_extrema + 1) / 2.0
    for i in range(n_extrema - 1):
        i1 = int(extr[i] - window_tail)
        i2 = int(extr[i + 1] - window_tail)
        length = i2 - i1
        if length <= 0:
            continue
        start = (1.0 / length) + (last_extrema[i] - 1.0)
        stop = float(last_extrema[i])
        ind[i1:i2] = _colon_or_linspace(start, 1.0 / length, stop, length)
    ind *= bound / ((n_extrema - 1) / 2.0)
    if ind.size:
        ind[0] = -bound
    return truncated_gaussian(ind, bound=bound)


def fig2_signal(stop: float = 10000.0, step: float = 0.5):
    """
    Synthetic mixture from MATLAB ``example_oemd_fig2.m`` / paper Figure 2.

    ::

        samp  = pi/2 : step : stop
        x = sin(samp)
            + sin(linspace(pi/2, stop/10, N))
            + sin(linspace(pi/2, stop/30, N))
            + linspace(0, 10, N)

    :param stop: float,
        MATLAB colon endpoint (paper uses ``10000``).
    :param step: float,
        MATLAB colon step (paper uses ``0.5``).
    :return: dict with ``samp``, ``comp1``, ``comp2``, ``comp3``,
        ``trend``, ``signal``.
    """
    samp = matlab_colon(np.pi / 2.0, float(step), float(stop))
    n_samples = samp.size
    comp1 = np.sin(samp)
    comp2 = np.sin(np.linspace(np.pi / 2.0, float(stop) / 10.0, n_samples))
    comp3 = np.sin(np.linspace(np.pi / 2.0, float(stop) / 30.0, n_samples))
    trend = np.linspace(0.0, 10.0, n_samples)
    signal = comp1 + comp2 + comp3 + trend
    return {
        "samp": samp,
        "comp1": comp1,
        "comp2": comp2,
        "comp3": comp3,
        "trend": trend,
        "signal": signal,
    }


class OEMDStage:
    """
    One IMF level of the Online EMD pipeline (MATLAB ``stage`` struct).

    Indices ``window_tail`` and ``window_head`` are **0-based** (MATLAB
    ``windowTail-1`` / ``windowHead-1``).  ``window_head == 0`` means the
    stage has never been slid, matching MATLAB ``windowHead==1``.
    """

    __slots__ = (
        "data",
        "imf",
        "weights",
        "window_tail",
        "window_head",
        "emd_algo",
        "n_extrema",
        "max_imf",
    )

    def __init__(
        self,
        emd_algo: int,
        n_extrema: int,
        max_imf: int,
    ) -> None:
        self.data = np.zeros(0, dtype=float)
        self.imf = np.zeros(0, dtype=float)
        self.weights = np.zeros(0, dtype=float)
        self.window_tail = 0
        self.window_head = 0
        self.emd_algo = int(emd_algo)
        self.n_extrema = int(n_extrema)
        self.max_imf = int(max_imf)

    def __repr__(self) -> str:
        return "OEMDStage(n_data={}, n_imf={}, tail={}, head={}, max_imf={})".format(
            self.data.size,
            self.imf.size,
            self.window_tail,
            self.window_head,
            self.max_imf,
        )


def oemd_init(
    max_imfs: int = -1,
    n_extrema: int = 10,
    emd_algo: Union[int, str] = 2,
) -> List[OEMDStage]:
    """
    Create the initial stage list (MATLAB ``oemd_init``).

    :param max_imfs: int,
        ``-1`` for unlimited IMFs; ``0`` extracts none; ``k>0`` extracts
        at most ``k`` IMFs.
    :param n_extrema: int,
        Sliding-window length in extrema (MATLAB ``nbExtrema``).
    :param emd_algo: int or str,
        Local-EMD stopping rule (see :func:`parse_emd_algo`).
    :return: list with a single empty :class:`OEMDStage`.
    """
    algo = parse_emd_algo(emd_algo)
    n_ext = int(n_extrema)
    if n_ext < _MIN_N_EXTREMA:
        raise ValueError(
            "n_extrema must be >= {} (paper / examples use 10); got {}".format(
                _MIN_N_EXTREMA, n_extrema
            )
        )
    return [OEMDStage(emd_algo=algo, n_extrema=n_ext, max_imf=int(max_imfs))]


def _ensure_next_stage(
    stages: List[OEMDStage], parent: OEMDStage, index: int
) -> OEMDStage:
    """
    Create ``stages[index+1]`` if missing.

    MATLAB ``oemd_iter`` grows ``stream`` in place.  A Python slice such as
    ``stages[1:]`` is a *new* list, so children must be appended on the
    original pipeline using an index rather than ``stages[1]``.
    """
    child_index = index + 1
    if len(stages) <= child_index:
        stages.append(
            OEMDStage(
                emd_algo=parent.emd_algo,
                n_extrema=parent.n_extrema,
                max_imf=parent.max_imf - 1,
            )
        )
    return stages[child_index]


def _append_residual(child: OEMDStage, residual: np.ndarray) -> None:
    residual = np.asarray(residual, dtype=float).ravel()
    if residual.size == 0:
        return
    child.data = np.concatenate([child.data, residual])


def oemd_iter(
    stages: List[OEMDStage],
    bound: float = _BOUND,
    start: int = 0,
) -> List[OEMDStage]:
    """
    Advance every IMF stage as far as the current buffers allow.

    Direct port of MATLAB ``oemd_iter.m``: while the current residual has
    at least ``n_extrema`` extrema, extract one local IMF, stitch it, commit
    the samples that leave the window, and push their residual to the next
    stage.  Then recurse on the next stage of the **same** list (MATLAB
    ``stream = [stream(1) oemd_iter(stream(2:end))]``).

    :param stages: list of :class:`OEMDStage`,
        Pipeline; ``stages[0]`` is the fastest IMF.
    :param bound: float,
        Gaussian truncation (MATLAB ``bound = 3``).
    :param start: int,
        Index of the stage to process (internal recursion).
    :return: the same list, mutated in place.
    """
    if start >= len(stages):
        return stages

    stage = stages[start]
    n_ext = stage.n_extrema
    if stage.max_imf == 0:
        return stages

    local = extrema_indices(stage.data[stage.window_tail :])
    extr = local.astype(int) + int(stage.window_tail)
    if extr.size < n_ext:
        return stages

    while extr.size >= n_ext:
        if stage.window_tail != 0 and int(extr[0]) != stage.window_tail + 1:
            extr = np.concatenate([[stage.window_tail + 1], extr.astype(int)])

        new_head = int(extr[n_ext - 1])
        window = stage.data[stage.window_tail : new_head + 2]
        acum = extract_first_imf(window, stage.emd_algo)
        prev_head = int(stage.window_head)

        if stage.window_tail != 0:
            weights = sliding_window_weights(
                extr, stage.window_tail, n_ext, bound=bound
            )
            core = acum[1:-1] if acum.size >= 2 else np.zeros(0, dtype=float)
            n_common = min(weights.size, core.size)
            weighted = weights[:n_common] * core[:n_common]
            weight_seg = weights[:n_common]

            dest_start = stage.window_tail + 1
            overlap_end = min(prev_head, dest_start + weighted.size)
            n_overlap = max(0, overlap_end - dest_start)
            n_overlap = min(n_overlap, weighted.size)

            need = dest_start + weighted.size
            if stage.imf.size < need:
                stage.imf = np.concatenate([stage.imf, np.zeros(need - stage.imf.size)])
            if stage.weights.size < need:
                stage.weights = np.concatenate(
                    [stage.weights, np.zeros(need - stage.weights.size)]
                )

            if n_overlap > 0:
                stage.imf[dest_start:overlap_end] = (
                    stage.imf[dest_start:overlap_end] + weighted[:n_overlap]
                )
                stage.weights[dest_start:overlap_end] = (
                    stage.weights[dest_start:overlap_end] + weight_seg[:n_overlap]
                )
                stage.imf[overlap_end:need] = weighted[n_overlap:]
                stage.weights[overlap_end:need] = weight_seg[n_overlap:]
            else:
                stage.imf[dest_start:need] = weighted
                stage.weights[dest_start:need] = weight_seg

            leave_end = int(extr[1])
            sl = slice(dest_start, leave_end)
            denom = stage.weights[sl].copy()
            tiny = np.abs(denom) < 1e-30
            denom[tiny] = 1.0
            stage.imf[sl] = stage.imf[sl] / denom
        else:
            weights = first_window_weights(extr, n_ext, bound=bound)
            core = acum[:-1] if acum.size else acum
            n_common = min(weights.size, core.size)
            stage.imf = core[:n_common] * weights[:n_common]
            stage.weights = weights[:n_common]

        next_tail_m = int(extr[1])
        if stage.window_tail != 0:
            residual = (
                stage.data[stage.window_tail + 1 : next_tail_m]
                - stage.imf[stage.window_tail + 1 : next_tail_m]
            )
        else:
            n_res = min(next_tail_m, stage.data.size, stage.imf.size)
            residual = stage.data[:n_res] - stage.imf[:n_res]

        child = _ensure_next_stage(stages, stage, start)
        _append_residual(child, residual)

        stage.window_tail = max(0, next_tail_m - 1)
        stage.window_head = new_head
        extr = extr[1:]

    oemd_iter(stages, bound=bound, start=start + 1)
    return stages


def residual_stage_index(stages: Sequence[OEMDStage]) -> int:
    """
    Index of the unused residual holder (MATLAB ``plotIMFs`` scan).

    Walks until ``window_head == 0`` (MATLAB ``windowHead==1``).  If every
    stage has been slid, returns ``len(stages)``.
    """
    idx = 0
    while idx < len(stages) and stages[idx].window_head != 0:
        idx += 1
    return idx


def stages_to_imfs(
    stages: Sequence[OEMDStage],
    n_samples: Optional[int] = None,
    fill: float = 0.0,
) -> np.ndarray:
    """
    Stack committed IMFs and the residual into an ``(n_imfs, n_samples)`` array.

    Unfilled tails (the lag of Online EMD) are set to ``fill``.  The last
    row is the residual buffer of the first un-slid stage, matching
    ``plotIMFs.m``.

    :param stages: sequence of :class:`OEMDStage`.
    :param n_samples: int or None,
        Row length.  Default is ``len(stages[0].data)``.
    :param fill: float,
        Value used for samples that have not been stitched yet.
    :return: 2-D array, possibly with a single residual row.
    """
    if not stages:
        return np.zeros((0, 0), dtype=float)
    if n_samples is None:
        n_samples = int(stages[0].data.size)
    n_samples = int(n_samples)
    res_idx = residual_stage_index(stages)
    rows: List[np.ndarray] = []

    def _pad(vec: np.ndarray) -> np.ndarray:
        out = np.full(n_samples, fill, dtype=float)
        n_copy = min(vec.size, n_samples)
        out[:n_copy] = vec[:n_copy]
        return out

    for idx in range(res_idx):
        rows.append(_pad(stages[idx].imf))
    if res_idx < len(stages):
        rows.append(_pad(stages[res_idx].data))
    elif not rows:
        rows.append(_pad(stages[0].data))
    return np.vstack(rows)


class OnlineEMD:
    """
    Online Empirical Mode Decomposition (Online EMD).

    Sliding-window EMD with truncated-Gaussian stitching, after Fontugne,
    Borgnat and Flandrin, ICASSP 2017.  Feed samples with :meth:`append`
    / :meth:`update` for a stream, or :meth:`fit_transform` for a batch.

    The local sifter is :class:`pysdkit._emd.emd.EMD` (the MATLAB C mex
    ``emdc`` / ``emdc_fix`` is not shipped).  ``emd_algo=2`` (Huang
    ``S=4``) matches the pure-MATLAB ``emd(..., 'FIX_H', 4)`` path.

    :param n_extrema: int,
        Extrema per sliding window ``l``.  Paper Figure 2 uses ``10``.
        Must be ``>= 4``; examples recommend ``>= 10``.
    :param max_imfs: int,
        Maximum IMFs (``-1`` unlimited), MATLAB ``maxIMF``.
    :param emd_algo: int or str,
        ``0`` / ``'rilling'`` — default EMD stop (Rilling-like);
        ``1`` / ``'fixe'`` — 10 siftings;
        ``2`` / ``'fix_h'`` — Huang criterion ``FIX_H=4`` (default).
    :param bound: float,
        Gaussian truncation :math:`\\tau` (paper / MATLAB: ``3``).
    """

    def __init__(
        self,
        n_extrema: int = 10,
        max_imfs: int = -1,
        emd_algo: Union[int, str] = 2,
        bound: float = _BOUND,
    ) -> None:
        """
        Store hyperparameters and initialise an empty stage list.

        :param n_extrema: int,
            Sliding-window length in extrema (must be ``>= 4``).
        :param max_imfs: int,
            ``-1`` unlimited, ``0`` none, ``k>0`` at most ``k`` IMFs.
        :param emd_algo: int or str,
            Local-EMD stopping rule.
        :param bound: float,
            Truncated-Gaussian abscissa (must be ``> 0``).
        :return: None
        """
        if int(n_extrema) < _MIN_N_EXTREMA:
            raise ValueError(
                "n_extrema must be >= {} (paper / examples use 10); got {}".format(
                    _MIN_N_EXTREMA, n_extrema
                )
            )
        if float(bound) <= 0.0:
            raise ValueError("bound must be positive; got {}".format(bound))
        self.n_extrema = int(n_extrema)
        self.max_imfs = int(max_imfs)
        self.emd_algo = parse_emd_algo(emd_algo)
        self.bound = float(bound)
        self.stages: List[OEMDStage] = oemd_init(
            max_imfs=self.max_imfs,
            n_extrema=self.n_extrema,
            emd_algo=self.emd_algo,
        )
        self.imfs: Optional[np.ndarray] = None
        self.residue: Optional[np.ndarray] = None

    def __str__(self) -> str:
        """
        Return the full algorithm name and abbreviation.

        :return: str,
            ``"Online Empirical Mode Decomposition (Online EMD)"``.
        """
        return "Online Empirical Mode Decomposition (Online EMD)"

    def __call__(
        self, signal: np.ndarray, max_imfs: Optional[int] = None
    ) -> np.ndarray:
        """Allow instances to be called like functions."""
        return self.fit_transform(signal, max_imfs=max_imfs)

    def reset(self, max_imfs: Optional[int] = None) -> List[OEMDStage]:
        """
        Drop buffered samples and start a new decomposition.

        :param max_imfs: int or None,
            Override ``self.max_imfs`` when given.
        :return: the new one-element stage list.
        """
        if max_imfs is not None:
            self.max_imfs = int(max_imfs)
        self.stages = oemd_init(
            max_imfs=self.max_imfs,
            n_extrema=self.n_extrema,
            emd_algo=self.emd_algo,
        )
        self.imfs = None
        self.residue = None
        return self.stages

    def append(self, samples: np.ndarray) -> None:
        """
        Append samples to the fastest-IMF buffer (MATLAB ``stage(1).data``).

        :param samples: array_like,
            Real 1-D chunk.  Empty input is ignored.
        :return: None
        :raises ValueError: If the chunk is not 1-D.
        """
        chunk = np.asarray(samples, dtype=float).ravel()
        if np.asarray(samples).ndim > 1 and min(np.asarray(samples).shape) > 1:
            raise ValueError(
                "Online EMD expects a univariate 1-D signal; got shape {}".format(
                    np.asarray(samples).shape
                )
            )
        if chunk.size == 0:
            return
        if not self.stages:
            self.reset()
        self.stages[0].data = np.concatenate([self.stages[0].data, chunk])

    def iterate(self) -> List[OEMDStage]:
        """
        Run :func:`oemd_iter` on the current buffers.

        :return: updated ``self.stages``.
        """
        if not self.stages:
            self.reset()
        oemd_iter(self.stages, bound=self.bound)
        self._store_imfs()
        return self.stages

    def update(self, samples: np.ndarray) -> np.ndarray:
        """
        Append a chunk and iterate (one streaming step).

        :param samples: array_like,
            New univariate samples.
        :return: current IMF matrix from :meth:`get_imfs`.
        """
        self.append(samples)
        self.iterate()
        return self.get_imfs()

    def fit_transform(
        self, signal: np.ndarray, max_imfs: Optional[int] = None
    ) -> np.ndarray:
        """
        Decompose a whole record (dump-then-iterate, as in ``example_ecg_fig5``).

        :param signal: array_like,
            Univariate 1-D signal.
        :param max_imfs: int or None,
            Override the constructor cap for this call.
        :return: array of shape ``(n_imfs, n_samples)``.
            The last row is the residual.  Samples that have not left the
            sliding window yet are ``0`` (Online EMD lag).
        """
        self.reset(max_imfs=max_imfs)
        self.append(signal)
        self.iterate()
        return self.get_imfs()

    def get_imfs(self, fill: float = 0.0) -> np.ndarray:
        """
        Stack IMFs and residual, padded to the length of the input buffer.

        :param fill: float,
            Value for samples that are still inside the stitching lag.
        :return: ``(n_imfs, n_samples)`` array.
        """
        matrix = stages_to_imfs(self.stages, fill=fill)
        self._store_imfs(matrix)
        return matrix

    def committed_length(self) -> int:
        """
        Number of samples whose fastest IMF has been committed.

        After at least one window, MATLAB ``windowTail`` is the last
        committed index (1-based), so the prefix length is
        ``window_tail + 1`` in 0-based indexing.  Returns ``0`` if the
        first stage has never slid.
        """
        if not self.stages:
            return 0
        stage = self.stages[0]
        if stage.window_head == 0:
            return 0
        return int(stage.window_tail) + 1

    def _store_imfs(self, matrix: Optional[np.ndarray] = None) -> None:
        if matrix is None:
            matrix = stages_to_imfs(self.stages, fill=0.0)
        if matrix.size == 0:
            self.imfs = matrix
            self.residue = np.zeros(0, dtype=float)
            return
        if matrix.shape[0] == 1:
            self.imfs = np.zeros((0, matrix.shape[1]), dtype=float)
            self.residue = matrix[0].copy()
        else:
            self.imfs = matrix[:-1].copy()
            self.residue = matrix[-1].copy()
