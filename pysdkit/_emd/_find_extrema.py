# -*- coding: utf-8 -*-
"""
Created on Sat Mar 5 21:38:26 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
The following code is mainly used to find extreme points in the EMD algorithm

Code taken from https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EMD.py
"""
import numpy as np
from pysdkit.utils._process import not_duplicate, find_zero_crossings
from typing import Tuple

FindExtremaOutput = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def find_extrema_parabol(time: np.ndarray, signal: np.ndarray) -> FindExtremaOutput:
    """
    Performs parabolic estimation of extremum in a one-dimensional signal array S.
    This method not only identifies the positions and values of local maxima and minima
    by fitting a parabola to three consecutive points (where the middle point is the vertex of the parabola),
    but also provides zero crossing points which are essential for signal analysis.

    :param time: Array representing the time or sequence indices.
    :param signal: One-dimensional array representing the signal values.
    :return: Positions and values of local maxima and minima, and positions of zero crossings.
    """

    # Detect zero crossings by comparing adjacent signal points
    indzer = find_zero_crossings(signal=signal)

    # Handling duplicate values which are crucial for accurate parabolic fitting
    idx = not_duplicate(signal)
    time = time[idx]
    signal = signal[idx]

    # Initialize variables for parabolic fitting
    Tp, T0, Tn = time[:-2], time[1:-1], time[2:]
    Sp, S0, Sn = signal[:-2], signal[1:-1], signal[2:]
    TnTp, T0Tn, TpT0 = Tn - Tp, T0 - Tn, Tp - T0
    scale = (
        Tp * Tn * Tn
        + Tp * Tp * T0
        + T0 * T0 * Tn
        - Tp * Tp * Tn
        - Tp * T0 * T0
        - T0 * Tn * Tn
    )

    # Compute coefficients of the parabola: y = ax^2 + bx + c
    a = T0Tn * Sp + TnTp * S0 + TpT0 * Sn
    b = (S0 - Sn) * Tp**2 + (Sn - Sp) * T0**2 + (Sp - S0) * Tn**2
    c = T0 * Tn * T0Tn * Sp + Tn * Tp * TnTp * S0 + Tp * T0 * TpT0 * Sn

    # Scale and adjust coefficients, and calculate vertex of parabola
    a = a / scale
    b = b / scale
    c = c / scale
    a[a == 0] = 1e-14
    tVertex = -0.5 * b / a
    idx = (tVertex < T0 + 0.5 * (Tn - T0)) & (tVertex >= T0 - 0.5 * (T0 - Tp))

    # Select valid vertices and classify as maxima or minima based on 'a'
    a, b, c = a[idx], b[idx], c[idx]
    tVertex = tVertex[idx]
    sVertex = a * tVertex * tVertex + b * tVertex + c
    local_max_pos, local_max_val = tVertex[a < 0], sVertex[a < 0]
    local_min_pos, local_min_val = tVertex[a > 0], sVertex[a > 0]

    return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer


def find_extrema_simple(time: np.ndarray, signal: np.ndarray) -> FindExtremaOutput:
    """
    Performs extrema detection in a given one-dimensional signal array S.
    This method identifies local maxima and minima by analyzing the first differences of the signal.
    The approach is straightforward and computationally efficient, making it suitable for signal processing applications
    where quick identification of critical turning points in the signal is required.

    :param time: Array representing the time or sequence indices.
    :param signal: One-dimensional array representing the signal values.
    :return: Positions and corresponding values of local maxima and minima, and the positions of zero crossings.
    """

    # Finds indexes of zero-crossings
    indzer = find_zero_crossings(signal=signal)

    # Finding local extrema points by using first differences
    d = np.diff(signal)

    # Constructing arrays of previous and next difference values
    # d1 * d2 < 0 checks for sign changes in differences (product being negative),
    # indicating a peak or trough between these difference points in signal S.
    d1, d2 = d[:-1], d[1:]
    indmin = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 < 0])[0] + 1
    indmax = np.nonzero(np.r_[d1 * d2 < 0] & np.r_[d1 > 0])[0] + 1

    # Handling cases when two or more points have the same value
    if np.any(d == 0):
        imax, imin = [], []

        bad = d == 0
        dd = np.diff(np.append(np.append(0, bad), 0))
        debs = np.nonzero(dd == 1)[0]
        fins = np.nonzero(dd == -1)[0]
        if debs[0] == 1:
            if len(debs) > 1:
                debs, fins = debs[1:], fins[1:]
            else:
                debs, fins = [], []

        if len(debs) > 0:
            if fins[-1] == len(signal) - 1:
                if len(debs) > 1:
                    debs, fins = debs[:-1], fins[:-1]
                else:
                    debs, fins = [], []

        lc = len(debs)
        if lc > 0:
            for k in range(lc):
                if d[debs[k] - 1] > 0:
                    if d[fins[k]] < 0:
                        imax.append(np.round((fins[k] + debs[k]) / 2.0))
                else:
                    if d[fins[k]] > 0:
                        imin.append(np.round((fins[k] + debs[k]) / 2.0))

        if len(imax) > 0:
            indmax = indmax.tolist()
            for x in imax:
                indmax.append(int(x))
            indmax.sort()

        if len(imin) > 0:
            indmin = indmin.tolist()
            for x in imin:
                indmin.append(int(x))
            indmin.sort()

    local_max_pos = time[indmax]
    local_max_val = signal[indmax]
    local_min_pos = time[indmin]
    local_min_val = signal[indmin]

    return local_max_pos, local_max_val, local_min_pos, local_min_val, indzer
