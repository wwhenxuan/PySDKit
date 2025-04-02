# -*- coding: utf-8 -*-
"""
Created on Sat Mar 8 22:53:11 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
The following code is mainly used to Performs extrapolation on edges by adding extra extrema in the EMD algorithm

Code taken from https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EMD.py
"""
import numpy as np
from typing import Tuple, Optional


def prepare_points_parabol(
    T: np.ndarray,
    S: np.ndarray,
    max_pos: np.ndarray,
    max_val: np.ndarray,
    min_pos: np.ndarray,
    min_val: np.ndarray,
    nbsym: int,
    DTYPE=np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs mirroring on signal which extrema do not necessarily

    belong on the position array.

    max_pos, max_val: the position and corresponding value of the local maximum.
    min_pos, min_val: the position and corresponding value of the local minimum.

    Used to perform a mirror operation at the extreme points of the signal so that smoother and more complete boundaries can be obtained when constructing the upper and lower envelopes of the signal.

    This processing is especially important at the beginning and end edges of the signal, because these areas may lack enough data points to accurately estimate the extreme values.

    The mirror operation helps provide enough data points to make interpolation (such as spline interpolation) more accurate and natural.
    """

    # Need at least two extrema to perform mirroring
    max_extrema = np.zeros((2, len(max_pos)), dtype=DTYPE)
    min_extrema = np.zeros((2, len(min_pos)), dtype=DTYPE)

    max_extrema[0], min_extrema[0] = max_pos, min_pos
    max_extrema[1], min_extrema[1] = max_val, min_val

    # Local variables
    end_min, end_max = len(min_pos), len(max_pos)

    # Left bound
    d_pos = max_pos[0] - min_pos[0]
    left_ext_max_type = d_pos < 0  # True -> max, else min

    # Left extremum is maximum
    if left_ext_max_type:
        if (S[0] > min_val[0]) and (np.abs(d_pos) > (max_pos[0] - T[0])):
            # mirror signal to first extrema
            expand_left_max_pos = 2 * max_pos[0] - max_pos[1 : nbsym + 1]
            expand_left_min_pos = 2 * max_pos[0] - min_pos[0:nbsym]
            expand_left_max_val = max_val[1 : nbsym + 1]
            expand_left_min_val = min_val[0:nbsym]
        else:
            # mirror signal to beginning
            expand_left_max_pos = 2 * T[0] - max_pos[0:nbsym]
            expand_left_min_pos = 2 * T[0] - np.append(T[0], min_pos[0 : nbsym - 1])
            expand_left_max_val = max_val[0:nbsym]
            expand_left_min_val = np.append(S[0], min_val[0 : nbsym - 1])

    # Left extremum is minimum
    else:
        if (S[0] < max_val[0]) and (np.abs(d_pos) > (min_pos[0] - T[0])):
            # mirror signal to first extrema
            expand_left_max_pos = 2 * min_pos[0] - max_pos[0:nbsym]
            expand_left_min_pos = 2 * min_pos[0] - min_pos[1 : nbsym + 1]
            expand_left_max_val = max_val[0:nbsym]
            expand_left_min_val = min_val[1 : nbsym + 1]
        else:
            # mirror signal to beginning
            expand_left_max_pos = 2 * T[0] - np.append(T[0], max_pos[0 : nbsym - 1])
            expand_left_min_pos = 2 * T[0] - min_pos[0:nbsym]
            expand_left_max_val = np.append(S[0], max_val[0 : nbsym - 1])
            expand_left_min_val = min_val[0:nbsym]

    if not expand_left_min_pos.shape:
        expand_left_min_pos, expand_left_min_val = min_pos, min_val
    if not expand_left_max_pos.shape:
        expand_left_max_pos, expand_left_max_val = max_pos, max_val

    expand_left_min = np.vstack((expand_left_min_pos[::-1], expand_left_min_val[::-1]))
    expand_left_max = np.vstack((expand_left_max_pos[::-1], expand_left_max_val[::-1]))

    # Right bound
    d_pos = max_pos[-1] - min_pos[-1]
    right_ext_max_type = d_pos > 0

    # Right extremum is maximum
    if not right_ext_max_type:
        if (S[-1] < max_val[-1]) and (np.abs(d_pos) > (T[-1] - min_pos[-1])):
            # mirror signal to last extrema
            idx_max = max(0, end_max - nbsym)
            idx_min = max(0, end_min - nbsym - 1)
            expand_right_max_pos = 2 * min_pos[-1] - max_pos[idx_max:]
            expand_right_min_pos = 2 * min_pos[-1] - min_pos[idx_min:-1]
            expand_right_max_val = max_val[idx_max:]
            expand_right_min_val = min_val[idx_min:-1]
        else:
            # mirror signal to end
            idx_max = max(0, end_max - nbsym + 1)
            idx_min = max(0, end_min - nbsym)
            expand_right_max_pos = 2 * T[-1] - np.append(max_pos[idx_max:], T[-1])
            expand_right_min_pos = 2 * T[-1] - min_pos[idx_min:]
            expand_right_max_val = np.append(max_val[idx_max:], S[-1])
            expand_right_min_val = min_val[idx_min:]

    # Right extremum is minimum
    else:
        if (
            (S[-1] > min_val[-1])
            and len(max_pos) > 1
            and (np.abs(d_pos) > (T[-1] - max_pos[-1]))
        ):
            # mirror signal to last extremum
            idx_max = max(0, end_max - nbsym - 1)
            idx_min = max(0, end_min - nbsym)
            expand_right_max_pos = 2 * max_pos[-1] - max_pos[idx_max:-1]
            expand_right_min_pos = 2 * max_pos[-1] - min_pos[idx_min:]
            expand_right_max_val = max_val[idx_max:-1]
            expand_right_min_val = min_val[idx_min:]
        else:
            # mirror signal to end
            idx_max = max(0, end_max - nbsym)
            idx_min = max(0, end_min - nbsym + 1)
            expand_right_max_pos = 2 * T[-1] - max_pos[idx_max:]
            expand_right_min_pos = 2 * T[-1] - np.append(min_pos[idx_min:], T[-1])
            expand_right_max_val = max_val[idx_max:]
            expand_right_min_val = np.append(min_val[idx_min:], S[-1])

    if not expand_right_min_pos.shape:
        expand_right_min_pos, expand_right_min_val = min_pos, min_val
    if not expand_right_max_pos.shape:
        expand_right_max_pos, expand_right_max_val = max_pos, max_val

    expand_right_min = np.vstack(
        (expand_right_min_pos[::-1], expand_right_min_val[::-1])
    )
    expand_right_max = np.vstack(
        (expand_right_max_pos[::-1], expand_right_max_val[::-1])
    )

    max_extrema = np.hstack((expand_left_max, max_extrema, expand_right_max))
    min_extrema = np.hstack((expand_left_min, min_extrema, expand_right_min))

    return max_extrema, min_extrema


def prepare_points_simple(
    T: np.ndarray,
    S: np.ndarray,
    max_pos: np.ndarray,
    max_val: Optional[np.ndarray],
    min_pos: np.ndarray,
    min_val: Optional[np.ndarray],
    nbsym: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs mirroring on signal which extrema can be indexed on the position array"""

    # Find indexes of pass
    ind_min = min_pos.astype(int)
    ind_max = max_pos.astype(int)

    # Local variables
    end_min, end_max = len(min_pos), len(max_pos)

    # Left bound - mirror nbsym points to the left
    if ind_max[0] < ind_min[0]:
        if S[0] > S[ind_min[0]]:
            lmax = ind_max[1 : min(end_max, nbsym + 1)][::-1]
            lmin = ind_min[0 : min(end_min, nbsym + 0)][::-1]
            lsym = ind_max[0]
        else:
            lmax = ind_max[0 : min(end_max, nbsym)][::-1]
            lmin = np.append(ind_min[0 : min(end_min, nbsym - 1)][::-1], 0)
            lsym = 0
    else:
        if S[0] < S[ind_max[0]]:
            lmax = ind_max[0 : min(end_max, nbsym + 0)][::-1]
            lmin = ind_min[1 : min(end_min, nbsym + 1)][::-1]
            lsym = ind_min[0]
        else:
            lmax = np.append(ind_max[0 : min(end_max, nbsym - 1)][::-1], 0)
            lmin = ind_min[0 : min(end_min, nbsym)][::-1]
            lsym = 0

    # Right bound - mirror nbsym points to the right
    if ind_max[-1] < ind_min[-1]:
        if S[-1] < S[ind_max[-1]]:
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
            rmin = ind_min[max(end_min - nbsym - 1, 0) : -1][::-1]
            rsym = ind_min[-1]
        else:
            rmax = np.append(ind_max[max(end_max - nbsym + 1, 0) :], len(S) - 1)[::-1]
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
            rsym = len(S) - 1
    else:
        if S[-1] > S[ind_min[-1]]:
            rmax = ind_max[max(end_max - nbsym - 1, 0) : -1][::-1]
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]
            rsym = ind_max[-1]
        else:
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
            rmin = np.append(ind_min[max(end_min - nbsym + 1, 0) :], len(S) - 1)[::-1]
            rsym = len(S) - 1

    # In case any array missing
    if not lmin.size:
        lmin = ind_min
    if not rmin.size:
        rmin = ind_min
    if not lmax.size:
        lmax = ind_max
    if not rmax.size:
        rmax = ind_max

    # Mirror points
    tlmin = 2 * T[lsym] - T[lmin]
    tlmax = 2 * T[lsym] - T[lmax]
    trmin = 2 * T[rsym] - T[rmin]
    trmax = 2 * T[rsym] - T[rmax]

    # If mirrored points are not outside passed time range.
    if tlmin[0] > T[0] or tlmax[0] > T[0]:
        if lsym == ind_max[0]:
            lmax = ind_max[0 : min(end_max, nbsym)][::-1]
        else:
            lmin = ind_min[0 : min(end_min, nbsym)][::-1]

        if lsym == 0:
            raise Exception("Left edge BUG")

        lsym = 0
        tlmin = 2 * T[lsym] - T[lmin]
        tlmax = 2 * T[lsym] - T[lmax]

    if trmin[-1] < T[-1] or trmax[-1] < T[-1]:
        if rsym == ind_max[-1]:
            rmax = ind_max[max(end_max - nbsym, 0) :][::-1]
        else:
            rmin = ind_min[max(end_min - nbsym, 0) :][::-1]

        if rsym == len(S) - 1:
            raise Exception("Right edge BUG")

        rsym = len(S) - 1
        trmin = 2 * T[rsym] - T[rmin]
        trmax = 2 * T[rsym] - T[rmax]

    zlmax = S[lmax]
    zlmin = S[lmin]
    zrmax = S[rmax]
    zrmin = S[rmin]

    tmin = np.append(tlmin, np.append(T[ind_min], trmin))
    tmax = np.append(tlmax, np.append(T[ind_max], trmax))
    zmin = np.append(zlmin, np.append(S[ind_min], zrmin))
    zmax = np.append(zlmax, np.append(S[ind_max], zrmax))

    max_extrema = np.array([tmax, zmax])
    min_extrema = np.array([tmin, zmin])

    # Make double sure, that each extremum is significant
    max_dup_idx = np.where(max_extrema[0, 1:] == max_extrema[0, :-1])
    max_extrema = np.delete(max_extrema, max_dup_idx, axis=1)
    min_dup_idx = np.where(min_extrema[0, 1:] == min_extrema[0, :-1])
    min_extrema = np.delete(min_extrema, min_dup_idx, axis=1)

    return max_extrema, min_extrema
