# -*- coding: utf-8 -*-
"""
Created on Sat Mar 5 21:57:53 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

The following code is mainly used to find extreme points in the EMD algorithm

Code taken from https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EMD.py
"""
import numpy as np
from typing import Tuple, Optional


def get_timeline(range_max: int, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Generates a numeric sequence representing a timeline for a signal.
    This sequence can be specified with a data type to ensure adequate representation of the data range.

    :param range_max: The largest value in the range, equivalent to `range(range_max)`, typically representing the length of the signal.
    :param dtype: The minimum definition type. The returned timeline will have a dtype that is the same or with a higher byte size.
    :return: The timeline array.
    """

    timeline = np.arange(0, range_max, dtype=dtype)
    # Ensure the timeline includes the maximum value accurately
    if timeline[-1] != range_max - 1:
        inclusive_dtype = smallest_inclusive_dtype(timeline.dtype, range_max)
        timeline = np.arange(0, range_max, dtype=inclusive_dtype)
    return timeline


def smallest_inclusive_dtype(ref_dtype: np.dtype, ref_value) -> np.dtype:
    """
    Determines the smallest numpy dtype that can include a specified reference value,
    maintaining the base type (integer or float) of the reference dtype.

    :ValueError: If the requested range exceeds the maximum limits of available numpy data types.

    :param ref_dtype: The reference dtype, used to select the base type (i.e., int or float) for the returned type.
    :param ref_value: A value which needs to be included in the returned dtype's range.
    :return: The appropriate dtype that includes the reference value.
    """

    # Determine appropriate dtype based on integer or float base type
    if np.issubdtype(ref_dtype, np.integer):
        for dtype in [np.uint16, np.uint32, np.uint64]:
            if ref_value < np.iinfo(dtype).max:
                return dtype
        max_val = np.iinfo(np.uint64).max
        raise ValueError(
            f"Requested too large integer range. Exceeds max(uint64) == {max_val}."
        )

    elif np.issubdtype(ref_dtype, np.floating):
        for dtype in [np.float16, np.float32, np.float64]:
            if ref_value < np.finfo(dtype).max:
                return dtype
        max_val = np.finfo(np.float64).max
        raise ValueError(
            f"Requested too large float range. Exceeds max(float64) == {max_val}."
        )

    else:
        raise ValueError(
            f"Unsupported dtype '{ref_dtype}'. Only intX and floatX are supported."
        )


def normalize_signal(t: np.ndarray) -> np.ndarray:
    """
    Normalize time array so that it doesn't explode on tiny values.

    Returned array starts with 0 and the smallest increase is by 1.

    :param t: Input 1D Signal - Numpy Array
    :return: Output 1D Signal after normalize - Numpy Array
    """
    d = np.diff(t)
    assert np.all(d != 0), "All time domain values needs to be unique"
    # ensure that the minimum time step after normalization is 1
    return (t - t[0]) / np.min(d)


def common_dtype(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Casts inputs (x, y) into a common numpy DTYPE.

    :param x: Input 1D Signal 1 - Numpy Array
    :param y: Input 1D Signal 2 - Numpy Array
    :return: Output two array with same common dtype - Numpy Array
    """
    # 获取两者的公共数据类型 get the common data type of both
    dtype = np.result_type(x.dtype, y.dtype)
    # 使两者的类型保持一致 make the two types consistent
    if x.dtype != dtype:
        x = x.astype(dtype)
    if y.dtype != dtype:
        y = y.astype(dtype)
    return x, y


def not_duplicate(ts: np.ndarray) -> np.ndarray:
    """
    Returns indices for not repeating values, where there is no extremum.

    This feature is particularly important for extreme value detection and data simplification in signal processing,
    and can help avoid double calculations of consecutive repeated values in extreme value detection and other analyses.
    For example, when determining which points should be used to calculate the envelope in the EMD algorithm,
    continuously repeated data points can be excluded, thereby improving calculation efficiency and accuracy.

    :param ts: Input 1D Signal 1 - Numpy Array
    :return: Index of distinct values in array
    """
    # mark duplicate values
    same = np.r_[ts[1:-1] == ts[0:-2]] & np.r_[ts[1:-1] == ts[2:]]
    # calculate the index of distinct values
    not_same_idx = np.arange(1, len(ts) - 1)[~same]
    # build the complete index array
    idx = np.empty(len(not_same_idx) + 2, dtype=np.int64)
    idx[0] = 0
    idx[-1] = len(ts) - 1
    idx[1:-1] = not_same_idx
    return idx


def find_zero_crossings(signal: np.ndarray) -> np.ndarray:
    """
    Detects zero crossings in a given signal. A zero crossing occurs when two consecutive signal points have opposite signs,
    indicating a transition from positive to negative values or vice versa. This function also considers signal points
    that are exactly zero as zero crossings.

    :param signal: The input signal as a NumPy array.
    :return: An array of indices where zero crossings occur.
    """
    # Finds indexes of zero-crossings based on sign changes between consecutive elements
    S1, S2 = signal[:-1], signal[1:]
    indzer = np.nonzero(S1 * S2 < 0)[0]
    # print(indzer)

    # Detect exact zeros in the signal as zero crossings
    if np.any(signal == 0):
        indz = np.nonzero(signal == 0)[0]

        # If multiple consecutive zeros exist, identify the start and end points of these flat (zero) regions
        if np.any(np.diff(indz) == 1):
            zer = signal == 0
            dz = np.diff(np.append(np.append(0, zer), 0))
            debz = np.nonzero(dz == 1)[0]
            finz = np.nonzero(dz == -1)[0] - 1
            indz = np.round((debz + finz) / 2.0)

        # Combine and sort the indices of zero crossings from sign changes and exact zeros
        indzer = np.sort(np.append(indzer, indz))

    return indzer
