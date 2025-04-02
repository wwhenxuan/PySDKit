# -*- coding: utf-8 -*-
"""
Created on 2024/5/18 22:15
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np


def fmirror(ts: np.ndarray, sym: int) -> np.ndarray:
    """
    Implements a signal mirroring expansion function.

    This function mirrors 'sym' elements at both the beginning and the end of the given array 'ts',
    to create a new extended array.

    :param ts: The one-dimensional numpy array to be mirrored.
    :param sym: The number of elements to mirror from both the start and the end of the array 'ts'.
                This value must be less than or equal to half the length of the array.
    :return: The array after mirror expansion, which will have a length equal to the original
              array length plus twice the 'sym'.

    :examples:

    >>> array = np.array([1, 2, 3, 4, 5])
    >>> fmirror(array, 2)

        array([2, 1, 1, 2, 3, 4, 5, 5, 4])

    :Note:

    If 'sym' exceeds half the length of the array,
    the function may not work as expected, so it's recommended to check the value of 'sym' beforehand.
    """
    fMirr = np.append(np.flip(ts[:sym], axis=0), ts)
    fMirr = np.append(fMirr, np.flip(ts[-sym:], axis=0))
    return fMirr


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    sym = len(x) // 2
    print(sym)
    x_mirror = fmirror(x, sym)
    print(x_mirror)

    print(x_mirror[sym:-sym])
