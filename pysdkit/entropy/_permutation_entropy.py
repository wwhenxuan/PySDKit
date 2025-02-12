# -*- coding: utf-8 -*-
"""
Created on 2025/02/12 11:38:53
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
import itertools

from typing import Optional, Tuple


def permutation_entropy(
    y: np.ndarray, m: Optional[int] = 2, t: Optional[int] = 1
) -> Tuple[np.ndarray, np.ndarray] | Tuple[float, float]:
    """
    Calculate the permutation entropy of the input time series of signal.

    Permutation entropy is a method to measure the complexity of time series, which is often used to analyze nonlinear and chaotic systems.
    The core idea of permutation entropy is to measure the degree of chaos of time series through the permutation pattern (i.e. the order of data points).

    [1] G Ouyang, J Li, X Liu, X Li, Dynamic Characteristics of Absence EEG Recordings with Multiscale Permutation Entropy Analysis,
    Epilepsy Research, doi: 10.1016/j.eplepsyres.2012.11.003.

    [2] X Li, G Ouyang, D Richards, Predictability analysis of absence seizures with permutation entropy, Epilepsy Research,  Vol. 77pp. 70-74, 2007.

    MATLAB code: https://ww2.mathworks.cn/matlabcentral/fileexchange/37289-permutation-entropy?s_tid=FX_rc1_behav
    :param y: The input 1D time series or signal of numpy ndarray.
    :param m: The order of permutation entropy.
    :param t: The delay time of permutation entropy.
    :return: The permutation entropy of input and the histogram for the order distribution.
    """

    # Get the length of the time series
    ly = len(y)

    # Generate all possible permutations
    permlist = list(itertools.permutations(range(m)))

    # Initialize count array
    c = np.zeros(len(permlist))

    # Iterate over the time series to extract the permutation pattern for each time window
    for j in range(ly - t * (m - 1)):
        # Extract and sort data in time windows
        iv = np.argsort(y[j : j + t * m : t])  # Get the sorted index
        # Compare all alignment modes
        for jj, perm in enumerate(permlist):
            if np.array_equal(np.array(perm), iv):
                c[jj] += 1

    # Compute the probability distribution of permutation patterns
    hist = c[c != 0]
    p = hist / np.sum(hist)

    # Calculating permutation entropy
    pe = -np.sum(p * np.log(p))

    return pe, hist


if __name__ == "__main__":

    ########## Permutation Entropy Test Examples ##########
    y = np.array([1, 3, 2, 5, 4])
    m = 3
    t = 1

    pe, hist = permutation_entropy(y, m, t)
    print("Permutation Entropy:", pe)
    print("Histogram of Permutation Orders:", hist)
