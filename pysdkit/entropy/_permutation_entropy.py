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

    Permutation entropy is essentially concerned with the complexity of local temporal structures (i.e., short-term dynamics).
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


def multi(Data: np.ndarray, S: int) -> np.ndarray:
    """
    Generate a coarse-grained version of the time series.

    :param Data: numpy array of time series data.
    :param S: scale factor.
    :return: the coarse-grained time series at scale S
    """
    # Get the length of the time series
    L = len(Data)
    # Determine how many segments the data will be divided into
    J = L // S

    # Coarse-grain the time series by taking the mean of each segment
    M_Data = np.array([np.mean(Data[i * S : (i + 1) * S]) for i in range(J)])

    # Return the coarse-grained time series
    return M_Data


def multiscale_permutation_entropy(
    X: np.ndarray, m: Optional[int], t: Optional[int], scale: Optional[int]
) -> np.ndarray:
    """
    Calculate the Multiscale Permutation Entropy (MPE) of input time series or signal.

    Multiscale Permutation Entropy (MPE) is an extended method based on Permutation Entropy (PE).
    It captures the multi-scale characteristics of signals by calculating the permutation entropy of time series at different time scales,
    thereby providing a more comprehensive description of the complexity of the system.

    In traditional permutation entropy, the complexity of the original time series is calculated.
    Multiscale permutation entropy can capture the complexity of the signal at different time scales by calculating it at multiple different time scales.

    Multiscale permutation entropy (MPE) can provide the complexity characteristics of time series at different scales by calculating permutation entropy at multiple scales.
    It has a wide range of applications, especially in biomedicine, complex system analysis,
    financial markets and other fields, and can reveal the multi-level nonlinear dynamic characteristics of the system.

    [1] G Ouyang, J Li, X Liu, X Li, Dynamic Characteristics of Absence EEG Recordings with Multiscale Permutation Entropy Analysis,
    Epilepsy Research, doi: 10.1016/j.eplepsyres.2012.11.003.

    [2] X Li, G Ouyang, D Richards, Predictability analysis of absence seizures with permutation entropy, Epilepsy Research,  Vol. 77pp. 70-74, 2007.

    MATLAB code: https://ww2.mathworks.cn/matlabcentral/fileexchange/37288-multiscale-permutation-entropy-mpe?s_tid=FX_rc2_behav
    
    :param X: The input 1D time series or signal of numpy ndarray.
    :param m: The order of permutation entropy (pattern length).
    :param t: The delay time of permutation entropy.
    :param scale: the scale factor.
    :return: MPE: multiscale permuation entropy
    """

    # Initialize an empty list to store the MPE values for each scale
    MPE = []

    # Iterate through each scale from 1 to the given Scale factor
    for j in range(1, scale + 1):
        # Coarse-grain the time series at scale j
        Xs = multi(X, j)
        # Calculate the permutation entropy of the coarse-grained time series
        PE = permutation_entropy(y=Xs, m=m, t=t)[0]
        # Append the computed PE value to the MPE list
        MPE.append(PE)

    # Return the list of multiscale permutation entropies
    return np.abs(np.array(MPE))


if __name__ == "__main__":

    ########## Permutation Entropy Test Examples ##########
    y = np.array([1, 3, 2, 5, 4])
    m = 3
    t = 1

    pe, hist = permutation_entropy(y, m, t)
    print("Permutation Entropy:", pe)
    print("Histogram of Permutation Orders:", hist)

    ########## Multiscale Permutation Entropy Test Examples ##########
    X = np.array([1, 3, 2, 5, 4, 6, 7, 8, 9])  # Example time series
    m = 2  # Set the permutation entropy order (length of patterns)
    t = 1  # Set the delay time
    Scale = 2  # Set the scale factor (the number of scales to compute)

    # Calculate the Multiscale Permutation Entropy
    MPE = multiscale_permutation_entropy(X, m, t, Scale)
    print("Multiscale Permutation Entropy:", MPE)
