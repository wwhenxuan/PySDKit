# -*- coding: utf-8 -*-
"""
Computing the Sample Entropy of a given signal or time series of 1D numpy ndarray.
"""

import numpy as np
from numpy import bool_
from scipy.spatial.distance import pdist

from typing import Optional, Tuple, Any, Union, List

from pysdkit.entropy._coarse_grain import as_1d, embed, coarse_grain


def sample_entropy(
    y: np.ndarray, m: Optional[int], r: Optional[float], dist_type="chebychev"
) -> Union[np.ndarray, float]:
    """
    Computing the Sample Entropy of a given signal or time series of 1D numpy ndarray.

    The Sample Entropy (SampEn) is a measure used to quantify the complexity or regularity of a time series.
    It is particularly useful in the context of physiological time-series analysis and can help in identifying the unpredictability and chaotic behaviors in the system from which the time series originates.
    SampEn is related to the Approximate Entropy (ApEn) but differs in that it does not include self-matches (a match with the same point itself), thus providing a more robust measure for assessing the regularity or randomness of data.

    We have made some adjustments to the original result return.
    The original MATLAB code will return NaN value when A and B are both equal to 0.
    We have adjusted it to return -np.log(2 / ((N - m - 1) * (N - m))) when A and B are both 0

    Richman,J.S.,& Moorman,J.R.(2000). Physiological time-series analysis using approximate entropy and sample entropy.
    American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.

    MATLAB code: https://ww2.mathworks.cn/matlabcentral/fileexchange/69381-sample-entropy?s_tid=FX_rc3_behav

    :param y: The input 1D signal or time series with numpy ndarray vector with dims. [1xN]
    :param m: Embedding dimension (m < N) less than the length of the 1D inputs.
    :param r: Tolerance (percentage applied to the SD).
    :param dist_type: (Optional) Distance type, specified by a string.
                    Default value: 'chebychev' (type help pdist for further information).
    :return: SampEn value. Since SampEn is not defined whenever B = 0,
             the output value in that case is -np.log(2 / ((N - m - 1) * (N - m))).
    """
    # Error detection and defaults
    if len(y) < 2:
        raise ValueError("Signal must have at least two points.")
    if m >= len(y):
        raise ValueError(
            "Embedding dimension must be smaller than the signal length (m < N)."
        )
    if not isinstance(dist_type, str):
        raise ValueError("Distance must be a string.")

    # Convert signal to a 1D numpy array
    y = np.asarray(y).flatten()
    N = len(y)  # Length of the signal
    sigma = np.std(y)  # Standard deviation of the signal

    # Create the matrix of matches
    matches = np.full(shape=(m + 1, N), fill_value=np.nan)

    for i in range(m + 1):
        matches[i, : N - i] = y[i:]  # Create matching vectors (embeddings)

    matches = matches.T

    # Compute distances for m-dimensional and m+1-dimensional vectors
    d_m = pdist(matches[:, :m], metric=dist_type)

    # If no matches exist for m-dimensional vectors, SampEn is undefined (set to Inf)
    if len(d_m) == 0:
        # If B = 0, SampEn is not defined: no regularity detected
        # Note: Upper bound is returned
        return -np.log(2 / ((N - m - 1) * (N - m)))

    # Check the matches for m+1
    # Compute distances for (m+1)-dimensional and m+1-dimensional vectors
    d_m1 = pdist(matches[:, : m + 1], metric=dist_type)

    # Count A and B
    # Number of matches for m-dimensional vectors
    # Note: logical operations over NaN values are always 0
    B = np.sum(d_m <= r * sigma)
    # Number of matches for m+1-dimensional vectors
    A = np.sum(d_m1 <= r * sigma)

    # If A or B is zero, SampEn would be infinite, apply the upper bound correction
    if A == 0 or B == 0:
        # If A=0 or B=0, SampEn would return an infinite value.
        # However, the lowest non-zero conditional probability that SampEn should report is A/B = 2/[(N-m-1)(N-m)]
        return -np.log(2 / ((N - m - 1) * (N - m)))
    else:
        # Sample entropy computation
        # Note: norm. comes from [nchoosek(N-m+1,2)/nchoosek(N-m,2)]
        return -np.log((A / B) * ((N - m + 1) / (N - m - 1)))


def buffer(x: np.ndarray, tau: int) -> np.ndarray:
    """Refactoring the `buffer` function in MATLAB"""
    # Calculate the length that needs to be added so that it can divide tau
    pad_length = (tau - len(x) % tau) % tau
    # Fill with 0 or NaN (depending on the requirement)
    x_padded = np.pad(x, (0, pad_length), mode="constant", constant_values=0)
    # Split x into (n, tau) shapes
    return x_padded.reshape(-1, tau)


def multiscale_sample_entropy(
    x: np.ndarray,
    m: Optional[int] = 2,
    r: Optional[float] = 0.15,
    tau: Optional[int] = 1,
) -> Tuple[Union[float, Any], bool_, bool_]:
    """
    Multiscale Sample Entropy (MSE) computation.

    Based on "Multiscale entropy analysis of biological signals"
    By Madalena Costa, Ary L. Goldberger, and C.-K. Peng
    Published on 18 February 2005 in Phys. Rev. E 71, 021906.

    MATLAB code: https://ww2.mathworks.cn/matlabcentral/fileexchange/62706-multiscale-sample-entropy

    :param x: (array): Input time series (1D array).
    :param m: (int): Embedding dimension (default 2).
    :param r: (float): Tolerance (default 0.15).
    :param tau: (int): Time delay for coarse-graining (default 1).
    :return: - e (float): Multiscale Sample Entropy.
             - A (int): Number of matching (m+1)-element sequences.
             - B (int): Number of matching m-element sequences.
    """
    # Coarse-graining of the signal
    # y = np.mean(np.reshape(x, (-1, tau)), axis=1)
    # 这里重写了MATLAB的`buffer`函数
    # Fill with 0 value
    buffered_x = buffer(x, tau)
    y = np.nanmean(buffered_x, axis=1)

    # Create (m+1)-element sequences
    X = np.array([y[i : i + m + 1] for i in range(len(y) - m)])

    # Calculate A: Number of matching (m+1)-element sequences
    A = np.sum(pdist(X, metric="chebychev") < r * np.nanstd(x, ddof=1))

    # Create m-element sequences
    X = X[:, :m]  # Extract m-element sequences
    B = np.sum(pdist(X, metric="chebychev") < r * np.nanstd(x, ddof=1))

    # Calculate Multiscale Sample Entropy (MSE)
    if A == 0 or B == 0:
        e = np.nan  # If no matching sequences, return NaN
    else:
        e = np.log(B / A)

    return e, A, B


def _sample_entropy_counts(y: np.ndarray, m: int, r: float) -> Tuple[int, int]:
    """Pair counts (A: m+1, B: m) with Chebyshev radius ``r * std(y)``."""
    y = as_1d(y)
    n = len(y)
    if n < m + 2:
        return 0, 0
    sigma = np.std(y, ddof=1) if n > 1 else 0.0
    radius = r * sigma
    templates_m = embed(y, m)
    templates_m1 = embed(y, m + 1)
    if templates_m.shape[0] < 2:
        return 0, 0
    b_count = int(np.sum(pdist(templates_m, metric="chebyshev") <= radius))
    a_count = int(np.sum(pdist(templates_m1, metric="chebyshev") <= radius))
    return a_count, b_count


def composite_multiscale_sample_entropy(
    x: np.ndarray,
    m: Optional[int] = 2,
    r: Optional[float] = 0.15,
    scale: Optional[int] = 3,
) -> np.ndarray:
    """
    Composite multiscale sample entropy (cMSE, Wu et al. 2013).

    At scale ``s`` the ``s`` phase-shifted Costa grains are each scored with
    sample entropy and then averaged.

    Wu, S. D., Wu, C. W., Lin, S. G., Wang, C. C., & Lee, K. Y. (2013).
    Time series analysis using composite multiscale entropy. Entropy,
    15(3), 1069-1084.

    :param x: 1-D input series.
    :param m: Embedding dimension.
    :param r: Tolerance as a fraction of each grain's standard deviation.
    :param scale: Highest scale factor (returns length ``scale``).
    :return: cMSE at scales ``1 .. scale``.
    """
    x = as_1d(x)
    if scale < 1:
        raise ValueError("scale must be an integer >= 1.")
    values: List[float] = []
    for s in range(1, scale + 1):
        scores = []
        for offset in range(s):
            grain = coarse_grain(x, s, offset=offset)
            if len(grain) < m + 2:
                continue
            scores.append(float(sample_entropy(grain, m, r)))
        values.append(float(np.mean(scores)) if scores else float("nan"))
    return np.asarray(values, dtype=float)


def refined_composite_multiscale_sample_entropy(
    x: np.ndarray,
    m: Optional[int] = 2,
    r: Optional[float] = 0.15,
    scale: Optional[int] = 3,
) -> np.ndarray:
    """
    Refined composite multiscale sample entropy (rcMSE, Wu et al. 2014).

    Match counts ``A`` and ``B`` are pooled over the ``s`` phase-shifted
    grains, then ``-log(sum A / sum B)``.

    Wu, S. D., Wu, C. W., Lin, S. G., Lee, K. Y., & Peng, C. K. (2014).
    Analysis of complex time series using refined composite multiscale
    entropy. Physics Letters A, 378(20), 1369-1374.

    :param x: 1-D input series.
    :param m: Embedding dimension.
    :param r: Tolerance as a fraction of each grain's standard deviation.
    :param scale: Highest scale factor (returns length ``scale``).
    :return: rcMSE at scales ``1 .. scale``.
    """
    x = as_1d(x)
    if scale < 1:
        raise ValueError("scale must be an integer >= 1.")
    values: List[float] = []
    for s in range(1, scale + 1):
        a_sum = 0
        b_sum = 0
        for offset in range(s):
            grain = coarse_grain(x, s, offset=offset)
            a_count, b_count = _sample_entropy_counts(grain, m, r)
            a_sum += a_count
            b_sum += b_count
        if a_sum == 0 or b_sum == 0:
            values.append(float("nan"))
        else:
            values.append(float(-np.log(a_sum / b_sum)))
    return np.asarray(values, dtype=float)


if __name__ == "__main__":
    ########## Example Usage of Sample Entropy ##########
    signal = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    )  # Example random signal
    m = 1  # Embedding dimension
    r = 0.2  # Tolerance (20% of the signal's standard deviation)

    value = sample_entropy(signal, m, r)  # Compute the SampEn value
    print(f"Sample Entropy: {value}")

    ########## Example Usage of Multiscale Sample Entropy ##########
    x = np.random.rand(200)  # Example random time series
    m = 2  # Embedding dimension
    r = 0.15  # Tolerance
    tau = 1  # Time delay for coarse-graining

    e, A, B = multiscale_sample_entropy(
        x, m, r, tau
    )  # Compute Multiscale Sample Entropy
    print(f"Multiscale Sample Entropy: {e}")
    print(f"A (matches for m+1): {A}, B (matches for m): {B}")

    # The following test method comes from https://archive.physionet.org/physiotools/mse/tutorial/tutorial.pdf
    trials = 30
    scales = 20

    t = np.zeros([scales, trials])

    r = 0.15
    m = 2

    n = 3000

    for tau in range(1, scales + 1):
        for j in range(1, trials + 1):
            signal = np.random.randn(n)

            e, _, _ = multiscale_sample_entropy(signal, m, r, tau)
            t[tau - 1, j - 1] = e

    from matplotlib import pyplot as plt

    plt.plot(np.mean(t, axis=1))
    plt.show()
