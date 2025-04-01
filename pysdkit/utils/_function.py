# -*- coding: utf-8 -*-
"""
Created on 2024/6/3 15:31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

This py includes a series of functional modules
"""
import numpy as np
from scipy import linalg

from typing import Optional

__all__ = [
    "max_min_normalization",
    "z_score_normalization",
    "max_absolute_normalization",
    "log_transformation",
    "decimal_scaling_normalization",
    "is_1d",
    "is_complex",
    "to_1d",
    "to_2d",
    "lags_matrix",
    "covariance_matrix",
    "index_of_orthogonality",
]


def max_min_normalization(x: np.ndarray) -> np.ndarray:
    """
    Perform min-max normalization on the input signal

    :param x: Input 1D sequence
    :return: Normalized sequence
    """
    return (x - x.min()) / (x.max() - x.min())


def z_score_normalization(x: np.ndarray) -> np.ndarray:
    """
    Perform Z-score normalization on the input signal

    :param x: Input 1D sequence
    :return: Normalized sequence
    """
    return (x - x.mean()) / x.std()


def max_absolute_normalization(x: np.ndarray) -> np.ndarray:
    """
    Perform Max Absolute normalization on the input signal

    :param x: Input 1D sequence
    :return: Normalized sequence
    """
    return x / np.max(np.abs(x))


def log_transformation(x: np.ndarray) -> np.ndarray:
    """
    Perform log transformation on the input signal

    :param x: Input 1D sequence
    :return: Transformed sequence
    """
    return np.log1p(x)


def decimal_scaling_normalization(x: np.ndarray) -> np.ndarray:
    """
    Perform Decimal Scaling normalization on the input signal

    :param x: Input 1D sequence
    :return: Normalized sequence
    """
    max_abs_value = np.max(np.abs(x))
    scaling_factor = 10 ** np.ceil(np.log10(max_abs_value + 1))
    return x / scaling_factor


def is_1d(x: np.ndarray) -> bool:
    """Check if the input sequence is one-dimensional"""
    if np.ndim(x) != 1:
        return False
    else:
        return True


def is_complex(x: np.ndarray) -> bool:
    """Check if the input sequence is a complex sequence"""
    return isinstance(x, (complex, np.complex64, np.complex128))


def to_1d(data: np.ndarray) -> np.ndarray:
    """
    Transform any data to a 1D numpy ndarray

    :param data: None, float, int or ndarray of any data type
    :return: the transformed 1D numpy ndarray
    """
    # Ensure the data type is correct
    data = np.asarray(data)

    # If the data has no dimensions, add one
    if np.ndim(data) == 0:
        return np.asarray([data])

    # If the data is already one-dimensional, return it directly
    elif np.ndim(data) == 1:
        return data

    # Handle more complex multi-dimensional data
    else:
        data = np.squeeze(data)
        if np.ndim(data) == 1:
            return data
        else:
            return np.ravel(data)


def to_2d(data: np.ndarray, column: Optional[bool] = False) -> np.ndarray:
    """
    Transform any data to a 2D numpy ndarray

    :param data: None, float, int or ndarray of any data type
    :param column: Whether to output a row vector or a column vector.
                   Determines where the new dimension is added.
    :return: the transformed 2D numpy ndarray
    """
    # Ensure the data type is correct
    data = np.asarray(data)

    # First, convert the data to a 1D vector for easier processing
    data = to_1d(data)

    if len(data.shape) == 1:
        # Add a new dimension at a specific position
        if column:
            return data[np.newaxis, :]
        else:
            return data[:, np.newaxis]
    else:
        return data


def lags_matrix(
    x: np.ndarray,
    mode: Optional[str] = "full",
    lags: Optional[int] = None,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    This function generates the lag matrix of a signal, also known as the data matrix or correlation matrix

    This type of matrix is very common in signal processing, time series analysis, adaptive filter design,
    system identification, and other fields. It can generate various types of matrices based on different modes,
    such as Toeplitz matrices, Hankel matrices, convolution matrices, etc.

    :param x: 1D numpy ndarray signal
    :param mode: Specifies the type of lag matrix to be generated. The supported modes include:

            - mode = 'full' :
                lags_matrix is the full Toeplitz convolutional matrix with dimensions [lags+N-1,lags],

                math:: out = [ [x,0..0]^T,[0,x,0..0]^T,...,[0,..0,x]^T ], where: N is the size of x.
            - mode = 'prew':
                lags_matrix is the prewindowed matrix with the first N columns of the full matrix, and dimension = [N,lags];
            - mode = 'postw':
                lags_matrix is the postwindowed matrix with the last N columns of the full matrix, and dimension = [N,lags];
            - mode = 'covar' or 'valid':
                lags_matrix is the trimmed full matrix with the first and last m columns cut off

                (out = full[lags:N-lags,:]), with dimension = [N-lags+1,lags];
            - mode = 'same':
                conv_matrix is the trimmed full matrix with the first and last m columns cut off

                (out = full[(lags-1)//2:N+(lags-1)//2,:]), with dimension = [N,lags];
            - mode = 'traj':
                lags_matrix is the trajectory or so-called caterpillar matrix with dimension = [N,lags];
            - mode = 'hanekl':
                lags_matrix is the Hankel matrix with dimension = [N,N];
            - mode = 'toeplitz':
                lags_matrix is the symmetric Toeplitz matrix, with dimension = [N,N].

    :param lags: An integer or None, representing the number of columns in the lag matrix (default is N // 2, where N is the length of the input signal).
    :param dtype: The numpy data type used, `None` means using the data type of the input signal
    :return: A 2D array representing the generated lag matrix.

    The generation method for each mode is mainly based on the arrangement and combination of different lagged versions of the input signal.

    By choosing the appropriate mode, matrices suitable for different signal processing and time series analysis tasks can be generated.
    """
    # Ensure the data type is correct
    x = np.asarray(x)

    # Get the length of the 1D signal
    seq_len = x.shape[0]

    # Handle the `lags` parameter
    if lags is None:
        lags = seq_len // 2

    # Correct the data type used
    if dtype is None:
        dtype = x.dtype

    # Select the specific operation based on the input mode

    if mode in ["caterpillar", "traj", "trajectory"]:
        # Generate the trajectory matrix (also known as the caterpillar matrix)
        trajmat = linalg.hankel(x, np.zeros(lags)).T
        trajmat = trajmat[:, : (x.shape[0] - lags + 1)]
        matrix = np.conj(trajmat.T)

    elif mode == "toeplitz":
        # Generate the symmetric Toeplitz matrix, with dimensions [N, N]
        matrix = linalg.toeplitz(x)

    elif mode == "hankel":
        # Generate the Hankel matrix, with dimensions [N, N]
        matrix = linalg.hankel(x)

    elif mode in [
        "full",
        "prewindowed",
        "postwindowed",
        "prew",
        "postw",
        "covar",
        "same",
        "valid",
    ]:

        matrix = np.zeros(shape=(seq_len + lags - 1, lags), dtype=dtype)

        # full mode
        # Generate the full convolution matrix, with dimensions [lags + N - 1, lags]
        # Each column of the matrix corresponds to a different lagged version of the input signal
        for i in range(lags):
            matrix[i : i + seq_len, i] = x
        if mode == "prewindowed" or mode == "prew":
            # Generate the prewindowed matrix, taking the first N rows of the full matrix, with dimensions [N, lags]
            matrix = matrix[:seq_len, :]
        elif mode == "postwindowed" or mode == "postw":
            # Generate the postwindowed matrix, taking the last N rows of the full matrix, with dimensions [N, lags]
            matrix = matrix[lags - 1 :, :]
        elif mode == "same":
            # Generate the trimmed matrix, removing the first and last lags columns of the full matrix, with dimensions [N, lags]
            matrix = matrix[(lags - 1) // 2 : (lags - 1) // 2 + seq_len, :]
        elif mode == "valid" or mode == "covar":
            # Generate the trimmed matrix, removing the first and last lags columns of the full matrix, with dimensions [N - lags + 1, lags]
            matrix = matrix[lags - 1 : -lags + 1, :]

    else:
        raise ValueError(
            """mode have to be one of ['full','prewindowed','postwindowed', 'prew','postw','covar','valid','same', 'traj', 'caterpillar', 'trajectory', 'hankel', 'toeplitz'] """
        )

    return matrix


def covariance_matrix(
    x: np.ndarray,
    mode: Optional[str] = "full",
    lags: Optional[int] = None,
    ret_base: Optional[bool] = False,
    dtype: Optional[np.dtype] = None,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    This function calculates the covariance matrix of the input signal's lag matrix
    It generates a specific lag matrix based on the input signal x and the specified mode,
    and then calculates the covariance matrix of that matrix.

    The covariance matrix is very important in signal processing, time series analysis,
    statistical modeling, and other fields, as it can describe the correlation of the signal
    at different lags.

    :param x: the input signal of 1d ndarray
    :param mode: Specifies the type of lag matrix to be generated. The supported modes include:

           - mode = 'full':
                lags_matrix is the full Toeplitz convolutional matrix with dimensions [lags+N-1,lags],

                math:: out = [ [x,0..0]^T,[0,x,0..0]^T,...,[0,..0,x]^T ], where: N is the size of x.
           - mode =  'prew':
                lags_matrix is the prewindowed matrix with the first N columns of the full matrix, and dimension = [N,lags];
           - mode = 'postw':
                lags_matrix is the postwindowed matrix with the last N columns of the full matrix, and dimension = [N,lags];
           - mode = 'covar' or 'valid':
                lags_matrix is the trimmed full matrix with the first and last m columns cut off

                (out = full[lags:N-lags,:]), with dimension = [N-lags+1,lags];
           - mode = 'same':
                conv_matrix is the trimmed full matrix with the first and last m columns cut off

                (out = full[(lags-1)//2:N+(lags-1)//2,:]), with dimension = [N,lags];
           - mode = 'traj':
                lags_matrix is the trajectory or so-called caterpillar matrix with dimension = [N,lags];
           - mode = 'hanekl':
                lags_matrix is the Hankel matrix with dimension = [N,N];
           - mode = 'toeplitz':
                lags_matrix is the symmetric Toeplitz matrix, with dimension = [N,N].
    :param lags: An integer or None, representing the number of columns in the lag matrix (default is N // 2, where N is the length of the input signal).
    :param ret_base: if true, then the lag matrix will also be returned
    :param dtype: The numpy data type used, `None` means using the data type of the input signal
    :return: - ret_base is False:
                * matrix: 2d ndarray.
             - ret_base is True:
                * matrix: 2d ndarray, covariance matrix.
                * lags_matrix: lag matrix.

    :Note:
        Lag matrices of different modes have different shapes and uses, and the choice of mode depends on the specific application scenario.
        The calculation of the covariance matrix is based on the dot product of the lag matrix, so its result reflects the correlation of the input signal at different lags.
        If the input signal is short, the value of lags may need to be adjusted to avoid generating an overly large lag matrix.
    """
    # First generate the lag matrix
    mtx = lags_matrix(x, lags=lags, mode=mode, dtype=dtype)

    # Calculate the covariance matrix
    if ret_base:
        return np.dot(mtx.T, np.conj(mtx)), mtx
    else:
        return np.dot(mtx.T, np.conj(mtx))


def index_of_orthogonality(signal: np.ndarray, IMFs: np.ndarray) -> float:
    """
    Any pair of IMFs is locally orthogonal.
    To evaluate EMD performance, an Index of Orthogonality (IO) was proposed,
    so that the closer to zero, the more effective will be the decomposition.

    :param signal: the row input signal.
    :param IMFs: Intrinsic Mode Function after decomposition.
    :return: the value of index of orthogonality

    Index of Orthogonality is proposed py Under Barbosa de Souza in
    A survey on Hilbert-Huang transform: Evolution, challenges and solutions.
    Digital Signal Processing 120 (2022) 103292.
    """

    def cum(x, y):
        # Helper functions
        return np.sum(np.abs(x * y))

    # the value of IO
    index = 0.0

    # Start calculating the evaluation metrics
    K, T = IMFs.shape
    for i in range(K):
        for j in range(K):
            if i != j and i < j:
                index += cum(IMFs[i, :], IMFs[j, :])

    return index / np.sum(signal**2)
