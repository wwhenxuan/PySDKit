import numpy as np
from pysdkit.utils import lags_matrix
from pysdkit.utils import to_2d

from typing import Optional, Tuple

__all__ = ["kernel_matrix", "euclidian_matrix"]


def kernel_matrix(
    x: np.ndarray,
    mode: Optional[str] = "full",
    kernel: Optional[str] = "linear",
    kpar: int = 1,
    lags: Optional[int] = None,
    return_base: Optional[bool] = False,
    normalization: Optional[bool] = True,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    This function is used to generate the kernel matrix of the input signal

    It is similar to the calculation of the covariance matrix, but the kernel function is used to measure the similarity between data points.
    The kernel matrix is very important in fields such as machine learning, signal processing, and time series analysis,
    especially when dealing with nonlinear relationships.

    The kernel matrix calculates the similarity between different lagged versions of the input signal through the kernel function.
    The choice of kernel function and parameter settings can be adjusted according to the specific application scenario.

    :param x: 1D numpy ndarray signal
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

    :param kernel: kernel = {exp, rbf, polynomial, sigmoid, linear, euclid, minkowsky, thin_plate, bump, polymorph}
    :param kpar: kernel parameter, depends on the kernel type
    :param lags: number of lags (N//2 dy default of `None`)
    :param return_base: if true, than lags matrix will be also returned
    :param normalization: if True, than matrix mean will be substructed
    :return:
        - ret_base is False:
            * kernel matrix: 2d ndarray.
        -  ret_base is True:
            * matrix: 2d ndarray, kernel matrix.
            * lags_matrix: lags matrix.

    :Note:
        Selection of kernel function: Different kernel functions are suitable for different application scenarios.
        For example, the RBF kernel is suitable for processing Gaussian distributed data, while the polynomial kernel is suitable for data with polynomial relationships.
        Normalization: Normalization can reduce the scale difference of the kernel matrix, making it more suitable for subsequent analysis.
        Adjustment of the number of lags: If the input signal is short, you may need to adjust the value of lags to avoid generating an overly large lag matrix.
    """
    # Adjust the input data format to prevent errors
    x = np.asarray(x)

    # 处理滞后项
    if lags is None:
        lags = x.shape[0] // 2

    # Generate lag matrix
    base = lags_matrix(x, lags=lags, mode=mode)

    if return_base:
        out = base

    # if euclidian_matrix applied for base
    if kernel in [
        "rbf",
        "thin_plate",
        "euclid",
        "minkowsky",
        "bump",
        "polymorph",
        "exp",
        "laplacian",
        "laplace",
        "gauss",
    ]:
        base = base.T

    # Use the kernel method on the resulting lag matrix
    R = _kernel(base, base, ktype=kernel, kpar=kpar)

    if normalization:
        # 是否进行数据的标准化
        column_sums = np.mean(R, axis=0)
        total_sum = np.mean(column_sums)
        J = np.ones(R.shape[0]) * column_sums
        R = R - J - J.T + total_sum

    if return_base:
        out = (R, out)
    else:
        out = R

    return out


def euclidian_matrix(X, Y, inner=False, square=True, normalize=False):
    """
    Matrix of euclidian distance I.E. Pairwise distance matrix

    :param X: 2d or 1d input ndarray
    :param Y: 2d or 1d input ndarray
    :param inner: inner or outer dimesions
    :param square: if false, then sqrt will be taken
    :param normalize: if true, distance will be normalized as d = d/(std(x)*std(y))
    :return: 2d ndarray, pairwise distance matrix
    """
    # Check the input dimensions of your data
    X, Y = _check_dim(X, Y)
    out = _euclid(X, Y, inner=inner)
    if not square:
        out = np.sqrt(out)
    if normalize:
        out /= np.std(X) * np.std(Y)

    return out


def _kernel(a, b=None, ktype="rbf", kpar=1 / 2):
    """
    Compute the kernel matrix (Gram matrix) between matrices a and b.

    :param a: 2d or 1d input ndarray
    :param b: 2d or 1d input ndarray
    :param ktype: {exp, rbf, polynomial, sigmoid, linear, euclid, minkowsky, thin_plate, bump, polymorph}
    :param kpar: kernel parameter depends on the kernel type
    :return: the kernel matrix (Gram matrix) between matrices a and b
    """
    # Check the dimensions of the input matrices and ensure they are compatible
    a, b = _check_dim(a, b)

    # Initialize the kernel matrix with zeros
    k = np.zeros(shape=(a.shape[0], b.shape[0]), dtype=np.complex64)

    # Compute the kernel matrix based on the specified kernel type
    if ktype == "linear":
        # Linear kernel: K(x, y) = x @ y.T
        k = _linear(a, b)
    elif ktype == "euclid":
        # Euclidean distance kernel: K(x, y) = ||x - y||^2
        k = _euclid(a, b)
    elif ktype == "minkowsky":
        # Minkowsky distance kernel: K(x, y) = ||x - y||^p, where p = kpar / 2
        k = np.power(_euclid(a, b), kpar / 2)
    elif ktype == "sigmoid":
        # Sigmoid kernel: K(x, y) = tanh(x @ y.T + kpar)
        k = np.tanh(_linear(a, b) + kpar)
    elif ktype in ["rbf", "gauss"]:
        # Radial Basis Function (RBF) kernel: K(x, y) = exp(-kpar * ||x - y||^2)
        k = np.exp(-kpar * _euclid(a, b))
    elif ktype in ["exp", "laplacian", "laplace"]:
        # Exponential kernel: K(x, y) = exp(-kpar * ||x - y||)
        k = np.exp(-kpar * np.sqrt(_euclid(a, b)))
    elif ktype in ["poly", "polynom", "polynomial"]:
        # Polynomial kernel: K(x, y) = (1 + x @ y.T)^kpar
        k = np.power(1 + _linear(a, b), kpar)
    elif ktype == "thin_plate":
        # Thin plate spline kernel: K(x, y) = ||x - y|| * log(||x - y||) / 2
        k = _euclid(a, b) * np.log(_euclid(a, b)) / 2
    elif ktype == "bump":
        # Bump kernel: K(x, y) = exp(-1 / (1 - kpar * ||x - y||))
        k = _euclid(a, b)
        k = np.exp(-1 / (1 - kpar * k))
    elif ktype == "polymorph":
        # Polymorph kernel: K(x, y) = ||x - y||^(kpar / 2) * log(||x - y||) / 2
        k = np.power(_euclid(a, b), kpar / 2)
        k = np.log(_euclid(a, b)) / 2
        k *= np.power(_euclid(a, b), (kpar - 1) / 2)
    elif ktype in ["rbf_inner"]:
        # RBF kernel with inner product: K(x, y) = exp(-kpar * ||x - y||^2) using inner product
        k = np.exp(-kpar * _euclid(a, b, True))
    else:
        # Raise an error if the kernel type is not supported
        raise NotImplementedError("use one of the kernel from help")

    # Return the computed kernel matrix
    return k


# ------------------------------------------------------------
def _check_dim(
    X: np.ndarray, Y: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Check whether the dimensions of the input data are compliant, and make adjustments if they are not"""
    X = np.asarray(X)

    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)

    if X.ndim == 1 and Y.ndim == 1:
        # Add a new dimension to one-dimensional data
        X = to_2d(X, column=False)
        Y = to_2d(Y, column=False)

    elif X.ndim == 1:
        X = to_2d(X, column=True)

    elif Y.ndim == 1:
        Y = to_2d(Y, column=True)

    return X, Y


def _linear(
    X: np.ndarray, Y: np.ndarray, open_dot: Optional[bool] = False
) -> np.ndarray:
    """
    Compute the linear kernel (inner product) between matrices X and Y.

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/metrics/pairwise.py#L980
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/utils/extmath.py#L120

    :param X: Input matrix of shape (n_samples_X, n_features)
    :param Y: Input matrix of shape (n_samples_Y, n_features), default=None
    :param open_dot: Boolean, specifies the direction of the dot product
    :return: Linear kernel matrix of shape (n_samples_X, n_samples_Y)
    """

    # Check if the dot product should be computed in the "open" direction (X @ Y.T)
    if open_dot:
        # Compute the dot product of X and the conjugate transpose of Y
        ret = np.dot(X, np.conj(Y.T))

    else:
        # Compute the dot product of the transpose of X and the conjugate of Y
        ret = np.dot(X.T, np.conj(Y))

    # Return the computed linear kernel matrix
    return ret


def _euclid(X: np.ndarray, Y: np.ndarray, inner: Optional[bool] = False) -> np.ndarray:
    """
    Compute the Euclidean distance matrix between each pair of rows from matrices X and Y.

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/utils/extmath.py#L50
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/utils/extmath.py#L120
    https://github.com/scikit-learn/scikit-learn/blob/fd237278e/sklearn/metrics/pairwise.py#L200

    :param X: Input matrix of shape (n_samples_X, n_features)
    :param Y: Input matrix of shape (n_samples_Y, n_features), default=None
    :param inner: Boolean, if True, compute the inner product instead of Euclidean distance
    :return: distance matrix of shape (n_samples_X, n_samples_Y)
    """
    axis = 1
    open_dot = True

    if inner:
        # 是否计算内积
        axis = 0
        open_dot = False

    XX = np.sum(np.square(X), axis=axis)[:, np.newaxis]
    YY = np.sum(np.square(Y), axis=axis)[np.newaxis, :]

    distances = -2 * _linear(X, Y, open_dot)

    distances += XX

    distances += YY

    return np.abs(distances)
