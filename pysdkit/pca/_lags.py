# -*- coding: utf-8 -*-
"""
Created on 2025/02/06 10:38:11
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np
import scipy

__all__ = ["lags_matrix", "covariance_matrix", "conv_matrix"]


# -------------------------------------------------------------------
def lags_matrix(x, mode="full", lags=None):
    """
    Data lags matrix.

    Parameters
    -------------
    * x: input 1d ndarray.
    * mode: string,
       mode = {'full',prew', 'postw', 'covar','traj', 'hanekl';'toeplitz'}
       > mode = 'full': lags_matrix is the full toeplitz convolutional matrix
           with dimentions [lags+N-1,lags],
            ..math::
            out = [ [x,0..0]^T,[0,x,0..0]^T,...,[0,..0,x]^T ],
            where: N is the x size.
       > mode =  'prew': lags_matrix is the prewindowed matrix with
            first N columns of full matrix, and dimention = [N,lags];
       > mode = 'postw': lags_matrix is the postwindowed matrix with
            last N columns of full matrix, and dimention = [N,lags];
       > mode = 'covar' or 'valid': lags_matrix is the trimmed full matrix with
            cut first and last m coluns (out = full[lags:N-lags,:]),
            with dimention = [N-lags+1,lags];
       > mode = 'same': conv_matrix is the trimmed full matrix with
            cut first and last m coluns (out = full[(lags-1)//2:N+(lags-1)//2,:]),
            with dimention = [N,lags];
       > mode = 'traj': lags_matrix is trajectory
           or so-called caterpillar matrix
            with dimention = [N,lags];
       > mode = 'hanekl': lags_matrix is the Hankel matrix
            with dimention = [N,N];
       > mode = 'toeplitz': lags_matrix is the symmetric Toeplitz matrix,
            with dimention = [N,N].
    * lags: int or None,
        number of lags (columns) in the output matrx (N//2 dy default).

    Returns
    -----------
    * lags_matrix: 2d ndarray.

    Notes
    ---------
    * The matrix is corresponded to the so-called corrmtx
       (see Marple S.L. DSA and etc),
       or convolutional_matrix (in Hayes);
       full mode corresponds to the autocorrelation mode
       in some notations.
    * In some implementation, see, for instance
       https://pypi.org/project/spectrum/
       M will have full dimention [m+N, m+1].
     * there are permissioned synonyms:
         'prew' and 'prewindowed';
         'postw' and 'postwindowed';
         'traj', 'caterpillar' and 'trajectory'.

    """

    x = np.asarray(x)

    N = x.shape[0]

    if lags == None:
        lags = N // 2

    if mode in ["caterpillar", "traj", "trajectory"]:
        trajmat = scipy.linalg.hankel(x, np.zeros(lags)).T
        trajmat = trajmat[:, : (x.shape[0] - lags + 1)]
        matrix = np.conj(trajmat.T)

    elif mode == "toeplitz":
        matrix = scipy.linalg.toeplitz(x)  # more fast

    elif mode == "hankel":
        matrix = scipy.linalg.hankel(x)  # more fast

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

        matrix = np.zeros((N + lags - 1, lags), dtype=x.dtype)

        # full mode
        # TODO: Vectorize CYCLE!
        for i in range(lags):
            matrix[i : i + N, i] = x

        #         if (mode in ['mcovar','modified']):mode = 'covar'; modify = True

        if mode == "prewindowed" or mode == "prew":
            matrix = matrix[:N, :]
        elif mode == "postwindowed" or mode == "postw":
            matrix = matrix[lags - 1 :, :]
        elif mode == "same":
            matrix = matrix[(lags - 1) // 2 : (lags - 1) // 2 + N, :]
        elif mode == "valid" or mode == "covar":
            matrix = matrix[lags - 1 : -lags + 1, :]

    else:
        raise ValueError(
            """ 
                         mode have to be one of 
                         ['full','prewindowed','postwindowed',
                        'prew','postw','covar','valid','same',
                        'traj', 'caterpillar', 
                        'trajectory', 'hankel', 'toeplitz'] """
        )

    #     # TODO: TRY IF NP.FLIP... FASTER ?
    #     if(modify):
    #         matrix = np.append(matrix,np.conj(matrix[:,::-1]),axis=0)

    return matrix


# -------------------------------------------------------------------
def covariance_matrix(x, mode="full", lags=None, ret_base=False):
    """
    Covariance of data lags matrix.

    Parameters
    -------------
    * x: input 1d ndarray.
    * mode: string,
       mode = {'full',prew', 'postw', 'covar','traj', 'hanekl';'toeplitz'}
       > mode = 'full': lags_matrix is the full toeplitz convolutional matrix
           with dimentions [lags+N-1,lags],
            ..math::
            out = [ [x,0..0]^T,[0,x,0..0]^T,...,[0,..0,x]^T ],
            where: N is the x size.
       > mode =  'prew': lags_matrix is the prewindowed matrix with
            first N columns of full matrix, and dimention = [N,lags];
       > mode = 'postw': lags_matrix is the postwindowed matrix with
            last N columns of full matrix, and dimention = [N,lags];
       > mode = 'covar' or 'valid': lags_matrix is the trimmed full matrix with
            cut first and last m coluns (out = full[lags:N-lags,:]),
            with dimention = [N-lags+1,lags];
       > mode = 'same': conv_matrix is the trimmed full matrix with
            cut first and last m coluns
            (out = full[(lags-1)//2:N+(lags-1)//2,:]),
            with dimention = [N,lags];
       > mode = 'traj': lags_matrix is trajectory
           or so-called caterpillar matrix
            with dimention = [N,lags];
       > mode = 'hanekl': lags_matrix is the Hankel matrix
            with dimention = [N,N];
       > mode = 'toeplitz': lags_matrix is the symmetric Toeplitz matrix,
            with dimention = [N,N].
    * lags: int or None,
        number of lags (N//2 dy default).
    * ret_base: bool,
        if true, than lags matrix will be also returned.

    Returns
    -----------
    > ret_base is False:
        * matrix: 2d ndarray.
    > ret_base is True:
        * matrix: 2d ndarray,
            covariance matrix.
        * lags_matrix:
            lags matrix.
    Notes
    ---------
    * There are permissioned synonyms:
         'prew' and 'prewindowed';
         'postw' and 'postwindowed';
         'traj', 'caterpillar' and 'trajectory'.

    See also
    -----------
    kernel_matrix
    lags_matrix
    """
    mtx = lags_matrix(x, lags=lags, mode=mode)

    if ret_base:
        return np.dot(mtx.T, np.conj(mtx)), mtx
    else:
        return np.dot(mtx.T, np.conj(mtx))


# -------------------------------------------------------------------
def conv_matrix(x, mode="full", lags=None):
    """
    Data convolution matrix (backward to correlation).

    Parameters
    -------------
    * x: input 1d ndarray.
    * mode: string,
       mode = {'full',prew', 'postw', 'covar','valid','same','traj', 'hanekl';'toeplitz'}
       > mode = 'full': lags_matrix is the full toeplitz convolutional matrix
           with dimentions [lags+N-1,lags],
            ..math::
            out = [ [0,..0,x]^T,[0,..0,x,0]^T,..., [0,x,0..0]^T, [x,0..0]^T ],
            where: N is the x size.
       > mode =  'prew': conv_matrix is the prewindowed matrix with
            first N columns of full matrix, and dimention = [N,lags];
       > mode = 'postw': conv_matrix is the postwindowed matrix with
            last N columns of full matrix, and dimention = [N,lags];
       > mode = 'covar': conv_matrix is the trimmed full matrix with
            cut first and last m coluns (out = full[lags-1:N,:]),
            with dimention = [N-lags+1,lags];
       > mode = 'valid': conv_matrix is the trimmed full matrix with
            cut first and last m coluns (out = full[lags-1:N,:]),
            with dimention = [N-lags+1,lags];
       > mode = 'same': conv_matrix is the trimmed full matrix with
            cut first and last m coluns (out = full[(lags-1)//2:N+(lags-1)//2,:]),
            with dimention = [N,lags];
       > mode = 'traj': conv_matrix is trajectory
           or so-called caterpillar matrix
            with dimention = [N,lags];
       > mode = 'hanekl': conv_matrix is the Hankel matrix
            with dimention = [N,N];
       > mode = 'toeplitz': conv_matrix is the symmetric Toeplitz matrix,
            with dimention = [N,N].
    * lags: int or None,
        number of lags (columns) in the output matrx (N//2 dy default).
    Returns
    -----------
    * conv_matrix: 2d ndarray.
    Notes
    ---------
    * The matrix is corresponded to the so-called corrmtx
        with mirrowed colunms
       (see Marple S.L. DSA and etc),
       or convolutional_matrix (in Hayes);
       full mode corresponds to the autocorrelation mode
       in some notations.
    * there are permissioned synonyms:
         'prew' and 'prewindowed';
         'postw' and 'postwindowed';
         'traj', 'caterpillar' and 'trajectory'.

    """

    x = np.asarray(x)

    N = x.shape[0]

    if lags == None:
        lags = N // 2

    if mode in ["caterpillar", "traj", "trajectory"]:
        trajmat = scipy.linalg.hankel(x, np.zeros(lags)).T
        trajmat = trajmat[:, : (x.shape[0] - lags + 1)]
        matrix = np.conj(trajmat.T)[::-1]

    elif mode == "toeplitz":
        matrix = scipy.linalg.toeplitz(x)[::-1]  # more fast

    elif mode == "hankel":
        matrix = scipy.linalg.hankel(x)[::-1]  # more fast

    elif mode in [
        "full",
        "prewindowed",
        "postwindowed",
        "prew",
        "postw",
        "covar",
        "valid",
        "same",
    ]:

        matrix = np.zeros((N + lags - 1, lags), dtype=x.dtype)

        # full mode
        # TODO: Vectorize CYCLE!
        for i in range(lags):
            matrix[lags - i - 1 : lags - i - 1 + N, i] = x
        #         if (mode in ['mcovar','modified']):mode = 'covar'; modify = True

        if mode == "prewindowed" or mode == "prew":
            matrix = matrix[:N, :]
        elif mode == "postwindowed" or mode == "postw":
            matrix = matrix[lags - 1 :, :]
        elif mode == "filed" or mode == "covar" or mode == "valid":
            matrix = matrix[lags - 1 : N, :]
        elif mode == "same":
            matrix = matrix[(lags - 1) // 2 : (lags - 1) // 2 + N, :]

    else:
        raise ValueError(
            """ 
                         mode have to be one of 
                         ['full','prewindowed','postwindowed','valid','same',
                        'prew','postw','covar','traj', 'caterpillar', 
                        'trajectory', 'hankel', 'toeplitz'] """
        )

    return matrix
