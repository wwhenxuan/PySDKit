# # -*- coding: utf-8 -*-
# """
# Created on 2025/02/06 10:37:17
# @author: Whenxuan Wang
# @email: wwhenxuan@gmail.com
# """
# import numpy as np
# import scipy
#
# from _lags import *
#
# __all__ = ["pca", "pca_cor", "kernel_pca"]
#
# from pysdkit.utils.correlation import correlation
#
#
# # -----------------------------------------------------------------
# def pca(x, order, mode="toeplitz", lags=None):
#     """
#     Estimation the signal components
#     based on the Principle-Components Analysis (PCA).
#
#     Parameters
#     ------------
#     * x: 1d ndarray,
#         input 1d signal.
#     * order: int,
#         order of the model
#         (number of valuable components,
#         size of signal subspace).
#     * mode: string,
#         The mode of correlation matrix,
#         mode = {full,covar,traj,toeplitz,hankel}.
#     * lags: int or None,
#         number of lags in covariance matrix
#         x.size//2 if None.
#
#     Returns
#     ------------
#     * components: 2d ndarray
#         components matrix with dimentions (x.shape,order).
#
#     Notes
#     ------------
#     * PCA is calculated as multiplication
#         of eigenvalues on eigenvectors of
#         the matrix
#         K(r(n),r(n)),
#         where r(n) – is the lags_matrix,fromed for x(n).
#
#     References
#     --------------
#     [1a] M.H. Hayes.
#         Statistical Digital Signal Processing and Modeling,
#         John Wiley & Sons, 1996.
#
#     Examples
#     --------------
#
#     See also
#     -------------
#     ssa,
#     kernel_pca,
#     kernel_ssa,
#     dmd,
#     kernel_dmd
#
#     """
#     x = np.asarray(x)
#     N = x.shape[0]
#
#     if lags == None:
#         lags = N // 2
#
#     # TODO: I'm not sure about number of lags
#     R = covariance_matrix(x, lags=lags, mode=mode)
#
#     # TODO: in my practice eigen value always sorted
#     # from the highest to the lowest, but probably sorting is required
#     es, ev = np.linalg.eig(R)
#
#     es = np.sqrt(es[:order])
#
#     principle_components = np.zeros(ev.shape, dtype=x.dtype)
#     for i in np.arange(order):
#         principle_components[i, :] = (ev[:, i]) * es[i]
#
#     return np.conj(principle_components)
#
#
# # -----------------------------------------------------------------
# def pca_cor(x, order, mode="toeplitz", cor_mode="same", lags=None):
#     """
#     Estimation the signal components
#     based on the Principle-Components Analysis (PCA)
#     for additinaly taken signal correlation function.
#
#     Parameters
#     ------------
#     * x: 1d ndarray,
#         input 1d signal.
#     * order: int,
#         order of the model
#         (number of valuable components,
#         size of signal subspace).
#     * mode: string,
#         the mode of correlation matrix,
#         mode = {full,covar,traj,toeplitz,hankel}.
#     * cor_mode: string,
#         the mode of additionally taken correlation function,
#         cor_mode={full,same,straight}.
#     * lags: int or None,
#         number of lags in covariance matrix
#         x.size//2 if None.
#
#     Returns
#     ------------
#     * components: 2d ndarray
#         components matrix with dimentions (x.shape,order).
#
#     Notes
#     ------------
#     * PCA is calculated as multiplication
#         of eigenvalues on eigenvectors of
#         the matrix
#         K(r(n),r(n)),
#         where r(n) – is the lags_matrix,fromed for x(n).
#
#     References
#     --------------
#     [1a] M.H. Hayes.
#         Statistical Digital Signal Processing and Modeling,
#         John Wiley & Sons, 1996.
#
#     Examples
#     --------------
#
#     See also
#     -------------
#     pca,
#     kernel_pca,
#     kernel_ssa,
#     ssa,
#     dmd,
#     kernel_dmd
#
#     """
#     x = np.asarray(x)
#     N = x.shape[0]
#
#     if lags == None:
#         lags = N // 2
#
#     r = correlation(x, mode=cor_mode)
#     # TODO: I'm not sure about number of lags
#     R = covariance_matrix(r, lags=lags, mode=mode)
#
#     # TODO: in my practice eigen value always sorted
#     # from the highest to the lowest, but probably sorting is required
#     es, ev = np.linalg.eig(R)
#
#     es = np.sqrt(es[:order])
#
#     principle_components = np.zeros(ev.shape, dtype=x.dtype)
#
#     for i in np.arange(order):
#         principle_components[i, :] = (ev[:, i]) * es[i]
#
#     return np.conj(principle_components)
#
#
# def kernel_pca(x, order, mode="toeplitz", kernel="linear", kpar=2, lags=None):
#     """
#     Estimation the signal components
#     based on the Principle-Components Analysis (PCA)
#     for additinaly taken signal correlation function.
#
#     Parameters
#     ------------
#     * x: 1d ndarray,
#         input 1d signal.
#     * order: int,
#         order of the model
#         (number of valuable components,
#         size of signal subspace).
#     * mode: string,
#         the mode of correlation matrix,
#         mode = {full,covar,traj,toeplitz,hankel}.
#     * kernal: string,
#         kernal type (linear as default)
#         kernal = {'rbf','thin_plate','linear','euclid',
#                         'minkowsky','sigmoid','polynom'}.
#     * kpar: are kernal parameters (depend on kernal type).
#     * lags: int or None,
#         number of lags in covariance matrix
#         x.size//2 if None.
#
#     Returns
#     ------------
#     * components: 2d ndarray
#         components matrix with dimentions (x.shape,order).
#
#     Notes
#     ------------
#     * PCA is calculated as multiplication
#         of eigenvalues on eigenvectors of
#         the matrix
#         K(r(n),r(n)),
#         where r(n) – is the lags_matrix,fromed for x(n).
#
#     References
#     --------------
#     [1a] M.H. Hayes.
#         Statistical Digital Signal Processing and Modeling,
#         John Wiley & Sons, 1996.
#
#     Examples
#     --------------
#
#     See also
#     -------------
#     pca,
#     pca_cor,
#     kernel_ssa,
#     ssa,
#     dmd,
#     kernel_dmd
#
#     """
#     x = np.asarray(x)
#     N = x.shape[0]
#
#     if lags == None:
#         lags = N // 2
#
#     R = kernel_matrix(
#         x, mode=mode, kernel=kernel, kpar=kpar, lags=lags, ret_base=False
#     )
#
#     es, ev = np.linalg.eig(R)
#
#     es = np.sqrt(es[:order])
#
#     principle_components = np.zeros(ev.shape, dtype=x.dtype)
#     for i in np.arange(order):
#         principle_components[i, :] = (ev[:, i]) * es[i]
#
#     return np.conj(principle_components)  # TODO: check why
#
#
# # --------------------------------------------
# def pca_fit_transform(x, order, mode="toeplitz"):
#     """
#     Alternative implementation with toeplitz and hankel matrices
#     """
#     x = np.asarray(x)
#     N = x.shape[0]
#
#     R, base = covariance_matrix(x, lags=N // 2, mode=mode, ret_base=True)
#
#     # TODO: in my practice eigen value always sorted
#     # from the highest to the lowest, but probably sorting is required
#     es, ev = np.linalg.eig(R)
#     ev = ev[:, :order]
#     es = es[:order]
#
#     # transfrom part for projections of new base matrix!
#     M = np.dot(base.T, ev)
#     #     for i in range (M.shape[1]):
#     #         M[:,i] /= es[i]
#     #         M[:,i] /= (N-i+1) // Unbias?!
#     #         if(mode=='hankel') and (i%2==0):
#     #             M[:,i] = np.conj(M[:,i])
#
#     return np.dot(base, M)
#
#
# # def kernel_pca(x, order, mode='toeplitz',  kernel = 'linear', gamma=2, lags = None):
# #     '''
# #     Kernel Principle Component Anaysis (PCA).
#
# #     Parameters:
# #     * x:     Input 1d signal.
# #     * order: Order of the model (number of valuable components, dize of signal subspace).
# #     * mode:  The mode of correlation function (toeplitz as default)
# #                 mode = {covar,mcovar,toeplitz,hankel,trajectory}.
# #     * kernal: Kernal type (linear as default)
# #                 kernal = {'rbf','thin_plate','linear','euclid','minkowsky','sigmoid','polynom'}
# #     * gamma: are kernal parameters (depend on kernal type).
# #     * lags:  number of lags in covariance matrix.
#
# #     Returns:
# #     * components nd ndarray with dimentions (x.shape,order).
#
# #     Notes:
# #     * For investigation modes see correlation.covariance_matrix.
# #     * Linear kernal takes the same as ordinary PCA.
# #     * Function in tests!
# #     * Not all kernals works in combination with every matrix modes.
#
# #     '''
# #     x = np.asarray(x)
# #     N = x.shape[0]
#
# #     if(lags == None): lags = N//2
# #     base = matrix.data_matrix(x, mcolumns=lags, mode=mode,modify=False)
# #     # conj need i do not know why.
#
# # #     base =np.vstack((base,np.conj(base[:,::-1]) ))
#
# #     R = matrix.kernel(base,base, ktype = kernel, kpar=gamma)
# #     es,ev=np.linalg.eig(R)
#
# #     es = np.sqrt(es[:order])
#
# #     principle_components = np.zeros((order,ev.shape[0]),dtype = x.dtype)
# #     for i in np.arange(order):
# #         principle_components[i,:] =(ev[:,i])*es[i]
#
# #     return np.conj(principle_components) # TODO: check why
#
#
# # def kernel_pca(x, order, mode='toeplitz',  kernel = 'rbf', gamma=2, fb = False):
# #     '''
# #     Kernel Principle Component Anaysis (PCA).
#
# #     Parameters:
# #     * x:     Input 1d signal.
# #     * order: Order of the model (number of valuable components, dize of signal subspace).
# #     * mode:  The mode of correlation function (toeplitz as default)
# #                 mode = {covar,mcovar,toeplitz,hankel,trajectory}.
# #     * kernal: Kernal type (rbf as default)
# #                 kernal = {'rbf','thin_plate','linear','euclid','minkowsky','sigmoid','polynom'}
# #     * gamma: are kernal parameters (depend on kernal type).
# #     * lags:  number of lags in covariance matrix.
#
# #     Returns:
# #     * components nd ndarray with dimentions (x.shape,order).
#
# #     Notes:
# #     * For investigation modes see correlation.covariance_matrix.
# #     * Linear kernal takes the same as ordinary PCA.
# #     * Function in tests!
# #     * Not all kernals works in combination with every matrix modes.
#
# #     '''
#
# #     x = np.asarray(x)
# #     N = x.shape[0]
#
# #     #TODO: I'm not sure about number of lags
# #     base = matrix.data_matrix(x, mcolumns=N//2, mode=mode)
#
# #     R = matrix.kernel(base,base, ktype = kernel, kpar=gamma)
#
# # #Normalization (from https://github.com/iqiukp/Support-Vector-Data-Description-SVDD/tree/master/DimensionalityReduction/drtoolbox/techniques)
# # #     N1 = R.shape[0]
#
# # #     column_sums = np.sum(R,axis=0) / N1
# # #     total_sum   = np.sum(column_sums) / N1
# # #     J = np.ones(N1)*column_sums
# # #     R = R - J - J.T+total_sum #Normalization
#
# # #     base =np.vstack((base,np.conj(base[:,::-1]) ))
# #     es,ev=matrix.eig(R)
#
# #     es = np.sqrt(es[:order])
#
# #     #TODO: I'm not sure about using ev[:N,i] (restriction to first N points)
# #     principle_components = np.zeros((order,N),dtype = x.dtype)
# #     for i in np.arange(order):
# #         principle_components[i,:] =np.conj(ev[:,i])*es[i]
#
# #     return principle_components
#
# # #-----------------------------------------------------------------
# # def kernal_pca(x, order, mode='full',  kernel = 'rbf', gamma=2, shift=1, fb = False):
# #     '''
# #     Kernal Principle Component Anaysis (PCA).
#
# #     Parameters:
# #     * x:     Input 1d signal.
# #     * order: Order of the model (number of valuable components, dize of signal subspace).
# #     * mode:  The mode of correlation function (caterpillar as default).
# #     * kernal: Kernal type (rbf as default).
# #     * gamma, shift: are kernal parameters (depend on kernal type).
# #     * fb:    If Ture, force-back matrice (modified covariance matrix will be taken).
#
# #     Returns:
# #     * components nd ndarray with dimentions (x.shape,order).
#
# #     Notes:
# #     * For investigation modes see correlation.covariance_matrix.
# #     * Function in tests!
#
# #     '''
# #     x = np.asarray(x)
# #     N = x.shape[0]
#
# #     #TODO: I'm not sure about number of lags
# #     base = signals.matrix.datamatrix(x, mcolumns=N, mode=mode)
#
# #     R = signals.matrix.kernal(base,gamma=gamma,shift=shift,mode=kernel)
#
# #     #Normalization
# #     N1 = R.shape[0]
# #     column_sums = np.sum(R,axis=0) / N1
# #     total_sum   = np.sum(column_sums) / N1
# #     J = np.ones(N1)*column_sums
# #     R = R - J - J.T
#
# #     es,ev=np.linalg.eig(R)
# #     ev = ev[:,:order]
# #     es = np.sqrt(es[:order])
#
# #     #TODO: I'm not sure about using ev[:N,i] (restriction to first N points)
# #     principle_components = np.zeros((N,order),dtype = x.dtype)
# #     for i in np.arange(order):
# #         principle_components[:,i] =(ev[:N,i])*es[i]
#
# #     return principle_components
#
#
# # -----------------------------------------------------------------
#
#
# # -----------------------------------------------------------------
# # def kernal_pca(x, order, mode='toeplitz',  kernel = 'rbf', gamma=2, shift=1, fb = False):
# #     '''
# #     Kernal Principle Component Anaysis (PCA).
#
# #     Parameters:
# #     * x:     Input 1d signal.
# #     * order: Order of the model (number of valuable components, dize of signal subspace).
# #     * mode:  The mode of correlation function (caterpillar as default).
# #     * kernal: Kernal type (rbf as default).
# #     * gamma, shift: are kernal parameters (depend on kernal type).
# #     * fb:    If Ture, force-back matrice (modified covariance matrix will be taken).
#
# #     Returns:
# #     * components nd ndarray with dimentions (x.shape,order).
#
# #     Notes:
# #     * For investigation modes see correlation.covariance_matrix.
# #     * Function in tests!
#
# #     '''
# #     x = np.asarray(x)
# #     N = x.shape[0]
# #     #TODO: I'm not sure about number of lags
# #     base = signals.matrix.datamatrix(x, mcolumns=N, mode=mode, modify=fb)
#
#
# #     R = signals.matrix.kernal(base,gamma=gamma,shift=shift,mode=kernel)
#
# #     N1 = R.shape[0]
#
# #     column_sums = np.sum(R,axis=0) / N1
# #     total_sum   = np.sum(column_sums) / N1
# #     J = np.ones(N1)*column_sums
# #     R = R - J - J.T #Normalization
#
# #     es,ev=np.linalg.eig(R)
# #     ev = ev[:,:order]
# #     es = np.sqrt(es[:order])
#
# #     #TODO: I'm not sure about using ev[:N,i] (restriction to first N points)
# #     principle_components = np.zeros((N,order),dtype = x.dtype)
#
# #     for i in np.arange(order):
# #         principle_components[:,i] =np.conj(ev[:N,i])*es[i]
#
# #     return principle_components
# # -----------------------------------------------------------------
#
# # from signals import correlation, matrix
# # __EPSILON__ = 1e-8
# # def pca(x, order, mode='full', fb=False):
# #     '''
# #     Estimation the signal components based on the principle-components analysis.
#
# #     Parameters:
# #     * x:     Input 1d signal.
# #     * order: Order of the model (number of valuable components, dize of signal subspace).
# #     * mode:  The mode of correlation function (caterpillar as default).
# #     * fb:    If Ture, force-back matrice (modified covariance matrix will be taken).
#
# #     Returns:
# #     * components nd ndarray with dimentions (x.shape,order).
#
# #     Notes:
# #     * For investigation modes see correlation.covariance_matrix.
#
# #     PCA algorithm:
# #     ..math:
# #         eig_vects, eig_vals = ev(xx^H)
# #         P(n)= eig_vects[:,:order]*eig_vals[:order],
# #     where
# #       * eig_vects are the eigen vecotrs,
# #       * eig_vals are the eigen velues,
# #       * ev(xx^H) eigen decomposition of the covariation matrix.
#
# #     Examples:
#
# #     '''
#
# #     x = np.asarray(x)
# #     N = x.shape[0]
# #     n_lags = N
# #     R,r_x = correlation.covariance_matrix(
# #                x, Nlags=n_lags, mode=mode, FB=fb, take_mean=False,ret_base=True)
#
# #     #TODO: in my practice eigen value always sorted
# #     # from the highest to the lowest, but probably sorting is required
# #     es,ev=np.linalg.eig(R)
# #     es = np.sqrt(es)+__EPSILON__
#
# #     projections = np.zeros((N,order),dtype = x.dtype)
#
# #     for i in np.arange(order):
# #         projections[:,i] = np.convolve(x.conj(), ev[:,i],'full')[N-1:]
#
#
# #     return np.conj(projections)
# # -----------------------------------------------------------------
#
#
# # def pca(x, order, mode='full', fb=False):
# #     '''
# #     Estimation the signal components based on the principle-components analysis.
#
# #     Parameters:
# #     * x:     Input 1d signal.
# #     * order: Order of the model (number of valuable components, dize of signal subspace).
# #     * mode:  The mode of correlation function (caterpillar as default).
# #     * fb:    If Ture, force-back matrice (modified covariance matrix will be taken).
#
# #     Returns:
# #     * components nd ndarray with dimentions (x.shape,order).
#
# #     Notes:
# #     * For investigation modes see correlation.covariance_matrix.
#
# #     PCA algorithm:
# #     ..math:
# #         eig_vects, eig_vals = ev(xx^H)
# #         P(n)= eig_vects[:,:order]*eig_vals[:order],
# #     where
# #       * eig_vects are the eigen vecotrs,
# #       * eig_vals are the eigen velues,
# #       * ev(xx^H) eigen decomposition of the covariation matrix.
#
# #     Examples:
#
# #     '''
#
# #     x = np.asarray(x)
# #     N = x.shape[0]
#
# #     R,r_x = correlation.covariance_matrix(
# #                x, Nlags=n_lags, mode=mode, FB=fb, take_mean=False,ret_base=True)
#
# #     #TODO: in my practice eigen value always sorted
# #     # from the highest to the lowest, but probably sorting is required
# #     es,ev=np.linalg.eig(R)
# #     es = np.sqrt(es)+__EPSILON__
#
# #     projections = np.zeros((N,order),dtype = x.dtype)
#
# #     for i in np.arange(order):
# #         principle_components = (ev[:,i])*es[i]
# #         projections[:,i] = np.dot(r_x, principle_components.H)/es[i]   #PCA Here!
#
#
# #     return np.conj(principle_components)
# # #-----------------------------------------------------------------
# # def svd_turnication(x, order, mode='full', fb=False):
# #     '''
# #     Estimation the signal components based on the singular-value decomposition (or eigen decomposition).
#
# #     Parameters:
# #     * x:     Input 1d signal.
# #     * order: Order of the model (number of valuable components, dize of signal subspace).
# #     * mode:  The mode of correlation function (caterpillar as default).
# #     * fb:    If Ture, force-back matrice (modified covariance matrix will be taken).
#
# #     Returns:
# #     * components nd ndarray with dimentions (x.shape,order).
#
# #     Notes:
# #     * For investigation modes see correlation.covariance_matrix.
#
# #     PCA algorithm:
# #     ..math:
# #         eig_vects, eig_vals = ev(xx^H)
# #         P(n)= eig_vects[:,:order]*eig_vals[:order],
# #     where
# #       * eig_vects are the eigen vecotrs,
# #       * eig_vals are the eigen velues,
# #       * ev(xx^H) eigen decomposition of the covariation matrix.
#
# #     Examples:
#
# #     '''
#
# #     x = np.asarray(x)
# #     N = x.shape[0]
#
# #     R = correlation.covariance_matrix(
# #                x, Nlags=n_lags, mode=mode, FB=fb, take_mean=False)
#
# #     #TODO: in my practice eigen value always sorted
# #     # from the highest to the lowest, but probably sorting is required
# #     es,ev=np.linalg.eig(R)
# #     es = np.sqrt(es)+__EPSILON__
#
# #     projections = np.zeros((N,order),dtype = x.dtype)
#
# #     for i in np.arange(order):
# #         projections[:,i] = (ev[:,i])*es[i]
#
# #     return np.conj(projections)
