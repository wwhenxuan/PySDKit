# -*- coding: utf-8 -*-
"""
Created on 2025/02/06 10:41:14
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
用于计算信号的相关性
"""
import numpy as np
from numpy.fft import fft, ifft

from typing import Optional


def correlation(
    x: np.ndarray,
    y: np.ndarray = None,
    mode: Optional[str] = "linear",
    unbias: Optional[bool] = False,
) -> np.ndarray:
    """
    Correlation function: R[i] = sum_n(x[n+i]*conj(y[n])).

    :param x: inputs 1d signal (ndarray)
    :param y: y is None - autocorrelation function is taken
    :param mode: mode = {'linear', 'full','same','straight'}:
                 * 'linear': compute correlation using mean and variance in the time domain
                 * 'straight': (by default) straight direction (size of model N), R = ifft(fft(x, 2N)*conj(fft(y, 2N)), 2N)[:N], size of model N.
                 * 'full': size of output 2N-1 (bith directions): R = ifft(fft(x, 2N)*conj(fft(y, 2N)), 2N), size of output 2N-1.
                 * 'same': the same size of output as input (N): R = R[N//2-1:-N//2], size of model N lags from -N/2 to N/2.
    :param unbias: if True, R = {R[n]/(N-|n|)}_n^N-1.
    :return: 1d ndarray, correlation function

    Notes
    -------------
    * Basicly the correlation is calculated above two derections (in the straight and backward direction),
      consequencely it is taken for double size of samples (full mode).
    * Here correlation function is calculated using fast Fourier transform, thus more correctly to say:
        R = ifft( fft(x, 2N)* conj( fft(y, 2N) ) ) [N+1:],[:N], where first part is backward part, and the second part is straightforward.
    """
    x = np.asarray(x)

    if y is None:
        # 计算自相关
        y = x
    else:
        # 两段输入序列的长度必须相等
        y = np.asarray(y)
        if y.shape != x.shape:
            raise ValueError("y.shape ! = x.shape")

    if mode == "linear":
        # 在时域中通过均值和方差计算相关性
        # 计算均值
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        # 计算协方差
        covariance = np.sum((x - mean_x) * (y - mean_y))

        # 计算标准差
        std_x = np.sqrt(np.sum((x - mean_x) ** 2))
        std_y = np.sqrt(np.sum((y - mean_y) ** 2))

        # 计算相关系数
        correlation = covariance / (std_x * std_y)

        return correlation

    # 通过傅里叶变换在频域中更加高效的计算相关性
    N = x.shape[0]
    Sp = fft(x, 2 * N) * np.conj((fft(y, 2 * N)))
    R = ifft(Sp)

    if unbias:
        R[:N] /= N - np.arange(N)
        R[N + 1 :] /= np.arange(N - 1) + 1
    else:
        R[:N] /= N
        R[N + 1 :] /= N - 1

    if mode is "full":
        # ifftshift
        R = np.append(R[N + 1 :], R[:N])

    elif mode is "straight":
        R = R[:N]

    elif mode is "same":
        R = np.append(R[N + N // 2 :], R[: N // 2])
    else:
        raise NotImplementedError("use one of the aviliable modes")

    return R


def autocorrelation(
    signal: np.ndarray,
    mode: Optional[str] = "linear",
    max_lag: Optional[int] = None,
    N: Optional[int] = None,
) -> np.ndarray:
    """
    Autocorrelation function for signal or time series.

    :param signal: the input 1D numpy ndarray
    :param mode: {'linear', 'fft'}，可以选用在时域中通过线性算法计算或是在频域中计算
    :param max_lag: 最大滞后量，默认为数据长度的一半
    :param N: int or None, need as heritage for old-style function
    :return: 自相关系数数组。

    **Note** 我们提供了两种不同的计算方法，根据您的实际需求可以选择是在时域还是在频域中计算。
    """
    # 确保输入的数据形式不会出错
    signal = np.asarray(signal)

    if mode == "linear":
        # 在时域中计算相关性
        if max_lag is None:
            max_lag = len(signal) // 2  # 默认最大滞后量为数据长度的一半

        # 获取信号的统计信息
        mean = np.mean(signal)
        var = np.var(signal)
        n = len(signal)

        # 初始化自相关系数数组
        acf = np.zeros(max_lag + 1)

        # 计算自相关系数
        for lag in range(max_lag + 1):
            acf[lag] = (
                np.sum((signal[: n - lag] - mean) * (signal[lag:] - mean))
                / var
                / (n - lag)
            )

        return acf

    elif mode == "fft":
        # 通过快速傅里叶变换在频域中计算相关性
        if N is None:
            N = signal.shape[0]

        # 进行快速傅里叶变换获得频域特征
        Sp = np.fft.fft(signal, 2 * N)
        R = np.fft.ifft(Sp * np.conj(Sp))

        R = R[:N]

        return R

    else:
        raise NotImplementedError("use one of the aviliable modes")


def xycor(s1, s2, mode="R12", modecor="same"):
    """
    Function for special cross-correlation modes.

    Parameters
    -------------
    * s1,s2: 1d ndarrays
        input signals.
    * mode: string,
        cross-correlation modes
        mode={'R12','R21','Rfb','Rfb12','Rfb21'}.
    * modecor: string,
        mode of correlation function,
        modecor={'same','full','straight'}.

    Returns
    ----------
    *R_x,R_y: 1d ndarrays,
        output arrays, depends on mode:
        > R12: Rf, R1.
        > R21: Rb, R2.
        > Rfb: Rf, Rb.
        > Rfb12: Rf*R1^*, Rb*R2^*.
        > Rfb21: Rf*R2^*, Rb*R1^*.

    Notes
    --------
    * There are following notations are use:
        Rf = s2 \cdot s1^*;
        Rb = s1 \cdot s2^*;
        R1 = s1 \cdot s1^*;
        R2 = s2 \cdot s2^*;
        where \cdot denote convolution operation.

    """
    unbias = False
    if mode in ["R12", "None", None]:
        R_x = correlation(s2, s1, mode=modecor, unbias=unbias)
        R_y = correlation(s1, s1, mode=modecor, unbias=unbias)

    elif mode == "R21":
        R_x = correlation(s1, s2, mode=modecor, unbias=unbias)
        R_y = correlation(s2, s2, mode=modecor, unbias=unbias)

    elif mode == "Rfb":
        R_x = correlation(s1, s2, mode=modecor, unbias=unbias)
        R_y = correlation(s2, s1, mode=modecor, unbias=unbias)

    elif mode == "Rfb12":
        Rf = correlation(s1, s2, mode=modecor, unbias=unbias)
        Rb = correlation(s2, s1, mode=modecor, unbias=unbias)
        R1 = correlation(s1, s1, mode=modecor, unbias=unbias)
        R2 = correlation(s2, s2, mode=modecor, unbias=unbias)
        R_x = Rf * np.conj(R1)
        R_y = Rb * np.conj(R2)

    elif mode == "Rfb21":
        Rf = correlation(s1, s2, mode=modecor, unbias=unbias)
        Rb = correlation(s2, s1, mode=modecor, unbias=unbias)
        R1 = correlation(s1, s1, mode=modecor, unbias=unbias)
        R2 = correlation(s2, s2, mode=modecor, unbias=unbias)
        R_x = Rf * np.conj(R2)
        R_y = Rb * np.conj(R1)

    return R_x, R_y


# --------------------------------------------------------------
def convolution(x, y, mode="straight"):
    """
    Convolution function:
    ::math..
     c[i] = sum_n(x[i]*conj(y[i-n])),
        which is the same as correlation(x,y[::-1]).

    Parameters
    ------------
    * x, y: inputs 1d array (ndarray)
        if y is None - autocorrelation function is taken.
    * mode: ['xcorr','full','same','None','straight']|:
      * 'None','straight' or 'xcorr':
        (by default) straight direction (size of model N),
        R = ifft(fft(x, 2N)*conj(fft(y, 2N)), 2N)[:N],
            size of model N.
      * 'full':
        size of output 2N-1 (bith directions):
        R = ifft(fft(x, 2N)*conj(fft(y, 2N)), 2N),
        size of output 2N-1.
      * 'same':
        the same size of output as input (N):
        R = R[N//2-1:-N//2], size of model N lags from -N/2 to N/2.

    Returns
    -----------
    * 1d array convolution function.

    Notes
    -----------------
    * Basicly convolution in formule above two derections (in the straight
        and backward direction) function, consequencely it is taken for double
        size of samples (full mode), if  mode = straight, only
        the first half part of correlation is taken.
    * Correlation function allow also same mode but it does
                                not reccomended for this function.
    """
    # TODO: Add linear and circular convolutions!
    # Add convolution by datamtx
    return correlation(x, y[::-1], mode=mode, unbias=False)


# --------------------------------------------------------------
# def crossCorr_clear(x1, x2, nlags=-1):
#     '''
#         Does not applied now, for test
#         old-style function
#             need to be improved
#         same as np.asarray([np.sum(x1[i:]*np.conj(x2[:N-i])) for i in np.arange(N)])
#     '''
#     r = [np.sum(x1*np.conj(x2))]
#     for i in range(1,len(x1)):
#         r += [np.sum(x1[i:]*np.conj(x2[:-i]))]
#     return np.array(r)
# --------------------------------------------------------------


# # Does not applied Now due to many modes (unconvinient to use)
# __CROSS_MODIFED_MODES__ = ['',None,'None','R12','R21','Rfb','Rfb12','Rfb21']

# def cross_modified_correlation(x,y, take_mean = True,unbias = True, FB = True, cormode = 'full', crossmode='None'):
#     '''
#         Does not applied now, for test

#         crossmode = ['None','R12','R21','Rfb','Rfb12','Rfb21']
#         cormode   = ['None','valid','xcorr','full','same', 'argmax']
#     '''
#     x = np.asarray(x)
#     N=x.shape[0]

#     if(not Nlags):
#         Nlags = N//2

#     if(y is None):
#         y = x
#     else:
#         y = np.asarray(x)
#         if(x.shape != y.shape):
#             raise ValueError('x.shape != y.shape')

#     if(crossmode=='R12'):
#         R11 = correlation(x,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R12 = correlation(x,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R = R12*np.conj(R11)

#     elif(mode=='R21'):
#         R11 = correlation(x,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R21 = correlation(y,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R = R21*np.conj(R11)

#     elif(mode=='Rfb'):
#         R12 = correlation(x,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R21 = correlation(y,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R = R21*np.conj(R12)

#     elif(mode=='Rfb12'):
#         R11 = correlation(x,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R22 = correlation(y,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R12 = correlation(x,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R21 = correlation(y,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R1 = R12*np.conj(R11)
#         R2 = R21*np.conj(R22)
#         R = R1*np.conj(R2)

#     elif(mode=='Rfb21'):
#         R11 = correlation(x,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R22 = correlation(y,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R12 = correlation(x,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R21 = correlation(y,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R1 = R12*np.conj(R22)
#         R2 = R21*np.conj(R11)
#         R = R1*np.conj(R2)

#     else:
#         R11 = correlation(x,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R12 = correlation(x,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R = R12*np.conj(R11)

#     return R
