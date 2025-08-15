# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 20:24:54 2025
@author: Whenxuan Wang Xingwang Shao
@email: sxw2830893287@gmail.com

Multi-Channel Feature Mode Decomposition (FMD) Implementation in Python

Optimized version using broadcasting and vectorization to minimize loops.
Supports simultaneous decomposition of multiple channels.

Based on:
Y. Miao, B. Zhang, C. Li, J. Lin, D. Zhang
"Feature Mode Decomposition: New Decomposition Theory for Rotating Machinery Fault Diagnosis"
IEEE Transactions on Industrial Electronics, 2022
"""

# import numpy as np
# import scipy.signal as signal
# from scipy.fftpack import fft
# from scipy.signal import firwin, hilbert
# from typing import List, Tuple, Optional

# class FMD2D:
#     """
#         Multi-Channel Feature Mode Decomposition with Broadcasting Optimization
#         This implementation supports simultaneous decomposition of multiple channels
#         and uses vectorized operations to minimize loops, especially those related to signal length.
#
#     :param filter_size: Length of the FIR filter
#     :param cut_num: Number of frequency band cuts for filter bank initialization
#     :param mode_num: Final number of modes to extract
#     :param max_iter_num: Maximum iteration number for filter updating
#     """
#
#     def __init__(self, filter_size:int=30, cut_num:int=7, mode_num:int=2, max_iter_num:int=20):
#
#         self.filter_size = filter_size
#         self.cut_num = cut_num
#         self.mode_num = mode_num
#         self.max_iter_num = max_iter_num
#
#     def fit_transform(self, fs:float, x:np.ndarray=None)->np.ndarray:
#         """
#         Performs eigenmode decomposition on the input signal x.
#         If x is a 1D array, it is processed as a single channel.
#         If x is a 2D array (samples, channels), the decomposition is performed independently for each channel.
#
#         :param fs: Sample frequency
#         :param x: Input signal
#         :return: Decomposed modes, shape is (n_modes, n_samples, n_channels)
#         """
#         x = np.asarray(x)
#         x = x.T
#
#         if x.ndim == 1:
#             x = x.reshape(-1, 1)
#         elif x.ndim != 2:
#             raise ValueError("Input signal x must be a 1D or 2D array of shape (samples, channels).")
#
#         num_samples, num_channels = x.shape
#         all_channel_modes = []
#
#         for i in range(num_channels):
#             channel_signal = x[:, i]
#
#             freq_bound = np.arange(0, 1, 1 / self.cut_num)
#             temp_filters = np.zeros((self.filter_size, self.cut_num))
#
#             # Initial filter bank
#             for n, fb in enumerate(freq_bound):
#                 b = firwin(self.filter_size,
#                            [fb + np.finfo(float).eps, fb + 1 / self.cut_num - np.finfo(float).eps],
#                            window='hann', pass_zero=False)
#                 temp_filters[:, n] = b
#
#             temp_sig = np.tile(channel_signal[:, None], (1, self.cut_num))
#             itercount = 2
#
#             while True:
#                 iternum = 2
#                 if itercount == 2:
#                     iternum = self.max_iter_num - (self.cut_num - self.mode_num) * iternum
#
#                 X = []
#                 for n in range(temp_filters.shape[1]):
#                     y_iter, f_iter, k_iter, T_iter = self._xxc_mckd(
#                         temp_sig[:, n], temp_filters[:, n], iternum, M=1
#                     )
#                     freq_resp = np.abs(fft(f_iter))[:self.filter_size // 2]
#                     peak_freq = (np.argmax(freq_resp)) * (fs / self.filter_size)
#                     X.append([y_iter[:, -1], f_iter[:, -1], k_iter[-1], freq_resp, peak_freq, T_iter])
#
#                 temp_sig = np.column_stack([xi[0] for xi in X])
#                 temp_filters = np.column_stack([xi[1] for xi in X])
#
#                 corr_matrix = np.abs(np.corrcoef(temp_sig, rowvar=False))
#                 corr_matrix = np.triu(corr_matrix, 1)
#                 I, J, _ = self._max_IJ(corr_matrix)
#
#                 XI = temp_sig[:, I] - np.mean(temp_sig[:, I])
#                 XJ = temp_sig[:, J] - np.mean(temp_sig[:, J])
#
#                 KI = self._CK(XI, X[I][5], 1)
#                 KJ = self._CK(XJ, X[J][5], 1)
#
#                 if KI > KJ:
#                     output = J
#                 else:
#                     output = I
#
#                 temp_sig = np.delete(temp_sig, output, axis=1)
#                 temp_filters = np.delete(temp_filters, output, axis=1)
#
#                 if temp_filters.shape[1] == self.mode_num:  # 修正了循环终止条件
#                     break
#
#                 itercount += 1
#
#             # Re-iterate the last remaining mode to obtain the final result
#             final_X = []
#             for n in range(temp_filters.shape[1]):
#                 y_iter, f_iter, k_iter, T_iter = self._xxc_mckd(
#                     temp_sig[:, n], temp_filters[:, n], self.max_iter_num, M=1
#                 )
#                 final_X.append([y_iter[:, -1], f_iter[:, -1]])
#
#             final_modes = np.column_stack([xi[0] for xi in final_X])
#             all_channel_modes.append(final_modes.T)
#
#         all_channel_modes = np.array(all_channel_modes).reshape(self.mode_num, num_samples, num_channels)
#
#         return all_channel_modes
#
#     def _xxc_mckd(
#             self, x:np.ndarray, f_init:np.ndarray, term_iter:int, T:Optional[float]=None, M:int=3
#     )->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#         """
#         Update filter using Maximum Correlated Kurtosis Deconvolution
#         :param x: Input signal
#         :param f_init: Initial filter coefficients
#         :param term_iter: Maximum iterations
#         :param T: Period in samples, if None, then caculate period using autocorrelation
#         :param M: Number of period to select
#         :return:
#         - y_final: Updated decomposed results of IMFs
#         - f_final: Updated filter coefficients
#         - ck_iter: Update Correlated Kurtosis values
#         - T: Updated period result
#         """
#
#         if T is None:
#             env = np.abs(hilbert(x)) - np.mean(np.abs(hilbert(x)))
#             T = self._TT(env)
#
#         T = int(round(T))
#         x = np.asarray(x).flatten()
#         L = len(f_init)
#         N = len(x)
#
#         XmT = np.zeros((L, N, M + 1))
#         for m in range(M + 1):
#             for l in range(L):
#                 if l == 0:
#                     XmT[l, m * T:, m] = x[:N - m * T]
#                 else:
#                     XmT[l, 1:, m] = XmT[l - 1, :-1, m]
#
#         Xinv = np.linalg.inv(XmT[:, :, 0] @ XmT[:, :, 0].T)
#
#         f = f_init.copy()
#         y_final = []
#         f_final = []
#         ck_iter = []
#
#         for _ in range(term_iter):
#             y = XmT[:, :, 0].T @ f
#             yt = np.zeros((N, M + 1))
#             for m in range(M + 1):
#                 if m == 0:
#                     yt[:, m] = y
#                 else:
#                     yt[T:, m] = yt[:-T, m-1]
#
#             alpha = np.zeros_like(yt)
#             for m in range(M + 1):
#                 idx = [i for i in range(M + 1) if i != m]
#                 alpha[:, m] = (np.prod(yt[:, idx], axis=1) ** 2) * yt[:, m]
#
#             beta = np.prod(yt, axis=1)
#
#             Xalpha = np.zeros(L)
#             for m in range(M + 1):
#                 Xalpha += XmT[:, :, m] @ alpha[:, m]
#
#             f = (np.sum(y ** 2) / (2 * np.sum(beta ** 2))) * (Xinv @ Xalpha)
#             f /= np.sqrt(np.sum(f ** 2))
#
#             ck_val = np.sum(np.prod(yt, axis=1) ** 2) / (np.sum(y ** 2) ** (M + 1))
#             ck_iter.append(ck_val)
#
#             env_y = np.abs(hilbert(y)) - np.mean(np.abs(hilbert(y)))
#             T = int(round(self._TT(env_y)))
#
#             XmT = np.zeros((L, N, M + 1))
#             for m in range(M + 1):
#                 for l in range(L):
#                     if l == 0:
#                         XmT[l, m * T:, m] = x[:N - m * T]
#                     else:
#                         XmT[l, 1:, m] = XmT[l - 1, :-1, m]
#
#             Xinv = np.linalg.inv(XmT[:, :, 0] @ XmT[:, :, 0].T)
#
#             y_final.append(signal.lfilter(f, [1], x))
#             f_final.append(f.copy())
#
#         y_final = np.column_stack(y_final) if len(y_final) > 1 else y_final[0][:, None]
#         f_final = np.column_stack(f_final) if len(f_final) > 1 else f_final[0][:, None]
#
#         return y_final, f_final, np.array(ck_iter), T
#
#     def _TT(self, y):
#         """
#         Estimate period using autocorrelation
#         :param y: Analyze signal envelope
#         :return: Period value
#         """
#         NA = signal.correlate(y, y, mode='full') / np.dot(y, y)
#         NA = NA[len(NA) // 2:]
#         zeroposi = 1
#         for lag in range(1, len(NA)):
#             if (NA[lag - 1] > 0 and NA[lag] < 0) or NA[lag] == 0:
#                 zeroposi = lag
#                 break
#
#         # Add a check in case no zero-crossing is found
#         if zeroposi >= len(NA):
#             zeroposi = 1
#
#         NA = NA[zeroposi:]
#         max_position = np.argmax(NA)
#         return zeroposi + max_position
#
#
#     def _CK(self, x:np.ndarray, T:int, M:int=1)->np.ndarray:
#         """
#         Compute Correlated Kurtosis values
#         :param x: Input signal
#         :param T: Signal period
#         :param M: Modes to be decomposed
#         :return: Correlated Kurtosis values
#         """
#         x = np.asarray(x).flatten()
#         N = len(x)
#         x_shift = np.zeros((M + 1, N))
#         x_shift[0] = x
#         for m in range(1, M + 1):
#             x_shift[m, T:] = x_shift[m - 1, :-T]
#         return np.sum(np.prod(x_shift, axis=0) ** 2) / (np.sum(x ** 2) ** (M + 1))
#
#     def _max_IJ(self, X:np.ndarray)->Tuple[int, int, float]:
#         """
#         Obtain index of max value of Matrix
#         :param X: Input matrix
#         :return:
#         - I: index of row
#         - J: index of column
#         - X[I, J]: max value of matrix
#         """
#         tempI = np.argmax(X, axis=0)
#         temp = np.max(X, axis=0)
#         J = np.argmax(temp)
#         I = tempI[J]
#         return I, J, X[I, J]
