# -*- coding: utf-8 -*-
"""
Created on Thu July 31 20:24:54 2025
@author: Whenxuan Wang Xingwang Shao
@email: sxw2830893287@gmail.com

Feature Mode Decomposition (FMD) Implementation in Python

Based on:
Y. Miao, B. Zhang, C. Li, J. Lin, D. Zhang
"Feature Mode Decomposition: New Decomposition Theory for Rotating Machinery Fault Diagnosis"
IEEE Transactions on Industrial Electronics, 2022
"""

import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft
from scipy.signal import firwin, hilbert
from typing import List, Tuple, Optional


class FMD:
    """
    Feature Mode Decomposition

    A new decomposition theory tailored for feature extraction of machinery fault.
    FMD decomposes different modes using designed adaptive finite-impulse response (FIR) filters.
    It takes the impulsiveness and periodicity of fault signals into consideration simultaneously
    using correlated kurtosis as the objective function.

    :param filter_size: Length of the FIR filter
    :param cut_num: Number of frequency band cuts for filter bank initialization
    :param mode_num: Final number of modes to extract
    :param max_iter_num: Maximum iteration number for filter updating
    """
    def __init__(self, filter_size:int=30, cut_num:int=7, mode_num:int=2, max_iter_num:int=20):

        self.filter_size = filter_size
        self.cut_num = cut_num
        self.mode_num = mode_num
        self.max_iter_num = max_iter_num

    def fit_transform(self, fs:float, x:np.ndarray) -> np.ndarray:
        """

        :param fs: Sampling frequency
        :param x: Input signal to be decomposed
        :return: the decomposed results of IMFs
        """
        x = np.asarray(x).flatten()
        freq_bound = np.arange(0, 1, 1 / self.cut_num)
        temp_filters = np.zeros((self.filter_size, self.cut_num))

        # Initial filter bank
        for n, fb in enumerate(freq_bound):
            b = firwin(self.filter_size, 
                       [fb + np.finfo(float).eps, fb + 1 / self.cut_num - np.finfo(float).eps], 
                       window='hann', pass_zero=False)
            temp_filters[:, n] = b

        temp_sig = np.tile(x[:, None], (1, self.cut_num))
        itercount = 2

        while True:
            iternum = 2
            if itercount == 2:
                iternum = self.max_iter_num - (self.cut_num - self.mode_num) * iternum

            X = []
            for n in range(temp_filters.shape[1]):
                y_iter, f_iter, k_iter, T_iter = self._xxc_mckd(
                    temp_sig[:, n], temp_filters[:, n], iternum, M=1
                )
                freq_resp = np.abs(fft(f_iter))[:self.filter_size // 2]
                peak_freq = (np.argmax(freq_resp)) * (fs / self.filter_size)
                X.append([y_iter[:, -1], f_iter[:, -1], k_iter[-1], freq_resp, peak_freq, T_iter])

            # Update temp_sig and temp_filters
            temp_sig = np.column_stack([xi[0] for xi in X])
            temp_filters = np.column_stack([xi[1] for xi in X])

            # Correlation matrix
            corr_matrix = np.abs(np.corrcoef(temp_sig, rowvar=False))
            corr_matrix = np.triu(corr_matrix, 1)
            I, J, _ = self._max_IJ(corr_matrix)

            XI = temp_sig[:, I] - np.mean(temp_sig[:, I])
            XJ = temp_sig[:, J] - np.mean(temp_sig[:, J])

            KI = self._CK(XI, X[I][5], 1)
            KJ = self._CK(XJ, X[J][5], 1)

            if KI > KJ:
                output = J
            else:
                output = I

            temp_sig = np.delete(temp_sig, output, axis=1)
            temp_filters = np.delete(temp_filters, output, axis=1)

            if temp_filters.shape[1] == self.mode_num - 1:
                break

            itercount += 1

        # Final modes
        final_modes = np.column_stack([xi[0] for xi in X])
        return final_modes.T

    def _xxc_mckd(
            self, x:np.ndarray, f_init:np.ndarray, term_iter:int, T:Optional[float]=None, M:int=3
    )->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Update filter using Maximum Correlated Kurtosis Deconvolution
        :param x: Input signal
        :param f_init: Initial filter coefficients
        :param term_iter: Maximum iterations
        :param T: Period in samples, if None, then caculate period using autocorrelation
        :param M: Number of period to select
        :return:
        - y_final: Updated decomposed results of IMFs
        - f_final: Updated filter coefficients
        - ck_iter: Update Correlated Kurtosis values
        - T: Updated period result
        """

        if T is None:
            env = np.abs(hilbert(x)) - np.mean(np.abs(hilbert(x)))
            T = self._TT(env)

        T = int(round(T))
        x = np.asarray(x).flatten()
        L = len(f_init)
        N = len(x)

        XmT = np.zeros((L, N, M + 1))
        for m in range(M + 1):
            for l in range(L):
                if l == 0:
                    XmT[l, m * T:, m] = x[:N - m * T]
                else:
                    XmT[l, 1:, m] = XmT[l - 1, :-1, m]

        Xinv = np.linalg.inv(XmT[:, :, 0] @ XmT[:, :, 0].T)

        f = f_init.copy()
        y_final = []
        f_final = []
        ck_iter = []

        for _ in range(term_iter):
            y = XmT[:, :, 0].T @ f
            yt = np.zeros((N, M + 1))
            for m in range(M + 1):
                if m == 0:
                    yt[:, m] = y
                else:
                    yt[T:, m] = yt[:-T, m-1]

            alpha = np.zeros_like(yt)
            for m in range(M + 1):
                idx = [i for i in range(M + 1) if i != m]
                alpha[:, m] = (np.prod(yt[:, idx], axis=1) ** 2) * yt[:, m]

            beta = np.prod(yt, axis=1)

            Xalpha = np.zeros(L)
            for m in range(M + 1):
                Xalpha += XmT[:, :, m] @ alpha[:, m]

            f = (np.sum(y ** 2) / (2 * np.sum(beta ** 2))) * (Xinv @ Xalpha)
            f /= np.sqrt(np.sum(f ** 2))

            ck_val = np.sum(np.prod(yt, axis=1) ** 2) / (np.sum(y ** 2) ** (M + 1))
            ck_iter.append(ck_val)

            env_y = np.abs(hilbert(y)) - np.mean(np.abs(hilbert(y)))
            T = int(round(self._TT(env_y)))

            XmT = np.zeros((L, N, M + 1))
            for m in range(M + 1):
                for l in range(L):
                    if l == 0:
                        XmT[l, m * T:, m] = x[:N - m * T]
                    else:
                        XmT[l, 1:, m] = XmT[l - 1, :-1, m]

            Xinv = np.linalg.inv(XmT[:, :, 0] @ XmT[:, :, 0].T)

            y_final.append(signal.lfilter(f, [1], x))
            f_final.append(f.copy())

        y_final = np.column_stack(y_final) if len(y_final) > 1 else y_final[0][:, None]
        f_final = np.column_stack(f_final) if len(f_final) > 1 else f_final[0][:, None]

        return y_final, f_final, np.array(ck_iter), T

    def _TT(self, y):
        """
        Estimate period using autocorrelation
        :param y: Analyze signal envelope
        :return: Period value
        """
        NA = signal.correlate(y, y, mode='full') / np.dot(y, y)
        NA = NA[len(NA) // 2:]
        zeroposi = 1
        for lag in range(1, len(NA)):
            if (NA[lag - 1] > 0 and NA[lag] < 0) or NA[lag] == 0:
                zeroposi = lag
                break

        # Add a check in case no zero-crossing is found
        if zeroposi >= len(NA):
            zeroposi = 1

        NA = NA[zeroposi:]
        max_position = np.argmax(NA)
        return zeroposi + max_position


    def _CK(self, x:np.ndarray, T:int, M:int=1)->np.ndarray:
        """
        Compute Correlated Kurtosis values
        :param x: Input signal
        :param T: Signal period
        :param M: Modes to be decomposed
        :return: Correlated Kurtosis values
        """
        x = np.asarray(x).flatten()
        N = len(x)
        x_shift = np.zeros((M + 1, N))
        x_shift[0] = x
        for m in range(1, M + 1):
            x_shift[m, T:] = x_shift[m - 1, :-T]
        return np.sum(np.prod(x_shift, axis=0) ** 2) / (np.sum(x ** 2) ** (M + 1))

    def _max_IJ(self, X:np.ndarray)->Tuple[int, int, float]:
        """
        Obtain index of max value of Matrix
        :param X: Input matrix
        :return:
        - I: index of row
        - J: index of column
        - X[I, J]: max value of matrix
        """
        tempI = np.argmax(X, axis=0)
        temp = np.max(X, axis=0)
        J = np.argmax(temp)
        I = tempI[J]
        return I, J, X[I, J]


def generate_test_signal(
        fs:float, duration:float, frequencies:List[List], amplitudes=None, noise_level=0.1
)->Tuple[np.ndarray, np.ndarray]:
    """
    Generate single-channel or multi-channel signals for testing

    :param fs: Sample frequency
    :param duration: Duration of signal in seconds
    :param frequencies: Describes the frequency components contained in each channel.
            - Single channel: [[f1, f2, ...]]
            - Multi-channel: [[ch1_f1, ch1_f2], [ch2_f1, ch2_f2], ...]
    :param amplitudes: Describes the amplitude components contained in each channel.
                        The amplitude corresponding to frequencies. If None, all component amplitudes default to 1.0.
                        The structure is the same as frequencies.
    :param noise_level: The intensity of the white Gaussian noise added to each channel.
    :returns
        - t: Timeline array.
        - signal: Generated signal.
            - For a single channel, the shape is (samples,).
            - For multiple channels, the shape is (samples, channels).
    """
    num_samples = int(fs * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    num_channels = len(frequencies)

    # 如果未提供振幅，则全部设为1.0
    if amplitudes is None:
        amplitudes = [[1.0] * len(freq_list) for freq_list in frequencies]

    # If amplitudes are not provided, they are all set to 1.0
    if len(frequencies) != len(amplitudes) or any(len(f) != len(a) for f, a in zip(frequencies, amplitudes)):
        raise ValueError("The structure of 'frequencies' and 'amplitudes' must match.")

    all_channels = []
    for i in range(num_channels):
        channel_signal = np.zeros(num_samples)
        for freq, amp in zip(frequencies[i], amplitudes[i]):
            channel_signal += amp * np.sin(2 * np.pi * freq * t)

        # add noise
        channel_signal += noise_level * np.random.randn(num_samples)
        all_channels.append(channel_signal)

    # If it is multi-channel,
    # convert the list to (channels, samples) and then transpose it to (samples, channels).
    # If it is a single channel, use the first element directly and keep it as a 1D array
    if num_channels > 1:
        signal = np.array(all_channels)
    else:
        signal = all_channels[0]

    return t, signal

# single-channel test
if __name__ == "__main__":
    from pysdkit.plot import plot_signal, plot_IMFs, plot_IMFs_amplitude_spectra
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    x=loadmat('x.mat')['x'].ravel()
    fs = 2e4
    t = np.arange(0, len(x))/fs
    fmd = FMD(30, 7, 2, 20)
    modes = fmd.fit_transform(fs, x)
    plot_signal(t, x, spectrum=True)
    plot_IMFs(x, modes)
    plot_IMFs_amplitude_spectra(modes, norm=True)
    plt.show()
