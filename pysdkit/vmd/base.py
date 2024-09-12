import numpy as np
from numpy import fft as f


class Base(object):
    """Base class for signal decomposition algorithms of VMD family"""

    @staticmethod
    def fft(ts: np.ndarray) -> np.ndarray:
        """Fast Fourier Transform"""
        return f.fft(ts)

    @staticmethod
    def ifft(ts: np.ndarray) -> np.ndarray:
        """Inverse Fast Fourier Transform"""
        return f.ifft(ts)

    @staticmethod
    def fftshift(ts: np.ndarray) -> np.ndarray:
        """Fast Fourier Transform Shift"""
        return f.fftshift(ts)

    @staticmethod
    def ifftshift(ts: np.ndarray) -> np.ndarray:
        """Inverse Fast Fourier Transform"""
        return f.ifftshift(ts)

    @staticmethod
    def fmirror(ts: np.ndarray, sym: int) -> np.ndarray:
        """
        Implements a signal mirroring expansion function.
        This function mirrors 'sym' elements at both the beginning and the end of the given array 'ts',
        to create a new extended array.
        :param ts: The one-dimensional numpy array to be mirrored.
        :param sym: The number of elements to mirror from both the start and the end of the array 'ts'.
                    This value must be less than or equal to half the length of the array.
        :return: The array after mirror expansion, which will have a length equal to the original
                  array length plus twice the 'sym'.
        Note:
        If 'sym' exceeds half the length of the array,
        the function may not work as expected, so it's recommended to check the value of 'sym' beforehand.
        """
        fMirr = np.append(np.flip(ts[:sym], axis=0), ts)
        fMirr = np.append(fMirr, np.flip(ts[-sym:], axis=0))
        return fMirr

    @staticmethod
    def multi_fmirror(ts: np.ndarray, C: int, T: int) -> np.ndarray:
        """
        Implements a multi-channel time series mirroring expansion function.
        This function mirrors the first and last 'T//2' elements of each channel in the input time series 'ts'.
        It is designed for use with time series data where 'ts' has multiple channels.

        :param ts: The two-dimensional numpy array representing the multi-channel time series.
                   Shape of 'ts' is expected to be (C, T), where C is the number of channels and T is the length of each time series.
        :param C: The number of channels in the time series.
        :param T: The length of each channel in the time series.
        :return: A numpy array of shape (C, 2*T) where each channel is extended with mirrored data at both the beginning and end.
                 The middle part of each channel remains the original time series.

        Note:
        The function mirrors 'T//2' elements because this value is used to define the range of data to be mirrored
        at both the start and the end of each channel. The output is twice the original length of 'T'
        to accommodate the mirrored sections at both ends along with the original time series.
        """
        fMirr = np.zeros(shape=(C, 2 * T))  # Initialize the output array with zeros of shape (C, 2 * T)
        fMirr[:, 0: T // 2] = np.flip(ts[:, 0: T // 2], axis=1)  # Mirror the first T//2 elements at the beginning
        fMirr[:, T // 2: 3 * T // 2] = ts  # Include the original time series in the middle
        fMirr[:, 3 * T // 2: 2 * T] = np.flip(ts[:, T // 2:], axis=1)  # Mirror the last T//2 elements at the end
        return fMirr
