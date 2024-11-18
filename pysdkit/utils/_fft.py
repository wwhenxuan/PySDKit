import numpy as np
from numpy import fft as f


def fft(ts: np.ndarray) -> np.ndarray:
    """Fast Fourier Transform"""
    return f.fft(ts)


def ifft(ts: np.ndarray) -> np.ndarray:
    """Inverse Fast Fourier Transform"""
    return f.ifft(ts)


def fftshift(ts: np.ndarray) -> np.ndarray:
    """Fast Fourier Transform Shift"""
    return f.fftshift(ts)


def ifftshift(ts: np.ndarray) -> np.ndarray:
    """Inverse Fast Fourier Transform"""
    return f.ifftshift(ts)
