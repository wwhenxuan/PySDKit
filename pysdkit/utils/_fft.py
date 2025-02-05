import numpy as np
from numpy import fft as f


def fft(ts: np.ndarray) -> np.ndarray:
    """Fast Fourier Transform"""
    return f.fft(ts)


def ifft(ts: np.ndarray) -> np.ndarray:
    """Inverse Fast Fourier Transform"""
    return f.ifft(ts)


def fft2d(img: np.ndarray) -> np.ndarray:
    """Fast Fourier Transform for 2D Images"""
    return f.fft2(img)


def ifft2d(img: np.ndarray) -> np.ndarray:
    """Inverse Fast Fourier Transform for 2D Images"""
    return f.ifft2(img)


def fftshift(ts: np.ndarray) -> np.ndarray:
    """Fast Fourier Transform Shift"""
    return f.fftshift(ts)


def ifftshift(ts: np.ndarray) -> np.ndarray:
    """Inverse Fast Fourier Transform"""
    return f.ifftshift(ts)
