# -*- coding: utf-8 -*-
"""
Created on 2024/7/12 13:41
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
Empirical Wavelet Transform for 1D signals

Original paper:
Gilles, J., 2013. Empirical Wavelet Transform. IEEE Transactions on Signal Processing, 61(16), pp.3999-4010.
Available at: https://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=6522142.
Original Matlab toolbox: https://www.mathworks.com/matlabcentral/fileexchange/42141-empirical-wavelet-transforms
Original Code from: https://github.com/vrcarva/ewtpy
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from typing import Optional, Tuple, Union


class EWT(object):
    """
    Empirical Wavelet Transform with Class Interface.

    Gilles, J., 2013. Empirical Wavelet Transform.
    IEEE Transactions on Signal Processing, 61(16), pp.3999–4010.

    Paper link: https://ieeexplore.ieee.org/lpdocs/epic03/wrapper.htm?arnumber=6522142.
    Python code: https://github.com/vrcarva/ewtpy
    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/42141-empirical-wavelet-transforms
    """

    def __init__(
        self,
        K: Optional[int] = 5,
        log: Optional[float] = 0,
        detect: Optional[str] = "locmax",
        completion: Optional[float] = 0,
        reg: Optional[str] = "average",
        lengthFilter: Optional[float] = 10,
        sigmaFilter: Optional[float] = 5,
    ) -> None:
        """
        :param K: Maximum number of modes (signal components) to detect and extract.
        :param log: Set to 0 or 1 to indicate whether to operate in the logarithmic spectrum.
        :param detect: Method for detecting boundaries in the Fourier domain ('locmax' or other detection methods).
        :param completion: Set to 0 or 1 to indicate whether to complete the number of modes to K if fewer are detected.
        :param reg: Regularization method applied to the filter bank ('none', 'gaussian', or 'average').
        :param lengthFilter: Width of the filters used in regularization (for Gaussian or average filters).
        :param sigmaFilter: Standard deviation for the Gaussian filter in the regularization step.
        """
        self.K = K
        self.log = log
        self.detect = detect
        self.completion = completion
        self.reg = reg
        self.lengthFilter = lengthFilter
        self.sigmaFilter = sigmaFilter

    def __call__(
        self,
        signal: np.ndarray,
        N: Optional[int] = None,
        return_all: Optional[bool] = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """allow instances to be called like functions"""
        return self.fit_transform(signal=signal, N=N, return_all=return_all)

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Empirical Wavelet Transform (EWT)"

    @staticmethod
    def fmirror(ts: np.ndarray, sym: int, end: int) -> np.ndarray:
        """Implements a signal mirroring expansion function."""
        fMirr = np.append(np.flip(ts[0 : sym - end], axis=0), ts)
        fMirr = np.append(fMirr, np.flip(ts[-sym - end : -end], axis=0))
        return fMirr

    def fit_transform(
        self,
        signal: np.ndarray,
        N: Optional[int] = None,
        return_all: Optional[bool] = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Perform Empirical Wavelet Transform on the input signal.

        :param signal: Input signal array to be decomposed.
        :param N: Number of modes to extract. Defaults to the value specified during initialization.
        :param return_all: If True, return the EWT decomposition, the filter bank, and the boundaries.
                           If False, return only the EWT decomposition.

        :returns:
            - _ewt: The extracted modes from the signal.
            - mfb: The filter bank applied in the Fourier domain (only if return_all is True).
            - boundaries: Boundaries detected in the Fourier spectrum (only if return_all is True).
        """
        N = self.K if N is None else N
        # Compute the one-sided magnitude of the signal's Fourier transform
        signal_ff = np.fft.fft(signal)
        signal_ff = abs(
            signal_ff[0 : int(np.ceil(signal_ff.size / 2))]
        )  # one-sided magnitude

        # Detect boundaries in the Fourier domain
        boundaries = EWT_Boundaries_Detect(
            signal_ff,
            self.log,
            self.detect,
            N,
            self.reg,
            self.lengthFilter,
            self.sigmaFilter,
        )
        boundaries = boundaries * np.pi / round(signal_ff.size)

        # If fewer boundaries are detected, complete to reach K-1 if completion is enabled
        if self.completion == 1 and len(boundaries) < N - 1:
            boundaries = EWT_Boundaries_Completion(boundaries, N - 1)

        # Extend the signal by mirroring to avoid boundary effects during filtering
        ltemp = int(np.ceil(signal.size / 2))
        fMirr = self.fmirror(ts=signal, sym=ltemp, end=1)
        ffMirr = np.fft.fft(fMirr)

        # Build the filter bank based on the detected boundaries
        mfb = EWT_Meyer_FilterBank(boundaries, ffMirr.size)

        # Filter the signal to extract each subband
        imfs = np.zeros(mfb.shape)
        for k in range(mfb.shape[1]):
            imfs[:, k] = np.real(np.fft.ifft(np.conjugate(mfb[:, k]) * ffMirr))
        imfs = imfs[ltemp - 1 : -ltemp, :].T

        # Return the requested data
        if return_all is True:
            return imfs, mfb, boundaries
        else:
            return imfs


def ewt(
    signal: np.ndarray,
    K: Optional[int] = 5,
    log: Optional[float] = 0,
    detect: Optional[str] = "locmax",
    completion: Optional[float] = 0,
    reg: Optional[str] = "average",
    lengthFilter: Optional[float] = 10,
    sigmaFilter: Optional[float] = 5,
    return_all: Optional[bool] = False,
):
    """
    Empirical Wavelet Transform with Function Interface.

    :param signal: The input signal to be decomposed.
    :param K: Maximum number of modes (signal components) to detect and extract.
    :param log: Set to 0 or 1 to indicate whether to work with the logarithmic spectrum.
    :param detect: Method for detecting boundaries in the Fourier domain ('locmax' or other methods).
    :param completion: Set to 0 or 1 to indicate whether to complete the number of modes to K if fewer are detected.
    :param reg: Regularization method applied to the filter bank ('none', 'gaussian', or 'average').
    :param lengthFilter: Width of the filters used in regularization (for Gaussian or average filters).
    :param sigmaFilter: Standard deviation for the Gaussian filter in the regularization step.
    :param return_all: If True, return the EWT decomposition, the filter bank, and the boundaries.
                       If False, return only the EWT decomposition.

    :returns: - _ewt - The extracted modes from the signal.
              - mfb: The filter bank in the Fourier domain (only if return_all is True).
              - boundaries: Boundaries detected in the Fourier spectrum (only if return_all is True).
    """
    # Compute the one-sided magnitude of the signal's Fourier transform
    ff = np.fft.fft(signal)
    ff = abs(ff[0 : int(np.ceil(ff.size / 2))])  # one-sided magnitude

    # Detect boundaries in the Fourier domain
    boundaries = EWT_Boundaries_Detect(
        ff, log, detect, K, reg, lengthFilter, sigmaFilter
    )
    boundaries = boundaries * np.pi / round(ff.size)

    # If fewer boundaries are detected, complete to reach K-1 if completion is enabled
    if completion == 1 and len(boundaries) < K - 1:
        boundaries = EWT_Boundaries_Completion(boundaries, K - 1)

    # Extend the signal by mirroring to avoid boundary effects during filtering
    ltemp = int(np.ceil(signal.size / 2))  # Similar to MATLAB's round function
    fMirr = fmirror(ts=signal, sym=ltemp, end=1)
    ffMirr = np.fft.fft(fMirr)

    # Build the filter bank based on the detected boundaries
    mfb = EWT_Meyer_FilterBank(boundaries, ffMirr.size)

    # Filter the signal to extract each subband
    ewt = np.zeros(mfb.shape)
    for k in range(mfb.shape[1]):
        ewt[:, k] = np.real(np.fft.ifft(np.conjugate(mfb[:, k]) * ffMirr))
    ewt = ewt[ltemp - 1 : -ltemp, :].T

    # Return the requested data
    if return_all is True:
        return ewt, mfb, boundaries
    else:
        return ewt


def EWT_Boundaries_Detect(
    ff: np.ndarray,
    log: Optional[int],
    detect: Optional[str],
    N: int,
    reg: Optional[str],
    lengthFilter: Optional[float],
    sigmaFilter: Optional[float],
) -> np.ndarray:
    """
    Segments the input function `ff` into a certain number of supports (frequency bands)
    using different techniques for boundary detection and regularization.

    :param ff: The function (signal) to segment (typically in the frequency domain).
    :param log: 0 or 1, indicating whether to apply logarithmic transformation to the spectrum.
    :param detect: The method used for boundary detection. Options include:
                   - 'locmax': Mid-point between consecutive local maxima (default).
                   - 'locmaxmin': Lowest minima between consecutive local maxima.
                   - 'locmaxminf': Lowest minima between consecutive local maxima of the original spectrum.
    :param N: Maximum number of supports (modes or signal components) to detect.
    :param reg: The regularization method to apply to the spectrum. Options include:
                - 'none': No regularization.
                - 'gaussian': Gaussian filtering with width `lengthFilter` and standard deviation `sigmaFilter`.
                - 'average': Average filtering with width `lengthFilter`.
    :param lengthFilter: The width of the Gaussian or average filter for regularization.
    :param sigmaFilter: The standard deviation of the Gaussian filter for regularization.

    :returns boundaries: A list of detected boundaries (in terms of indices).

    This function detects boundaries in the frequency domain based on different detection
    methods. The spectrum can be regularized using Gaussian or average filters, which smooths
    out the spectrum before detection.

    - The 'locmax' method detects the mid-points between consecutive local maxima.
    - The 'locmaxmin' method extracts the lowest minima between consecutive maxima.
    - The 'locmaxminf' method uses the regularized spectrum for maxima detection and the original spectrum for minima.

    """
    # Apply log transformation if needed
    if log == 1:
        ff = np.log(ff)

    # Apply regularization if needed
    if reg == "average":
        regFilter = np.ones(lengthFilter) / lengthFilter
        presig = np.convolve(ff, regFilter, mode="same")  # Averaging filter

    elif reg == "gaussian":
        regFilter = np.zeros(lengthFilter)
        regFilter[regFilter.size // 2] = (
            1  # Ensure center is set for Gaussian filtering
        )
        presig = np.convolve(
            ff, gaussian_filter(regFilter, sigmaFilter), mode="same"
        )  # Gaussian filter
    else:
        presig = ff  # No regularization applied

    # Boundary detection based on the selected method
    if detect == "locmax":
        boundaries = LocalMax(presig, N)  # Mid-point between consecutive local maxima

    elif detect == "locmaxmin":
        boundaries = LocalMaxMin(
            presig, N
        )  # Lowest local minima between selected maxima

    elif detect == "locmaxminf":
        boundaries = LocalMaxMin(
            presig, N, fm=ff
        )  # Minima on the original spectrum between maxima on regularized spectrum

    else:
        raise ValueError("Invalid detection method provided.")

    return boundaries + 1  # Increment indices by 1 to account for zero-based indexing


def fmirror(ts: np.ndarray, sym: int, end: int) -> np.ndarray:
    """Implements a signal mirroring expansion function."""
    fMirr = np.append(np.flip(ts[0 : sym - end], axis=0), ts)
    fMirr = np.append(fMirr, np.flip(ts[-sym - end : -end], axis=0))
    return fMirr


def LocalMax(ff: np.ndarray, N: int) -> np.ndarray:
    """
    Segments the input function `ff` into a maximum of `K` bands by locating
    the K largest local maxima and calculating the middle points between them.

    :param ff: The input function (signal) to segment.
    :param N: The maximum number of bands (supports) to detect.

    :returns:
        - bound: A list of indices representing the detected boundaries between bands.

    This function works by identifying local maxima in the input function `ff`,
    selecting the `K` largest maxima, and then computing the midpoint between each
    consecutive maximum to define the segment boundaries.
    """
    N = N - 1
    locmax = np.zeros(ff.size)  # Initialize array to store local maxima
    locmin = max(ff) * np.ones(ff.size)  # Initialize array for local minima

    # Loop through the signal to find local maxima and minima
    for i in np.arange(1, ff.size - 1):
        if ff[i - 1] < ff[i] and ff[i] > ff[i + 1]:
            locmax[i] = ff[i]  # Mark local maxima
        if ff[i - 1] > ff[i] and ff[i] <= ff[i + 1]:
            locmin[i] = ff[i]  # Mark local minima

    N = min(N, locmax.size)  # Ensure K does not exceed the number of available maxima
    # Find the indices of the K largest maxima
    maxidxs = np.sort(locmax.argsort()[::-1][:N])

    # Calculate the midpoints between consecutive maxima to define the boundaries
    bound = np.zeros(N)
    for i in range(N):
        if i == 0:
            a = 0
        else:
            a = maxidxs[i - 1]
        bound[i] = (a + maxidxs[i]) / 2  # Calculate midpoint

    return bound


def LocalMaxMin(f: np.ndarray, N: int, fm: Optional[int] = 0) -> np.ndarray:
    """
    Segments the input function `f` into a maximum of `K` bands by detecting
    the lowest local minima between the `K` largest local maxima. If `fm` is provided,
    local maxima are computed on `f` and local minima are computed on `fm`. Otherwise,
    both maxima and minima are computed on `f`.

    This is useful when you want to compute the maxima on a regularized version of
    your signal (f) while detecting the "true" minima on a different version (fm).

    :param f: The function (signal) to segment.
    :param N: The maximum number of bands (supports) to detect.
    :param fm: (Optional) The function on which local minima will be computed.
               If not provided, local minima will be computed on `f`.

    :returns:
        - bound: A list of indices representing the detected boundaries between bands.

    This function detects local maxima and minima in the signal and then calculates
    boundaries based on the K largest maxima and the lowest minima between them.
    """
    # Initialize array to store local maxima
    locmax = np.zeros(f.size)

    # If fm is not provided, use f for both maxima and minima detection
    if type(fm) == int:
        f2 = f
    else:
        f2 = fm

    # Initialize array for local minima
    locmin = max(f2) * np.ones(f2.size)

    # Detect local minima and maxima in the signal
    for i in np.arange(1, f.size - 1):
        if (f[i - 1] < f[i]) and (f[i] > f[i + 1]):
            locmax[i] = f[i]  # Mark local maxima
        if (f2[i - 1] > f2[i]) and (f2[i] < f2[i + 1]):
            locmin[i] = f2[i]  # Mark local minima

    # Initialize the boundaries array
    bound = np.zeros(N)

    if N != -1:
        N = N - 1
        # Keep the K largest maxima
        Imax = np.sort(locmax.argsort()[::-1][:N])

        # Detect the lowest minima between two consecutive maxima
        for i in range(N):
            if i == 0:
                a = 1
            else:
                a = Imax[i - 1]

            # Sort minima between the current and previous maxima
            lmin = np.sort(locmin[a : Imax[i]])
            ind = np.argsort(locmin[a : Imax[i]])

            # Find the true minimum
            tmpp = lmin[0]
            n = 0
            if n < len(lmin):
                n = 1
                while (n < len(lmin)) and (tmpp == lmin[n]):
                    n += 1

            # Set the boundary to the midpoint of the smallest minima
            bound[i] = a + ind[n // 2] - 1
    else:
        # Case when K is set to -1, meaning no limit on the number of bands
        k = 0
        for i in range(locmin.size):
            if locmin[i] < max(f2):
                bound[k] = i - 1
                k += 1

    return bound


def EWT_Boundaries_Completion(boundaries: np.ndarray, NT: int) -> np.ndarray:
    """
    Completes the boundaries vector to ensure a total of `NT` boundaries by
    equally splitting the last band (highest frequencies) between the final boundary
    and π.

    :param boundaries: The initial boundaries vector to be completed.
    :param NT: The total number of boundaries desired.

    :returns:
        - boundaries: The completed boundaries vector.

    This function adds additional boundaries by uniformly dividing the highest frequency
    band (i.e., the region from the last boundary to π) into equal parts, such that the total
    number of boundaries reaches `NT`.
    """
    # Calculate the number of additional boundaries needed
    Nd = NT - len(boundaries)

    # Calculate the width of each new segment to be added
    deltaw = (np.pi - boundaries[-1]) / (Nd + 1)

    # Add the new boundaries to the vector
    for k in range(Nd):
        boundaries = np.append(boundaries, boundaries[-1] + deltaw)

    return boundaries


def EWT_Meyer_FilterBank(boundaries: np.ndarray, Nsig: int) -> np.ndarray:
    """
    Generates the filter bank (scaling function + wavelets) corresponding to the
    provided set of frequency segments.

    :param boundaries: A vector containing the boundaries of frequency segments.
                       Note that 0 and π should NOT be included in this vector.
    :param Nsig: The length of the input signal.

    :returns:
        - mfb: A matrix where each column represents a filter in the Fourier domain.
               The first column corresponds to the scaling function, and the
               subsequent columns represent the successive wavelets.

    This function generates a set of Meyer wavelets and a scaling function in the
    Fourier domain based on the provided frequency boundaries.
    """
    Npic = len(boundaries)  # Number of frequency boundaries
    # Compute gamma, which controls the transition width of the wavelets
    gamma = 1
    for k in range(Npic - 1):
        r = (boundaries[k + 1] - boundaries[k]) / (boundaries[k + 1] + boundaries[k])
        if r < gamma:
            gamma = r
    r = (np.pi - boundaries[Npic - 1]) / (np.pi + boundaries[Npic - 1])
    if r < gamma:
        gamma = r
    gamma = (1 - 1 / Nsig) * gamma  # Ensure gamma is strictly less than the minimum

    # Initialize the filter bank matrix
    mfb = np.zeros([Nsig, Npic + 1])

    # EWT_Meyer_Scaling - Generate the scaling function
    Mi = int(np.floor(Nsig / 2))  # Half the signal length
    w = np.fft.fftshift(
        np.linspace(0, 2 * np.pi - 2 * np.pi / Nsig, num=Nsig)
    )  # Frequency grid
    w[0:Mi] = -2 * np.pi + w[0:Mi]  # Shift the negative frequencies
    aw = abs(w)  # Absolute frequency values
    yms = np.zeros(Nsig)  # Initialize the scaling function
    an = 1.0 / (2 * gamma * boundaries[0])
    pbn = (1.0 + gamma) * boundaries[0]
    mbn = (1.0 - gamma) * boundaries[0]

    # Generate the scaling function based on the first boundary
    for k in range(Nsig):
        if aw[k] <= mbn:
            yms[k] = 1
        elif (aw[k] >= mbn) and (aw[k] <= pbn):
            yms[k] = np.cos(np.pi * EWT_beta(an * (aw[k] - mbn)) / 2)

    yms = np.fft.ifftshift(yms)  # Shift the scaling function back
    mfb[:, 0] = yms  # Store the scaling function in the filter bank

    # Generate the wavelets for the remaining boundaries
    for k in range(Npic - 1):
        mfb[:, k + 1] = EWT_Meyer_Wavelet(boundaries[k], boundaries[k + 1], gamma, Nsig)

    # Generate the final wavelet for the last boundary segment (up to π)
    mfb[:, Npic] = EWT_Meyer_Wavelet(boundaries[Npic - 1], np.pi, gamma, Nsig)

    return mfb


def EWT_beta(x: float) -> float:
    """
    Beta = EWT_beta(x)
    function used in the construction of Meyer's wavelet
    """
    if x < 0:
        bm = 0
    elif x > 1:
        bm = 1
    else:
        bm = (x**4) * (35.0 - 84.0 * x + 70.0 * (x**2) - 20.0 * (x**3))
    return bm


def EWT_Meyer_Wavelet(wn: float, wm: float, gamma: float, Nsig: int) -> np.ndarray:
    """
    Generates the 1D Meyer wavelet in the Fourier domain associated with the
    scale segment [wn, wm] with a transition ratio `gamma`.

    :param wn: The lower boundary of the frequency segment.
    :param wm: The upper boundary of the frequency segment.
    :param gamma: The transition ratio that controls the width of the transition zones
                  between frequency bands.
    :param Nsig: The number of points in the signal.

    :returns:
        - ymw: The Fourier transform of the wavelet for the frequency band [wn, wm].

    This function creates a Meyer wavelet in the Fourier domain with smooth transitions
    between frequency bands defined by `wn` and `wm`.
    """
    # Compute the frequency grid
    Mi = int(np.floor(Nsig / 2))  # Half the signal length
    w = np.fft.fftshift(
        np.linspace(0, 2 * np.pi - 2 * np.pi / Nsig, num=Nsig)
    )  # Frequency grid
    w[0:Mi] = -2 * np.pi + w[0:Mi]  # Shift the negative frequencies
    aw = abs(w)  # Absolute frequency values

    # Initialize the wavelet in the Fourier domain
    ymw = np.zeros(Nsig)

    # Parameters for the Meyer wavelet transitions
    an = 1.0 / (2 * gamma * wn)
    am = 1.0 / (2 * gamma * wm)
    pbn = (1.0 + gamma) * wn
    mbn = (1.0 - gamma) * wn
    pbm = (1.0 + gamma) * wm
    mbm = (1.0 - gamma) * wm

    # Construct the wavelet in the Fourier domain based on the segment boundaries
    for k in range(Nsig):
        if (aw[k] >= pbn) and (aw[k] <= mbm):  # Main passband
            ymw[k] = 1
        elif (aw[k] >= mbm) and (aw[k] <= pbm):  # Smooth transition in upper boundary
            ymw[k] = np.cos(np.pi * EWT_beta(am * (aw[k] - mbm)) / 2)
        elif (aw[k] >= mbn) and (aw[k] <= pbn):  # Smooth transition in lower boundary
            ymw[k] = np.sin(np.pi * EWT_beta(an * (aw[k] - mbn)) / 2)

    # Shift the wavelet back to the normal order
    ymw = np.fft.ifftshift(ymw)

    return ymw
