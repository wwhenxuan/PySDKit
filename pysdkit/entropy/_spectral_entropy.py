# -*- coding: utf-8 -*-
"""
Spectral (periodogram) entropy of a 1-D series.
"""

import numpy as np
from typing import Optional

from pysdkit.entropy._coarse_grain import as_1d, shannon


def spectral_entropy(
    y: np.ndarray,
    n_fft: Optional[int] = None,
    normalize: bool = True,
) -> float:
    """
    Spectral entropy of a 1-D series.

    The one-sided periodogram is normalized to a probability mass, then
    Shannon entropy is computed.  Oscillatory modes concentrate power in
    few bins (low entropy); broadband noise fills the spectrum (high
    entropy).

    Powell, G. E., & Percival, I. C. (1979). A spectral entropy method
    for distinguishing regular and irregular motion of Hamiltonian
    systems. Journal of Physics A, 12(11), 2053.

    Inouye, T., et al. (1991). Quantification of EEG irregularity by use
    of the entropy of the power spectrum. Electroencephalography and
    Clinical Neurophysiology, 79(3), 204-210.

    :param y: 1-D input series.
    :param n_fft: FFT length (default ``len(y)``).
    :param normalize: If True, divide by ``log(n_bins)`` so the value
                      lies in ``[0, 1]`` for a full-band spectrum.
    :return: Spectral entropy (nats if ``normalize`` is False).
    """
    y = as_1d(y)
    if len(y) < 2:
        raise ValueError("Signal must have at least two points.")
    if n_fft is None:
        n_fft = len(y)
    if n_fft < 2:
        raise ValueError("n_fft must be an integer > 1.")

    power = np.abs(np.fft.rfft(y, n=n_fft)) ** 2
    total = np.sum(power)
    if total <= 0.0:
        return 0.0
    prob = power / total
    value = shannon(prob)
    if normalize:
        n_bins = max(prob.size, 2)
        value /= np.log(n_bins)
    return float(value)
