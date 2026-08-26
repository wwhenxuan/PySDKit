# -*- coding: utf-8 -*-
"""
Blind deconvolution with on-the-fly period estimation.

Ports of IMCKD, ACYCBD and SMHD from the MATLAB pack
"Period estimation of deconvolution (MCKD, CYCBD, SMHD)".
They recover a sparse / cyclostationary source from
``x = h * s + n`` by learning an inverse FIR; the cycle is
estimated from the Hilbert envelope each iteration.

These are feature extractors, not EMD-style decomposers.
"""

from ._common import (
    analytic_envelope,
    as_real_1d,
    demean,
    ehps,
    envelope_spectrum,
    estimate_period,
    first_zero_crossing,
    matlab_kurtosis,
    matlab_round,
    peak_frequency,
    xcorr_coeff,
)
from ._imckd import correlated_kurtosis, delay_tensor, imckd
from ._acycbd import acycbd, corr_matrix, periodic
from ._smhd import smhd, sparse_map
from ._plot import (
    annotate_harmonics,
    harmonic_label,
    harmonic_peaks,
    marked_envelope_spectrum,
)

__all__ = [
    "analytic_envelope",
    "as_real_1d",
    "demean",
    "ehps",
    "envelope_spectrum",
    "estimate_period",
    "first_zero_crossing",
    "matlab_kurtosis",
    "matlab_round",
    "peak_frequency",
    "xcorr_coeff",
    "delay_tensor",
    "correlated_kurtosis",
    "imckd",
    "corr_matrix",
    "periodic",
    "acycbd",
    "sparse_map",
    "smhd",
    "harmonic_label",
    "harmonic_peaks",
    "annotate_harmonics",
    "marked_envelope_spectrum",
]
