# -*- coding: utf-8 -*-
"""
Underdetermined blind source separation (BSS).

Port of Yu's MATLAB pack (originally named YGBSS): frequency-energy
clustering of the STFT, binary masks, inverse STFT, and the SDOF /
MAC helpers used in the modal-identification examples.

These are feature extractors / separators, not EMD-style decomposers.
"""

from ._stft import (
    default_window_length,
    frequency_axis_stft,
    hamming_window,
    matlab_round,
    odd_window_length,
    padding_line,
    tfristft,
    tfrstft,
    tfrstft_uniform,
)
from ._bss import (
    BSS,
    bss,
    cosine_distance,
    cosine_masks,
    frequency_energy,
    peakdet,
)
from ._modal import (
    modal_assurance_criterion,
    mrsp2mpfd,
    sdof_local,
    sign_from_correlation,
)

__all__ = [
    "matlab_round",
    "odd_window_length",
    "hamming_window",
    "default_window_length",
    "frequency_axis_stft",
    "tfrstft",
    "tfrstft_uniform",
    "tfristft",
    "padding_line",
    "peakdet",
    "frequency_energy",
    "cosine_distance",
    "cosine_masks",
    "BSS",
    "bss",
    "modal_assurance_criterion",
    "sign_from_correlation",
    "sdof_local",
    "mrsp2mpfd",
]
