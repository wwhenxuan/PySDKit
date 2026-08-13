# -*- coding: utf-8 -*-
"""
Time-Frequency Analysis (TFA).

Methods that operate primarily on time-frequency representations
(e.g. STFT) for mode separation and instantaneous-frequency tracking.
"""

from .vtfmtd import (
    VTFMTD,
    vtfmtd,
    stft,
    frequency_axis,
    bin_index_grid,
    expand_omega_init,
    first_difference_gram,
    estimate_if_centroid,
    smooth_if,
    moving_average_if,
    omega_bins_to_hz,
    load_dual_signal_noise,
    load_single_nsignal,
    load_map2,
)

__all__ = [
    "VTFMTD",
    "vtfmtd",
    "stft",
    "frequency_axis",
    "bin_index_grid",
    "expand_omega_init",
    "first_difference_gram",
    "estimate_if_centroid",
    "smooth_if",
    "moving_average_if",
    "omega_bins_to_hz",
    "load_dual_signal_noise",
    "load_single_nsignal",
    "load_map2",
]
