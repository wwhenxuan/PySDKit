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
)
from .sst import (
    SST,
    sst,
    as_channels,
    bump_transfer,
    cwavelet_transform,
    frequency_axis_linear,
    invert_sst_band,
    morlet_transfer,
    multivariate_bandwidth,
    parse_wavelet,
    reflect_pad,
    selective_cwt_mask,
    sst_wavelet_linear,
    universal_threshold,
)
from .set import (
    SET,
    set,
    brevridge,
    brevridge_mult,
    frequency_axis_set,
    gaussian_window,
    odd_window_length,
    reconstruct_from_ridges,
    set_transform,
    stft_gaussian_pair,
    synchroextracting_operator,
)
from pysdkit.data._loaders import (
    load_dual_signal_noise,
    load_map2,
    load_single_nsignal,
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
    "SST",
    "sst",
    "as_channels",
    "bump_transfer",
    "cwavelet_transform",
    "frequency_axis_linear",
    "invert_sst_band",
    "morlet_transfer",
    "multivariate_bandwidth",
    "parse_wavelet",
    "reflect_pad",
    "selective_cwt_mask",
    "sst_wavelet_linear",
    "universal_threshold",
    "SET",
    "set",
    "brevridge",
    "brevridge_mult",
    "frequency_axis_set",
    "gaussian_window",
    "odd_window_length",
    "reconstruct_from_ridges",
    "set_transform",
    "stft_gaussian_pair",
    "synchroextracting_operator",
    "load_dual_signal_noise",
    "load_single_nsignal",
    "load_map2",
]
