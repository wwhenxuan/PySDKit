# -*- coding: utf-8 -*-
"""
Extreme-Point Symmetric Mode Decomposition (ESMD).
"""

from .esmd import (
    ESMD,
    esmd,
    find_extrema,
    mean_curve,
    sift_mode,
    decompose_fixed_sift,
    variance_ratio,
    scan_variance_ratios,
    instantaneous_amplitude,
    instantaneous_frequency,
    total_energy,
    make_esmd_example3,
    load_wind_demo,
)

__all__ = [
    "ESMD",
    "esmd",
    "find_extrema",
    "mean_curve",
    "sift_mode",
    "decompose_fixed_sift",
    "variance_ratio",
    "scan_variance_ratios",
    "instantaneous_amplitude",
    "instantaneous_frequency",
    "total_energy",
    "make_esmd_example3",
    "load_wind_demo",
]
