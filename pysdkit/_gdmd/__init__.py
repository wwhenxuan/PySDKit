# -*- coding: utf-8 -*-
"""
Created on 2025/07/31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Generalized Dispersion Mode Decomposition (GDMD) and
Variational Generalized Nonlinear Mode Decomposition (VGNMD).
"""

from .gdmd import (
    GDMD,
    gdmd,
    gdmd_core,
    curve_smooth,
    differ,
    make_dispersive_signal,
    spectrum_to_time,
    unilateral_spectrum,
    tf_spec_from_gd,
)

from .vgnmd import (
    VGNMD,
    vgnmd,
    atffc,
    mtdc,
    voa,
    stft_vgnmd,
    make_vgnmd_demo_signal,
    acmd_single,
)

__all__ = [
    "GDMD",
    "gdmd",
    "gdmd_core",
    "curve_smooth",
    "differ",
    "make_dispersive_signal",
    "spectrum_to_time",
    "unilateral_spectrum",
    "tf_spec_from_gd",
    "VGNMD",
    "vgnmd",
    "atffc",
    "mtdc",
    "voa",
    "stft_vgnmd",
    "make_vgnmd_demo_signal",
    "acmd_single",
]
