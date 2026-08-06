# -*- coding: utf-8 -*-
"""
Created on 2025/07/31
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com

Generalized Dispersion Mode Decomposition (GDMD),
Variational Generalized Nonlinear Mode Decomposition (VGNMD),
Improved VGNMD (IVGNMD),
and Adaptive Generalized Dispersive Mode Decomposition (AGDMD / AGNCMD).
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

from .ivgnmd import (
    IVGNMD,
    ivgnmd,
    atffc_ivgnmd,
    se,
    tfsc,
    tfst,
    tfptd,
    mtdc_ridge,
    voa_ivgnmd,
    make_ivgnmd_demo_signal,
)

from .agncmd import (
    AGNCMD,
    AGDMD,
    agncmd,
    agdmd,
    agdi,
    ddgdi,
    if_dn,
    tvlp,
    low_filter,
    findev,
    arccos_phase,
    bandwidth_estimation,
    dispersion_compensation,
    agdmd_core,
    make_agncmd_demo_signal,
    spectrum_to_time_agdmd,
    stft_agncmd,
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
    "IVGNMD",
    "ivgnmd",
    "atffc_ivgnmd",
    "se",
    "tfsc",
    "tfst",
    "tfptd",
    "mtdc_ridge",
    "voa_ivgnmd",
    "make_ivgnmd_demo_signal",
    "AGNCMD",
    "AGDMD",
    "agncmd",
    "agdmd",
    "agdi",
    "ddgdi",
    "if_dn",
    "tvlp",
    "low_filter",
    "findev",
    "arccos_phase",
    "bandwidth_estimation",
    "dispersion_compensation",
    "agdmd_core",
    "make_agncmd_demo_signal",
    "spectrum_to_time_agdmd",
    "stft_agncmd",
]
