# -*- coding: utf-8 -*-
"""
Time Iterative Decomposition (TID).

Algorithms that peel modes by iterative / sequential time-domain procedures
(rather than variational or filter-bank formulations in the frequency domain).

Includes:
- ALIF  — Adaptive Local Iterative Filtering
- FMD   — Feature Mode Decomposition
- HVD   — Hilbert Vibration Decomposition
- ITD   — Intrinsic Time-Scale Decomposition
- SSA   — Singular Spectral Analysis
"""

from .alif import ALIF
from .fmd import CK, FMD, TT, max_IJ
from .iterative_filtering import IterativeFiltering
from .hvd import HVD
from .itd import ITD
from .ssa import SSA

__all__ = [
    "ALIF",
    "CK",
    "FMD",
    "TT",
    "HVD",
    "ITD",
    "SSA",
    "IterativeFiltering",
    "max_IJ",
]
