# -*- coding: utf-8 -*-
"""
Adaptive Chirp Mode Decomposition (ACMD) and variants.

Includes:
- ACMD     — Adaptive Chirp Mode Decomposition
- BA_ACMD  — Bandwidth-aware Adaptive Chirp Mode Decomposition
- DD_ACMD  — Data-driven Adaptive Chirp Mode Decomposition
"""

from .acmd import ACMD
from .ba_acmd import BA_ACMD, BAACMD
from .dd_acmd import DD_ACMD, DDACMD

__all__ = [
    "ACMD",
    "BA_ACMD",
    "BAACMD",
    "DD_ACMD",
    "DDACMD",
]
