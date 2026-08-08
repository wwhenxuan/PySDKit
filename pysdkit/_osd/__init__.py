# -*- coding: utf-8 -*-
"""
Optimization / swarm based signal decomposition utilities.

- :class:`SWD` — Swarm Decomposition
- :class:`OSD` — Optimization-based Signal Decomposition
"""

from .swd import SWD, swd
from .osd import OSD
from .components import (
    Component,
    FiniteSet,
    MeanSquareSmall,
    SmoothDiff,
    SmoothSecondDifference,
    Sparse,
    SparseDiff,
    SparseFirstDiffConvex,
    SparseSecondDiffConvex,
)

__all__ = [
    "SWD",
    "swd",
    "OSD",
    "Component",
    "MeanSquareSmall",
    "SmoothDiff",
    "SmoothSecondDifference",
    "SparseDiff",
    "SparseFirstDiffConvex",
    "SparseSecondDiffConvex",
    "Sparse",
    "FiniteSet",
]
