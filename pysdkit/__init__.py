"""
A Python library for signal decomposition algorithms.
"""

__version__ = "0.4.16"

__all__ = [
    "EMD",
    "EEMD",
    "CEEMDAN",
    "REMD",
    "MEMD",
    "TVF_EMD",
    "FAEMD",
    "EMD2D",
    "HVD",
    "ITD",
    "LMD",
    "RLMD",
    "SSA",
    "vmd",
    "VMD",
    "ACMD",
    "MVMD",
    "VME",
    "CVMD2D",
    "VNCMD",
    "INCMD",
    "ewt",
    "EWT",
    "JMD",
    "Moving_Decomp",
    "data",
    "entropy",
    "hht",
    "plot",
    "tsa",
    "utils",
    "__version__",
]

# Empirical Mode Decomposition
from ._emd import EMD

# Ensemble Empirical Mode Decomposition
from ._emd import EEMD

# Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
from ._emd import CEEMDAN

# Robust Empirical Mode Decomposition
from ._emd import REMD

# Multivariate Empirical Mode Decomposition
from ._emd import MEMD

# Time Varying Filter based Empirical Mode Decomposition
from ._emd import TVF_EMD

# Fast and Adaptive Empirical Mode Decomposition
from ._faemd import FAEMD

# Empirical Mode Decomposition 2D for images
from ._emd2d import EMD2D

# Hilbert Vibration Decomposition
from ._hvd import HVD

# Intrinsic Time-Scale Decomposition
from ._itd import ITD

# Local Mean Decomposition
from ._lmd import LMD

# Robust Local Mean Decomposition
from ._lmd import RLMD

# Singular Spectral Analysis (SSA) algorithm
from ._ssa import SSA

# Variational mode decomposition
from ._vmd import vmd, VMD

# Adaptive Chirp Mode Decomposition
from ._vmd import ACMD

# Multivariate Variational mode decomposition
from ._vmd import MVMD

# Variational Mode Extraction, to extract a specific mode from the signal
from ._vmd import VME

# Variational Mode Decomposition for 2D Image
from ._vmd2d import VMD2D

# Compact Variational Mode Decomposition for 2D Images
from ._vmd2d import CVMD2D

# Variational Nonlinear Chirp Mode Decomposition
from ._vncmd import VNCMD

# Iterative nonlinear chirp mode decomposition
from ._vncmd import INCMD

# Empirical Wavelet Transform
from ._ewt import ewt, EWT

# Jump Plus AM-FM Mode Decomposition
from ._jmd import JMD

# Moving Average decomposition
from .tsa import Moving_Decomp
