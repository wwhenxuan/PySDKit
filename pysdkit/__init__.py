"""
A Python library for signal decomposition algorithms.
"""

__version__ = "0.4.17"

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

# Empirical Fourier Decomposition
from ._emd import EFD

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

# Singular Spectral Analysis
from ._ssa import SSA

# Variational Mode Decomposition
from ._vmd import vmd, VMD

# Adaptive Chirp Mode Decomposition
from ._vmd import ACMD

# Multivariate Variational Mode Decomposition
from ._vmd import MVMD

# Variational Mode Extraction, to extract a specific mode from the signal
from ._vmd import VME

# Variational Mode Decomposition for 2D Image
from ._vmd2d import VMD2D

# Compact Variational Mode Decomposition for 2D Images
from ._vmd2d import CVMD2D

# Variational Nonlinear Chirp Mode Decomposition
from ._vncmd import VNCMD

# Iterative Nonlinear Chirp Mode Decomposition
from ._vncmd import INCMD

# Empirical Wavelet Transform
from ._ewt import ewt, EWT

# Jump Plus AM-FM Mode Decomposition
from ._jmd import JMD

# Moving Average Decomposition
from .tsa import Moving_Decomp


def greet():
    print(
        """
 ____          ____   ____   _  __ _  _   
|  _ \  _   _ / ___| |  _ \ | |/ /(_)| |_ 
| |_) || | | |\___ \ | | | || ' / | || __|
|  __/ | |_| | ___) || |_| || . \ | || |_ 
|_|     \__, ||____/ |____/ |_|\_\|_| \__|
        |___/                                            
    
A Python library for signal decomposition algorithms.
https://github.com/wwhenxuan/PySDKit
"""
    )


def print_functions():
    """"""
    print("""
_______________________________________________________________
Algorithm Name                                  | Abbreviation
_______________________________________________________________
Empirical Mode Decomposition                    |    EMD
Ensemble Empirical Mode Decomposition           |    EEMD     
Complete Ensemble EMD with Adaptive Noise       |    CEEMDAN
Robust Empirical Mode Decomposition             |    REMD
Multivariate Empirical Mode Decomposition       |    MEMD
Time Varying Filter based EMD                   |    TVF_EMD
Empirical Fourier Decomposition                 |    EFD
Fast and Adaptive Empirical Mode Decomposition  |    FAEMD
Empirical Mode Decomposition 2D for images      |    EMD2D
Hilbert Vibration Decomposition                 |    HVD
Intrinsic Time-Scale Decomposition              |    ITD
Local Mean Decomposition                        |    LMD
Robust Local Mean Decomposition                 |    RLMD
Singular Spectral Analysis                      |    SSA
Variational Mode Decomposition                  |    VMD
Multivariate Variational Mode Decomposition     |    MVMD
Variational Mode Extraction                     |    VME
Variational Mode Decomposition for 2D Image     |    VMD2D
Compact VMD for 2D Image                        |    CVMD2D
Variational Nonlinear Chirp Mode Decomposition  |    VNCMD
Iterative Nonlinear Chirp Mode Decomposition    |    INCMD
Empirical Wavelet Transform                     |    EWT
Jump Plus AM-FM Mode Decomposition              |    JMD
Moving Average Decomposition                    |    Moving
_______________________________________________________________
""")


__all__ = [
    "EMD",
    "EEMD",
    "CEEMDAN",
    "REMD",
    "MEMD",
    "TVF_EMD",
    "EFD",
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
    "greet",
    "print_functions",
    "__version__",
]
