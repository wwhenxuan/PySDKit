"""
A Python library for signal decomposition algorithms.
"""

__version__ = "0.4.49"

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

# Serial Empirical Mode Decomposition
from ._emd import SEMD

# Time Varying Filter based Empirical Mode Decomposition
from ._emd import TVF_EMD

# Empirical Fourier Decomposition
from ._emd import EFD

# Fast and Adaptive Empirical Mode Decomposition
from ._faemd import FAEMD, FAEMD2D, FAEMD3D

# Empirical Mode Decomposition 2D for images
from ._emd2d import EMD2D

# Bidimensional Multivariate Empirical Mode Decomposition
from ._emd2d import BMEMD

# Hilbert Vibration Decomposition (time iterative decomposition)
from ._tid import HVD

# Intrinsic Time-Scale Decomposition (time iterative decomposition)
from ._tid import ITD

# Extreme-Point Symmetric Mode Decomposition
from ._emd import ESMD, esmd

# Local Mean Decomposition
from ._lmd import LMD

# Robust Local Mean Decomposition
from ._lmd import RLMD

# Singular Spectral Analysis (time iterative decomposition)
from ._tid import SSA

# Swarm Decomposition
from ._osd import SWD, swd

# Optimization-based Signal Decomposition
from ._osd import OSD

# Generalized Dispersion Mode Decomposition
from ._gdmd import GDMD, gdmd

# Variational Generalized Nonlinear Mode Decomposition
from ._gdmd import VGNMD, vgnmd

# Improved Variational Generalized Nonlinear Mode Decomposition
from ._gdmd import IVGNMD, ivgnmd

# Adaptive Generalized Dispersive Mode Decomposition (AGDMD / AGNCMD)
from ._gdmd import AGNCMD, agncmd, AGDMD, agdmd

# Variational Mode Decomposition
from ._vmd import vmd, VMD

# Adaptive Chirp Mode Decomposition
from ._acmd import ACMD

# Bandwidth-Aware Adaptive Chirp Mode Decomposition
from ._acmd import BA_ACMD

# Data-driven Adaptive Chirp Mode Decomposition
from ._acmd import DD_ACMD

# Multivariate Variational Mode Decomposition
from ._vmd import MVMD

# Variational Mode Extraction, to extract a specific mode from the signal
from ._vmd import VME

# Orthogonalized Variational Mode Decomposition
from ._vmd import OVMD

# Successive Variational Mode Decomposition
from ._vmd import svmd, SVMD

# Short-Time Variational Mode Decomposition
from ._vmd import stvmd, STVMD

# Variational Mode Decomposition for 2D Image
from ._vmd2d import VMD2D

# Compact Variational Mode Decomposition for 2D Images
from ._vmd2d import CVMD2D

# Variational Nonlinear Chirp Mode Decomposition
from ._vncmd import VNCMD

# Iterative Nonlinear Chirp Mode Decomposition
from ._vncmd import INCMD

# Adaptive Variational Nonlinear Chirp Mode Decomposition
from ._vncmd import AVNCMD

# Short-Time Narrow-Banded Mode Decomposition
from ._vncmd import STNBMD, stnbmd

# Adaptive Local Iterative Filtering (time iterative decomposition)
from ._tid import ALIF

# Variational Time-Frequency Mode Tracking Decomposition (TFA)
from ._tfa import VTFMTD, vtfmtd

# Adaptive Polymorphic Mode Decomposition / Impulsive Mode Decomposition
from ._imd import APMD, IMD, imd

# Empirical Wavelet Transform
from ._ewt import ewt, EWT

# Empirical Wavelet Transform for 2D Images
from ._ewt import ewt2d, EWT2D

# Jump Plus AM-FM Mode Decomposition
from ._jmd import JMD

# Multivariate Jump Plus AM-FM Mode Decomposition
from ._jmd import MJMD

# Successive Jump and Mode Decomposition (univariate / multivariate)
from ._jmd import SJMD, SMJMD

# Feature Mode Decomposition
from ._tid import FMD

# Moving Average Decomposition
from .tsa import Moving_Decomp

# Seasonal-Trend decomposition using LOESS (STL)
from .tsa import STL

# Multiple Seasonal-Trend decomposition using LOESS (MSTL)
from .tsa import MSTL

# Hilbert-Huang Transform
from ._emd import HHT


def greet():
    print(
        r"""
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
    print(
        """
_______________________________________________________________
Algorithm Name                                  | Abbreviation
_______________________________________________________________
Empirical Mode Decomposition                    |    EMD
Ensemble Empirical Mode Decomposition           |    EEMD     
Complete Ensemble EMD with Adaptive Noise       |    CEEMDAN
Robust Empirical Mode Decomposition             |    REMD
Multivariate Empirical Mode Decomposition       |    MEMD
Serial Empirical Mode Decomposition             |    SEMD
Time Varying Filter based EMD                   |    TVF_EMD
Empirical Fourier Decomposition                 |    EFD
Fast and Adaptive Empirical Mode Decomposition  |    FAEMD
Bidimensional FAEMD                             |    FAEMD2D
Tridimensional FAEMD                            |    FAEMD3D
Empirical Mode Decomposition 2D for images      |    EMD2D
Bidimensional Multivariate EMD                  |    BMEMD
Hilbert Vibration Decomposition                 |    HVD
Intrinsic Time-Scale Decomposition              |    ITD
Extreme-Point Symmetric Mode Decomposition      |    ESMD
Local Mean Decomposition                        |    LMD
Robust Local Mean Decomposition                 |    RLMD
Singular Spectral Analysis                      |    SSA
Swarm Decomposition                             |    SWD
Optimization-based Signal Decomposition         |    OSD
Generalized Dispersion Mode Decomposition       |    GDMD
Variational Generalized Nonlinear Mode Dec.     |    VGNMD
Improved VGNMD (crossed chirp / dispersive)     |    IVGNMD
Adaptive Generalized Dispersive Mode Dec.       |    AGNCMD / AGDMD
Variational Mode Decomposition                  |    VMD
Multivariate Variational Mode Decomposition     |    MVMD
Variational Mode Extraction                     |    VME
Orthogonalized Variational Mode Decomposition   |    OVMD
Successive Variational Mode Decomposition       |    SVMD
Short-Time Variational Mode Decomposition       |    STVMD
Variational Mode Decomposition for 2D Image     |    VMD2D
Compact VMD for 2D Image                        |    CVMD2D
Variational Nonlinear Chirp Mode Decomposition  |    VNCMD
Iterative Nonlinear Chirp Mode Decomposition    |    INCMD
Adaptive Variational Nonlinear Chirp Mode Dec.  |    AVNCMD
Short-Time Narrow-Banded Mode Decomposition     |    STNBMD
Adaptive Local Iterative Filtering              |    ALIF
Variational TF Mode Tracking Decomposition      |    VTFMTD
Adaptive Polymorphic Mode Decomposition         |    APMD
Impulsive Mode Decomposition                    |    IMD
Empirical Wavelet Transform                     |    EWT
Empirical Wavelet Transform for 2D Image        |    EWT2D
Jump Plus AM-FM Mode Decomposition              |    JMD
Multivariate Jump Plus AM-FM Mode Decomposition |    MJMD
Successive Jump and Mode Decomposition          |    SJMD / SMJMD
Feature Mode Decomposition                      |    FMD
Moving Average Decomposition                    |    Moving
Seasonal-Trend decomposition using LOESS        |    STL
Multiple Seasonal-Trend decomposition (LOESS)   |    MSTL
Hilbert-Huang Transform                         |    HHT
_______________________________________________________________
"""
    )


__all__ = [
    "EMD",
    "EEMD",
    "CEEMDAN",
    "REMD",
    "MEMD",
    "SEMD",
    "TVF_EMD",
    "EFD",
    "FAEMD",
    "FAEMD2D",
    "FAEMD3D",
    "EMD2D",
    "BMEMD",
    "HVD",
    "ITD",
    "ESMD",
    "esmd",
    "LMD",
    "RLMD",
    "SSA",
    "SWD",
    "swd",
    "OSD",
    "GDMD",
    "gdmd",
    "VGNMD",
    "vgnmd",
    "IVGNMD",
    "ivgnmd",
    "AGNCMD",
    "agncmd",
    "AGDMD",
    "agdmd",
    "vmd",
    "VMD",
    "ACMD",
    "BA_ACMD",
    "DD_ACMD",
    "MVMD",
    "VME",
    "OVMD",
    "svmd",
    "SVMD",
    "stvmd",
    "STVMD",
    "VMD2D",
    "CVMD2D",
    "VNCMD",
    "INCMD",
    "AVNCMD",
    "STNBMD",
    "stnbmd",
    "ALIF",
    "VTFMTD",
    "vtfmtd",
    "APMD",
    "IMD",
    "imd",
    "ewt",
    "EWT",
    "ewt2d",
    "EWT2D",
    "JMD",
    "MJMD",
    "SJMD",
    "SMJMD",
    "FMD",
    "Moving_Decomp",
    "STL",
    "MSTL",
    "models",
    "data",
    "entropy",
    "HHT",
    "plot",
    "tsa",
    "utils",
    "greet",
    "print_functions",
    "__version__",
]
