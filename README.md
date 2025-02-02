# PySDKit: signal decomposition in Python 

<div align="center">

[![PyPI version](https://badge.fury.io/py/PySDKit.svg)](https://pypi.org/project/PySDKit/) 
![License](https://img.shields.io/github/license/wwhenxuan/PySDKit)
[![codecov](https://codecov.io/gh/wwhenxuan/PySDKit/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/wwhenxuan/PySDKit)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-brightgreen.svg)](https://arxiv.org/abs/1234.56789) 
[![Downloads](https://pepy.tech/badge/pysdkit)](https://pepy.tech/project/pysdkit)
![Visits Badge](https://badges.pufler.dev/visits/ForestsKing/D3R)

A Python library for signal decomposition algorithms 🥳

<img src="https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/images/Logo_sd.png" alt="Logo_sd" width="500"/>

</div>

## Installation 🚀

You can install `PySDKit` through pip:

~~~
pip install pysdkit
~~~

We only used [`NumPy`](https://numpy.org/), [`Scipy`](https://scipy.org/) and [`matplotlib`](https://matplotlib.org/) when developing the project.

## Example script ✨

This project integrates simple signal processing methods, signal decomposition and visualization, and builds a general interface similar to [`Scikit-learn`](https://scikit-learn.org/stable/). It is mainly divided into three steps:
1. Import the signal decomposition method;
2. Create an instance for signal decomposition;
3. Use the `fit_transform` method to implement signal decomposition;
4. Visualize and analyze the original signal and the intrinsic mode functions IMFs obtained by decomposition.

~~~python
from pysdkit import EMD
from pysdkit.data import test_emd
from pysdkit.plot import plot_IMFs

t, signal = test_emd()

# create an instance for signal decomposition
emd = EMD()
# implement signal decomposition
IMFs = emd.fit_transform(signal, max_imf=2)
plot_IMFs(signal, IMFs)
~~~

![example](https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/images/example.jpg)

The EMD in the above example is the most classic [`empirical mode decomposition`](https://www.mathworks.com/help/signal/ref/emd.html) algorithm in signal decomposition. For more complex signals, you can try other algorithms such as variational mode decomposition ([`VMD`](https://ieeexplore.ieee.org/abstract/document/6655981)).

~~~python
import numpy as np
from pysdkit import VMD

# load new signal
signal = np.load("./example/example.npy")

# use variational mode decomposition
vmd = VMD(alpha=500, K=3, tau=0.0, tol=1e-9)
IMFs = vmd.fit_transform(signal=signal)
print(IMFs.shape)

vmd.plot_IMFs(save_figure=True)
~~~

![vmd_example](https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/images/vmd_example.jpg)

Better observe the characteristics of the decomposed intrinsic mode function in the frequency domain.

~~~python
from pysdkit.plot import plot_IMFs_amplitude_spectra

# frequency domain visualization
plot_IMFs_amplitude_spectra(IMFs, smooth="exp")   # use exp smooth
~~~

![frequency_example](https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/images/frequency_example.jpg)

## Target 🎯

|                          Algorithm                           |                            Paper                             |                             Code                             | State |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :---: |
|              EMD (Empirical Mode Decomposition)              |                         [[paper]]()                          | [[code]](https://www.mathworks.com/help/signal/ref/emd.html) |   ✔️   |
|       MEMD (Multivariate Empirical Mode Decomposition)       | [[paper]](https://royalsocietypublishing.org/doi/full/10.1098/rspa.2009.0502) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd) |   ✖️   |
|      BEMD (Bidimensional Empirical Mode Decomposition)       | [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0262885603000945) | [[code]](https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/BEMD.py) |   ✖️   |
|         CEMD (Complex Empirical Mode Decomposition)          | [[paper]](https://ieeexplore.ieee.org/abstract/document/4063369) |                          [[code]]()                          |   ✖️   |
|         EEMD (Ensemble Empirical Mode Decomposition)         | [[paper]](https://www.sciencedirect.com/topics/physics-and-astronomy/ensemble-empirical-mode-decomposition) | [[code]](https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EEMD.py) |   ✖️   |
|          REMD (Robust Empirical Mode Decomposition)          | [[paper]](https://www.sciencedirect.com/science/article/pii/S0019057821003785) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/70032-robust-empirical-mode-decomposition-remd) |   ✖️   |
| BMEMD (Bidimensional Multivariate Empirical Mode Decomposition) |   [[paper]](https://ieeexplore.ieee.org/document/8805082)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/72343-bidimensional-multivariate-empirical-mode-decomposition?s_tid=FX_rc1_behav) |   ✖️   |
|     CEEMDAN (Complete Ensemble EMD with Adaptive Noise)      |   [[paper]](https://ieeexplore.ieee.org/document/5947265)    | [[code]](https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EEMD.py) |   ✖️   |
|           TVF_EMD (Time Varying Filter Based EMD)            | [[paper]](https://www.sciencedirect.com/science/article/pii/S0165168417301135) |   [[code]](https://github.com/stfbnc/pytvfemd/tree/master)   |   ✔️   |
|                FAEMD (Fast and Adaptive EMD)                 |   [[paper]](https://ieeexplore.ieee.org/document/8447300)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd) |   ✖️   |
|        MV_FAEMD (Multivariate Fast and Adaptive EMD)         |   [[paper]](https://ieeexplore.ieee.org/document/8447300)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd) |   ✖️   |
|      MD_FAEMD (Multidimensional Fast and Adaptive EMD)       |   [[paper]](https://ieeexplore.ieee.org/document/8447300)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd) |   ✖️   |
|                LMD (Local Mean Decomposition)                | [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S1051200418308133?via%3Dihub) | [[code]](https://github.com/shownlin/PyLMD/blob/master/PyLMD/LMD.py) |   ✔️   |
|            RLMD (Robust Local Mean Decomposition)            | [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0888327017301619) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/66935-robust-local-mean-decomposition-rlmd) |   ✖️   |
|               FMD (Feature Mode Decomposition)               |   [[paper]](https://ieeexplore.ieee.org/document/9732251)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/108099-feature-mode-decomposition-fmd) |   ✖️   |
|              EWT (Empirical Wavelet Transform)               |   [[paper]](https://ieeexplore.ieee.org/document/6522142)    | [[code]](https://www.mathworks.com/help/wavelet/ug/empirical-wavelet-transform.html) |   ✔️   |
|            EWT2D (2D Empirical Wavelet Transform)            |         [[paper]](https://arxiv.org/abs/2405.06188)          |        [[code]](https://github.com/bhurat/EWT-Python)        |   ✖️   |
|             VMD (Variational Mode Decomposition)             |   [[paper]](https://ieeexplore.ieee.org/document/6655981)    |          [[code]](https://github.com/vrcarva/vmdpy)          |   ✔️   |
|      MVMD (Multivariate Variational Mode Decomposition)      |   [[paper]](https://ieeexplore.ieee.org/document/8890883)    |          [[code]](https://github.com/yunyueye/MVMD)          |   ✔️   |
|    VMD2D (Two-Dimensional Variational Mode Decomposition)    | [[paper]](https://ww3.math.ucla.edu/camreport/cam14-16.pdf)  | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/45918-two-dimensional-variational-mode-decomposition?s_tid=srchtitle) |   ✖️   |
|       SVMD (Successive Variational Mode Decomposition)       | [[paper]](https://www.sciencedirect.com/science/article/pii/S0165168420301535) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/98649-successive-variational-mode-decomposition-svmd-m?s_tid=FX_rc3_behav) |   ✖️   |
|    VNCMD (Variational Nonlinear Chirp Mode Decomposition)    |   [[paper]](https://ieeexplore.ieee.org/document/7990179)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/64292-variational-nonlinear-chirp-mode-decomposition) |   ✔️   |
|   MNCMD (Multivariate Nonlinear Chirp Mode Decomposition)    | [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0165168420302103) |                          [[code]]()                          |   ✖️   |
| AVNCMD (Adaptive Variational Nonlinear Chirp Mode Decomposition) | [[paper]](https://ieeexplore.ieee.org/abstract/document/9746147) |         [[code]](https://github.com/HauLiang/AVNCMD)         |   ✖️   |
|           ACMD (Adaptive Chirp Mode Decomposition)           |                         [[paper]]()                          | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/121373-data-driven-adaptive-chirp-mode-decomposition?s_tid=srchtitle) |   ✔️   |
| BA-ACMD (Bandwidth-aware adaptive chirp mode decomposition)  |                         [[paper]]()                          | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/132792-bandwidth-aware-adaptive-chirp-mode-decomposition-ba-acmd?s_tid=srchtitle) |   ✖️   |
|           JMD (Jump Plus AM-FM Mode Decomposition)           |         [[paper]](https://arxiv.org/abs/2407.07800)          | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/169388-jump-plus-am-fm-mode-decomposition-jmd?s_tid=prof_contriblnk) |   ✖️   |
|    MJMD (Multivariate Jump Plus AM-FM Mode Decomposition)    |         [[paper]](https://arxiv.org/abs/2407.07800)          | [[code]](https://ms-intl.mathworks.com/matlabcentral/fileexchange/169393-multivariate-jump-plus-am-fm-mode-decomposition-mjmd?s_tid=FX_rc2_behav) |   ✖️   |
|      ESMD (Extreme-Point Symmetric Mode Decomposition)       |          [[paper]](https://arxiv.org/abs/1303.6540)          |         [[code]](https://github.com/WuShichao/esmd)          |   ✖️   |
