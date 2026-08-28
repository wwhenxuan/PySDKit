<div align="center">

# PySDKit: signal decomposition in Python

[![PyPI version](https://badge.fury.io/py/PySDKit.svg)](https://pypi.org/project/PySDKit/) 
[![Documentation Status](https://readthedocs.org/projects/pysdkit/badge/?version=latest)](https://pysdkit.readthedocs.io/en/latest/)
![License](https://img.shields.io/github/license/wwhenxuan/PySDKit)
[![Python](https://img.shields.io/badge/python-3.8+-blue?logo=python)](https://www.python.org/)
[![Downloads](https://pepy.tech/badge/pysdkit)](https://pepy.tech/project/pysdkit)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python library for signal decomposition algorithms 🥳

[Installation](#Installation) |
[Example Script](#Example-Script) |
[Target](#Target) |
[Acknowledgements](#Acknowledgements)

<img src="https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/images/Logo_sd.png" alt="Logo_sd" width="500"/>

</div>

## Installation 🚀 <a id="Installation"></a>

You can install `PySDKit` through pip:

~~~
pip install pysdkit
~~~

We only used [`NumPy`](https://numpy.org/), [`Scipy`](https://scipy.org/) and [`matplotlib`](https://matplotlib.org/) when developing the project.

## Example Script ✨ <a id="Example-Script"></a>

This project integrates simple signal processing methods, signal decomposition and visualization, and builds a general interface similar to [`Scikit-learn`](https://scikit-learn.org/stable/). It is mainly divided into three steps:
1. Import the signal decomposition method;
2. Create an instance for signal decomposition;
3. Use the `fit_transform` method to implement signal decomposition;
4. Visualize and analyze the original signal and the intrinsic mode functions IMFs obtained by decomposition.

~~~python
import numpy as np
from pysdkit import EMD
from pysdkit.plot import plot_IMFs

t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 5 * t) + 0.7 * np.sin(2 * np.pi * 25 * t) + 0.45 * np.sin(2 * np.pi * 80 * t)

emd = EMD()
IMFs = emd.fit_transform(signal, max_imfs=3)
plot_IMFs(signal, IMFs, view="2d_freq", fs=1000, freq_max=150)
~~~

![example](https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/images/decomposition_demo.jpg)

The EMD in the above example is the most classic [`empirical mode decomposition`](https://www.mathworks.com/help/signal/ref/emd.html) algorithm in signal decomposition. For more complex signals, you can try other algorithms such as variational mode decomposition ([`VMD`](https://ieeexplore.ieee.org/abstract/document/6655981)).

~~~python
from pysdkit import VMD
from pysdkit.data import test_vmd
from pysdkit.plot import plot_IMFs

t, signal, fs = test_vmd()

vmd = VMD(alpha=2000, K=4, tau=0.0, tol=1e-7)
IMFs = vmd.fit_transform(signal)
plot_IMFs(signal, IMFs, view="2d_freq", fs=fs, freq_max=fs / 2)
~~~

![vmd_example](https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/images/vmd_example.jpg)

For multichannel recordings, algorithms such as multivariate VMD ([`MVMD`](https://doi.org/10.1109/TSP.2019.2951223)) keep shared oscillations **mode-aligned** across channels:

~~~python
import numpy as np
from pysdkit import MVMD
from pysdkit.plot import plot_IMFs

t = np.arange(0, 1, 0.001)
# ch1: 2+36 Hz, ch2: 24+36 Hz, ch3: 80+36 Hz (36 Hz shared)
signal = np.vstack([
    np.cos(2*np.pi*2*t) + np.cos(2*np.pi*36*t),
    np.cos(2*np.pi*24*t) + np.cos(2*np.pi*36*t),
    np.cos(2*np.pi*80*t) + np.cos(2*np.pi*36*t),
])

mvmd = MVMD(alpha=2000, K=4, tau=0.0, init="uniform")
IMFs = mvmd.fit_transform(signal)   # shape: (K, T, C)
plot_IMFs(signal, IMFs)             # per-channel panels
~~~

![mvmd_example](https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/images/mvmd_example.jpg)

## Target 🎯 <a id="Target"></a>

`PySDKit` is still under development. We are currently working on reproducing the signal decomposition algorithms in the table below, including not only common decomposition algorithms for `univariate signals` such as EMD, VMD, OVMD, AVNCMD, ALIF and APMD, but also decomposition algorithms for `multivariate signals` such as MEMD and MVMD. We will also further reproduce the decomposition algorithms for `two-dimensional images` to make PySDKit not only suitable for signal processing, but also for image analysis and understanding. See [`Mission`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/README.md) for the reasons why we developed PySDKit.

|                          Algorithm                           |                            Paper                             |                            Source                            | Example |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----: |
| [`EMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/emd.py) (Empirical Mode Decomposition) | [[paper]](https://royalsocietypublishing.org/doi/abs/10.1098/rspa.1998.0193) | [[code]](https://www.mathworks.com/help/signal/ref/emd.html) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/emd/emd.ipynb) |
| [`EEMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/eemd.py) (Ensemble Empirical Mode Decomposition) | [[paper]](https://www.sciencedirect.com/topics/physics-and-astronomy/ensemble-empirical-mode-decomposition) | [[code]](https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EEMD.py) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/emd_variants/eemd.ipynb) |
| [`REMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/remd.py) (Robust Empirical Mode Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/pii/S0019057821003785) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/70032-robust-empirical-mode-decomposition-remd) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/emd_variants/remd.ipynb) |
| [`CEEMDAN`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/ceemdan.py) (Complete Ensemble EMD with Adaptive Noise) |   [[paper]](https://ieeexplore.ieee.org/document/5947265)    | [[code]](https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EEMD.py) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/emd_variants/ceemdan.ipynb) |
| [`TVF_EMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/tvf_emd.py) (Time Varying Filter Based EMD) | [[paper]](https://www.sciencedirect.com/science/article/pii/S0165168417301135) |   [[code]](https://github.com/stfbnc/pytvfemd/tree/master)   | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/emd_variants/tfv_emd.ipynb) |
| [`MEMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/memd.py) (Multivariate Empirical Mode Decomposition) | [[paper]](https://royalsocietypublishing.org/doi/full/10.1098/rspa.2009.0502) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/memd/memd.ipynb) |
| [`APITMEMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/apitmemd.py) (Adaptive-Projection Intrinsically Transformed MEMD) | [[paper]](https://www.commsp.ee.ic.ac.uk/~mandic/research/EMD_Stuff/AH_VG_DL_DPM_APITMEMD_RSTA_2016.pdf) | [[code]](https://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/memd/apitmemd.ipynb) |
| [`NSTEMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/nstemd.py) (Nonuniformly Sampled Trivariate EMD) | [[paper]](https://ieeexplore.ieee.org/document/7178660) | [[code]](https://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/emd_variants/nstemd.ipynb) |
| [`OnlineEMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/online_emd.py) (Online Empirical Mode Decomposition) | [[paper]](https://ieeexplore.ieee.org/document/7952969) | [[code]](https://github.com/romain-fontugne/onlineEMD) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/emd_variants/online_emd.ipynb) |
| [`EMD2D`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd2d/emd2d.py) (Empirical Mode Decomposition 2D for images) |          [[paper]](http://aquador.vovve.net/IEMD/)           |           [[code]](http://aquador.vovve.net/IEMD/)           | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/image/emd2d.ipynb) |
| [`BMEMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd2d/bmemd.py) (Bidimensional Multivariate Empirical Mode Decomposition) |   [[paper]](https://ieeexplore.ieee.org/document/8805082)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/72343-bidimensional-multivariate-empirical-mode-decomposition?s_tid=FX_rc1_behav) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/image/bmemd.ipynb) |
| [`FAEMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_faemd/faemd.py) (Fast and Adaptive EMD) |   [[paper]](https://ieeexplore.ieee.org/document/8447300)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/faemd/faemd.ipynb) |
| [`FAEMD2D`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_faemd/faemd2d.py) (Two-Dimensional Fast and Adaptive EMD) |   [[paper]](https://ieeexplore.ieee.org/document/8447300)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/faemd/faemd2d.ipynb) |
| [`FAEMD3D`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_faemd/faemd3d.py) (Three-Dimensional Fast and Adaptive EMD) |   [[paper]](https://ieeexplore.ieee.org/document/8447300)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/faemd/faemd3d.ipynb) |
| [`HVD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_tid/hvd.py) (Hilbert Vibration Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/pii/S0022460X06001556) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/178804-hilbert-vibration-decomposition?s_tid=FX_rc1_behav) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/temp_iter/hvd.ipynb) |
| [`ITD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_tid/itd.py) (Intrinsic Time-Scale Decomposition) | [[paper]](https://royalsocietypublishing.org/doi/10.1098/rspa.2006.1761) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/69380-intrinsic-time-scale-decomposition-itd) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/temp_iter/itd.ipynb) |
| [`ALIF`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_tid/alif.py) (Adaptive Local Iterative Filtering) |          [[paper]](https://arxiv.org/abs/1411.6051)          |           [[code]](https://github.com/Cicone/ALIF)           | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/temp_iter/alif.ipynb) |
| [`IMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_imd/imd.py) (Impulsive Mode Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/pii/S0888327024001250) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/182635-impulsive-mode-decomposition?s_tid=FX_rc3_behav) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/imd/imd.ipynb) |
| [`APMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_imd/apmd.py) (Adaptive Polymorphic Mode Decomposition) |     [[paper]](https://doi.org/10.1016/j.dsp.2024.104913)     | [[code]](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_imd/apmd.py) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/imd/apmd.ipynb) |
| [`LMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_lmd/lmd.py) (Local Mean Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S1051200418308133?via%3Dihub) | [[code]](https://github.com/shownlin/PyLMD/blob/master/PyLMD/LMD.py) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/lmd/lmd.ipynb) |
| [`RLMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_lmd/rlmd.py) (Robust Local Mean Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0888327017301619) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/66935-robust-local-mean-decomposition-rlmd) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/lmd/rlmd.ipynb) |
| [`FMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_tid/fmd.py) (Feature Mode Decomposition) |   [[paper]](https://ieeexplore.ieee.org/document/9732251)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/108099-feature-mode-decomposition-fmd) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/temp_iter/fmd.ipynb) |
| [`SSA`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_tid/ssa.py) (Singular Spectral Analysis) | [[paper]](https://orca.cardiff.ac.uk/id/eprint/15208/1/Zhiglavsky_SSA_encyclopedia.pdf) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/58967-singular-spectrum-analysis-beginners-guide) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/ssa/ssa.ipynb) |
| [`EWT`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_ewt/ewt.py) (Empirical Wavelet Transform) |   [[paper]](https://ieeexplore.ieee.org/document/6522142)    | [[code]](https://www.mathworks.com/help/wavelet/ug/empirical-wavelet-transform.html) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/ewt/ewt_ewt2d.ipynb) |
| [`EFD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_ewt/efd.py) (Empirical Fourier Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0888327021005355) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/97747-empirical-fourier-decomposition-efd) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/ewt/efd.ipynb) |
| [`EWT2D`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_ewt/ewt2d.py) (2D Empirical Wavelet Transform) |         [[paper]](https://arxiv.org/abs/2405.06188)          |        [[code]](https://github.com/bhurat/EWT-Python)        | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/ewt/ewt_ewt2d.ipynb) |
| [`VMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd/vmd_c.py) (Variational Mode Decomposition) |   [[paper]](https://ieeexplore.ieee.org/document/6655981)    |          [[code]](https://github.com/vrcarva/vmdpy)          | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/vmd/vmd.ipynb) |
| [`MVMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd/mvmd.py) (Multivariate Variational Mode Decomposition) |   [[paper]](https://ieeexplore.ieee.org/document/8890883)    |          [[code]](https://github.com/yunyueye/MVMD)          | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/vmd/mvmd.ipynb) |
| [`VMD2D`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd2d/vmd2d.py) (Two-Dimensional Variational Mode Decomposition) | [[paper]](https://ww3.math.ucla.edu/camreport/cam14-16.pdf)  | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/45918-two-dimensional-variational-mode-decomposition?s_tid=srchtitle) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/image/vmd2d.ipynb) |
| [`CVMD2D`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd2d/cvmd2d.py) (Two-Dimensional Compact Variational Mode Decomposition) | [[paper]](https://link.springer.com/article/10.1007/s10851-017-0710-z) | [[code]](https://ww2.mathworks.cn/matlabcentral/fileexchange/67285-two-dimensional-compact-variational-mode-decomposition-2d-tv-vmd?s_tid=FX_rc2_behav) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/image/cvmd2d.ipynb) |
| [`VME`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd/vme.py) (Variational Mode Extraction) |   [[paper]](https://ieeexplore.ieee.org/document/7997854)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/76003-variational-mode-extraction-vme-m?s_tid=srchtitle) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/vmd/vme.ipynb) |
| [`OVMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd/ovmd.py) (Orthogonalized Variational Mode Decomposition) |   [[paper]](https://doi.org/10.1016/j.sigpro.2025.110251)    | [[code]](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd/ovmd.py) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/vmd/ovmd.ipynb) |
| [`SVMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd/svmd.py) (Successive Variational Mode Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/pii/S0165168420301535) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/98649-successive-variational-mode-decomposition-svmd-m?s_tid=FX_rc3_behav) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/vmd/svmd.ipynb) |
| [`STVMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd/stvmd.py) (Short Time Variational Mode Decomposition) |   [[paper]](https://doi.org/10.1016/j.sigpro.2025.110203)    | [[code]](https://github.com/plustar/Short-Time-Variational-Mode-Decomposition) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/vmd/stvmd.ipynb) |
| [`VNCMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vncmd/vncmd.py) (Variational Nonlinear Chirp Mode Decomposition) |   [[paper]](https://ieeexplore.ieee.org/document/7990179)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/64292-variational-nonlinear-chirp-mode-decomposition) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/vncmd/vncmd.ipynb) |
| [`INCMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vncmd/incmd.py) (Iterative Nonlinear Chirp Mode Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/pii/S0022460X2030403X?via%3Dihub) |      [[code]](https://github.com/sheadan/IterativeNCMD)      | [[notebook]]() |
| [`AVNCMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vncmd/avncmd.py) (Adaptive Variational Nonlinear Chirp Mode Decomposition) | [[paper]](https://ieeexplore.ieee.org/abstract/document/9746147) |         [[code]](https://github.com/HauLiang/AVNCMD)         | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/vncmd/avncmd.ipynb) |
| [`ACMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_acmd/acmd.py) (Adaptive Chirp Mode Decomposition) |                         [[paper]]()                          | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/121373-data-driven-adaptive-chirp-mode-decomposition?s_tid=srchtitle) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/acmd/acmd.ipynb) |
| [`BA-ACMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_acmd/ba_acmd.py) (Bandwidth-aware Adaptive Chirp Mode Decomposition) | [[paper]](https://journals.sagepub.com/doi/abs/10.1177/14759217231174699) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/132792-bandwidth-aware-adaptive-chirp-mode-decomposition-ba-acmd?s_tid=srchtitle) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/acmd/ba_acmd.ipynb) |
| [`DD-ACMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_acmd/dd_acmd.py) (Data-driven Adaptive Chirp Mode Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0888327022010652) | [[code]](https://ww2.mathworks.cn/matlabcentral/fileexchange/121373-data-driven-adaptive-chirp-mode-decomposition) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/acmd/dd_acmd.ipynb) |
| [`GDMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_gdmd/gdmd.py) (Generalized Dispersion Mode Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/pii/S0022460X20306295) | [[code]](https://www.sciencedirect.com/science/article/pii/S0022460X20306295) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/gdmd/gdmd.ipynb) |
| [`VGNMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_gdmd/vgnmd.py) (Variational Generalized Nonlinear Mode Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/pii/S088832702300821X) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/154792-variational-generalized-nonlinear-mode-decomposition-vgmnd) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/gdmd/vgnmd.ipynb) |
| [`IVGNMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_gdmd/ivgnmd.py) (Improved Variational Generalized Nonlinear Mode Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/pii/S0888327025001086) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/180043-improved-vgnmd) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/gdmd/ivgnmd.ipynb) |
| [`AGNCMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_gdmd/agncmd.py) (Adaptive Generalized Dispersive Mode Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/pii/S0022460X25004031) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/181499-adaptive-generalized-dispersive-mode-decomposition-agdmd) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/gdmd/agncmd.ipynb) |
| [`JMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_jmd/jmd.py) (Jump Plus AM-FM Mode Decomposition) |         [[paper]](https://arxiv.org/abs/2407.07800)          | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/169388-jump-plus-am-fm-mode-decomposition-jmd?s_tid=prof_contriblnk) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/jmd/jmd.ipynb) |
| [`MJMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_jmd/mjmd.py) (Multivariate Jump Plus AM-FM Mode Decomposition) |         [[paper]](https://arxiv.org/abs/2407.07800)          | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/169393-multivariate-jump-plus-am-fm-mode-decomposition-mjmd?s_tid=prof_contriblnk) | [[notebook]]() |
| [`SJMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_jmd/sjmd.py) / `SMJMD` (Successive Jump and Mode Decomposition) |         [[paper]](https://arxiv.org/abs/2504.08453)          |                          [[code]]()                          | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/jmd/sjmd.ipynb) |
| [`ESMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/esmd.py) (Extreme-Point Symmetric Mode Decomposition) |          [[paper]](https://arxiv.org/abs/1303.6540)          |         [[code]](https://github.com/WuShichao/esmd)          | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/emd_variants/esmd.ipynb) |
| [`STNBMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vncmd/stnbmd.py) (Short-Time Narrow-Band Mode Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/pii/S0022460X16002443?via%3Dihub) | [[code]](https://ww2.mathworks.cn/matlabcentral/fileexchange/56226-short-time-narrow-band-mode-decomposition-stnbmd-toolbox) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/vncmd/stnbmd.ipynb) |
| [`VTFMTD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_tfa/vtfmtd.py) (Variational TF Mode Tracking Decomposition) | [[paper]](https://www.sciencedirect.com/science/article/pii/S0165168426001179) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/183389-variational-time-frequency-mode-tracking-decomposition) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/tfa/vtfmtd.ipynb) |
| [`SST`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_tfa/sst.py) (Synchrosqueezing Transform) | [[paper]](https://doi.org/10.1016/j.sigpro.2014.08.010) | [[code]](https://www.commsp.ee.ic.ac.uk/~mandic/research/sst.htm) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/tfa/sst.ipynb) |
| [`SET`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_tfa/set.py) (Synchroextracting Transform) | [[paper]](https://doi.org/10.1109/TIE.2017.2696503) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/62483-synchroextracting-transform?s_tid=srchtitle) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/tfa/set.ipynb) |
| [`SWD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_osd/swd.py) (Swarm Decomposition) |   [[paper]](https://doi.org/10.1016/j.sigpro.2016.09.004)    |  [[code]](https://github.com/gkaposto/Swarm-Decomposition)   | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/osd/swd.ipynb) |
| [`OSD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_osd/osd.py) (Optimization-based Signal Decomposition) | [[paper]](https://web.stanford.edu/~boyd/papers/sig_decomp_mprox.html) |   [[code]](https://github.com/cvxgrp/signal-decomposition)   | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/osd/osd_intro.ipynb) |
| [`STL`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/tsa/_stl.py) (Seasonal-Trend decomposition using LOESS) | [[paper]](https://www.nniiem.ru/file/news/2016/stl-statistical-model.pdf) | [[code]](https://www.statsmodels.org/stable/examples/notebooks/generated/stl_decomposition.html) | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/tsa/stl.ipynb) |
| [`MSTL`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/tsa/_mstl.py) (Multivariate Seasonal-Trend decomposition using LOESS) |         [[paper]](https://arxiv.org/abs/2107.13462)          |         [[code]](https://github.com/KishManani/MSTL)         | [[notebook]](https://github.com/wwhenxuan/PySDKit/blob/main/examples/tsa/mstl.ipynb) |



## Acknowledgements 🎖️ <a id="Acknowledgements"></a>

We would like to thank the researchers in signal processing for providing us with valuable algorithms and promoting the continuous progress in this field. However, since the main programming language used in `signal processing` is `Matlab`, and `Python` is the main battlefield of `machine learning` and `deep learning`, the usage of signal decomposition in machine learning and deep learning is far less extensive than `wavelet transformation`. In order to further promote the organic combination of signal decomposition and machine learning, we developed `PySDKit`. We would like to express our gratitude to [PyEMD](https://github.com/laszukdawid/PyEMD), [Sktime](https://www.sktime.net/en/latest/index.html), [Scikit-learn](https://scikit-learn.org/stable/), [Scikit-Image](https://scikit-image.org/docs/stable/), [statsmodels](https://www.statsmodels.org/stable/index.html), [vmdpy](https://github.com/vrcarva/vmdpy),  [MEMD-Python-](https://github.com/mariogrune/MEMD-Python-),  [ewtpy](https://github.com/vrcarva/ewtpy), [EWT-Python](https://github.com/bhurat/EWT-Python), [PyLMD](https://github.com/shownlin/PyLMD), [pywt](https://github.com/PyWavelets/pywt), [SP_Lib](https://github.com/hustcxl/SP_Lib), [dsatools](https://github.com/MVRonkin/dsatools) and [signal-decomposition](https://github.com/cvxgrp/signal-decomposition).

## Contributing 🤗

This project exists thanks to all the people who contribute. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) for how to add an algorithm, tests, and
gallery examples (the same guide is in the
[documentation](https://pysdkit.readthedocs.io/en/stable/development/index.html)).

<a href="https://github.com/wwhenxuan/PySDKit/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=wwhenxuan/PySDKit" />
</a>

