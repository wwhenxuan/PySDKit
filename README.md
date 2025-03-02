# PySDKit: signal decomposition in Python

<div align="center">

[![PyPI version](https://badge.fury.io/py/PySDKit.svg)](https://pypi.org/project/PySDKit/) 
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
from pysdkit import EMD
from pysdkit.data import test_emd
from pysdkit.plot import plot_IMFs

t, signal = test_emd()

# create an instance for signal decomposition
emd = EMD()
# implement signal decomposition
IMFs = emd.fit_transform(signal, max_imfs=2)
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

## Target 🎯 <a id="Target"></a>

`PySDKit` is still under development. We are currently working on reproducing the signal decomposition algorithms in the table below, including not only common decomposition algorithms for `univariate signals` such as EMD and VMD, but also decomposition algorithms for `multivariate signals` such as MEMD and MVMD. We will also further reproduce the decomposition algorithms for `two-dimensional images` to make PySDKit not only suitable for signal processing, but also for image analysis and understanding. See [`Mission`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/README.md) for the reasons why we developed PySDKit.

|                                                                   Algorithm                                                                   |                            Paper                             |                             Code                             | State |
|:---------------------------------------------------------------------------------------------------------------------------------------------:| :----------------------------------------------------------: | :----------------------------------------------------------: | :---: |
|                  [`EMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/emd.py) (Empirical Mode Decomposition)                   | [[paper]](https://royalsocietypublishing.org/doi/abs/10.1098/rspa.1998.0193) | [[code]](https://www.mathworks.com/help/signal/ref/emd.html) |   ✔️   |
|           [`MEMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/memd.py) (Multivariate Empirical Mode Decomposition)           | [[paper]](https://royalsocietypublishing.org/doi/full/10.1098/rspa.2009.0502) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd) |   ✔️   |
|         [`BEMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd2d/bemd.py) (Bidimensional Empirical Mode Decomposition)          | [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0262885603000945) | [[code]](https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/BEMD.py) |   ✖️   |
|                                               [`CEMD`]() (Complex Empirical Mode Decomposition)                                               | [[paper]](https://ieeexplore.ieee.org/abstract/document/4063369) |                          [[code]]()                          |   ✖️   |
|             [`EEMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/eemd.py) (Ensemble Empirical Mode Decomposition)             | [[paper]](https://www.sciencedirect.com/topics/physics-and-astronomy/ensemble-empirical-mode-decomposition) | [[code]](https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EEMD.py) |   ✔️   |
|              [`REMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/remd.py) (Robust Empirical Mode Decomposition)              | [[paper]](https://www.sciencedirect.com/science/article/pii/S0019057821003785) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/70032-robust-empirical-mode-decomposition-remd) |   ✔️   |
|  [`BMEMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd2d/bmemd.py) (Bidimensional Multivariate Empirical Mode Decomposition)  |   [[paper]](https://ieeexplore.ieee.org/document/8805082)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/72343-bidimensional-multivariate-empirical-mode-decomposition?s_tid=FX_rc1_behav) |   ✖️   |
|        [`CEEMDAN`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/ceemdan.py) (Complete Ensemble EMD with Adaptive Noise)        |   [[paper]](https://ieeexplore.ieee.org/document/5947265)    | [[code]](https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/EEMD.py) |   ✔️   |
|              [`TVF_EMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_emd/tvf_emd.py) (Time Varying Filter Based EMD)              | [[paper]](https://www.sciencedirect.com/science/article/pii/S0165168417301135) |   [[code]](https://github.com/stfbnc/pytvfemd/tree/master)   |   ✔️   |
|                   [`FAEMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_faemd/faemd.py) (Fast and Adaptive EMD)                   |   [[paper]](https://ieeexplore.ieee.org/document/8447300)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd) |   ✔️   |
|         [`FAEMD2D`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_faemd/faemd2d.py) (Two-Dimensional Fast and Adaptive EMD)         |   [[paper]](https://ieeexplore.ieee.org/document/8447300)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd) |   ✖️   |
|        [`FAEMD3D`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_faemd/faemd3d.py) (Three-Dimensional Fast and Adaptive EMD)        |   [[paper]](https://ieeexplore.ieee.org/document/8447300)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/71270-fast-and-adaptive-multivariate-and-multidimensional-emd) |   ✖️   |
|                 [`HVD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_hvd/hvd.py) (Hilbert Vibration Decomposition)                 | [[paper]](https://www.sciencedirect.com/science/article/pii/S0022460X06001556) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/178804-hilbert-vibration-decomposition?s_tid=FX_rc1_behav) |   ✔️   |
|               [`ITD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_itd/itd.py) (Intrinsic Time-Scale Decomposition)                | [[paper]](https://royalsocietypublishing.org/doi/10.1098/rspa.2006.1761) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/69380-intrinsic-time-scale-decomposition-itd) |   ✔️   |
|                                                [`ALIF`]() (Adaptive Local Iterative Filtering)                                                |          [[paper]](https://arxiv.org/abs/1411.6051)          | [[code]](https://ww2.mathworks.cn/matlabcentral/fileexchange/56210-alif) |   ✖️   |
|                    [`LMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_lmd/lmd.py) (Local Mean Decomposition)                     | [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S1051200418308133?via%3Dihub) | [[code]](https://github.com/shownlin/PyLMD/blob/master/PyLMD/LMD.py) |   ✔️   |
|                [`RLMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_lmd/rlmd.py) (Robust Local Mean Decomposition)                | [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0888327017301619) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/66935-robust-local-mean-decomposition-rlmd) |   ✖️   |
|                   [`FMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_fmd/fmd.py) (Feature Mode Decomposition)                    |   [[paper]](https://ieeexplore.ieee.org/document/9732251)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/108099-feature-mode-decomposition-fmd) |   ✖️   |
|                   [`SSA`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_ssa/ssa.py) (Singular Spectral Analysis)                    | [[paper]](https://orca.cardiff.ac.uk/id/eprint/15208/1/Zhiglavsky_SSA_encyclopedia.pdf) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/58967-singular-spectrum-analysis-beginners-guide) |   ✔️   |
|                   [`EWT`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_ewt/ewt.py) (Empirical Wavelet Transform)                   |   [[paper]](https://ieeexplore.ieee.org/document/6522142)    | [[code]](https://www.mathworks.com/help/wavelet/ug/empirical-wavelet-transform.html) |   ✔️   |
|               [`EWT2D`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_ewt/ewt2d.py) (2D Empirical Wavelet Transform)                |         [[paper]](https://arxiv.org/abs/2405.06188)          |        [[code]](https://github.com/bhurat/EWT-Python)        |   ✖️   |
|                [`VMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd/vmd_c.py) (Variational Mode Decomposition)                 |   [[paper]](https://ieeexplore.ieee.org/document/6655981)    |          [[code]](https://github.com/vrcarva/vmdpy)          |   ✔️   |
|          [`MVMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd/mvmd.py) (Multivariate Variational Mode Decomposition)          |   [[paper]](https://ieeexplore.ieee.org/document/8890883)    |          [[code]](https://github.com/yunyueye/MVMD)          |   ✔️   |
|      [`VMD2D`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd2d/vmd2d.py) (Two-Dimensional Variational Mode Decomposition)       | [[paper]](https://ww3.math.ucla.edu/camreport/cam14-16.pdf)  | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/45918-two-dimensional-variational-mode-decomposition?s_tid=srchtitle) |   ✔️   |
| [`CVMD2D`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd2d/cvmd2d.py) (Two-Dimensional Compact Variational Mode Decomposition)  | [[paper]](https://link.springer.com/article/10.1007/s10851-017-0710-z) | [[code]](https://ww2.mathworks.cn/matlabcentral/fileexchange/67285-two-dimensional-compact-variational-mode-decomposition-2d-tv-vmd?s_tid=FX_rc2_behav) |   ✔️   |
|           [`SVMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd/svmd.py) (Successive Variational Mode Decomposition)           | [[paper]](https://www.sciencedirect.com/science/article/pii/S0165168420301535) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/98649-successive-variational-mode-decomposition-svmd-m?s_tid=FX_rc3_behav) |   ✖️   |
|      [`VNCMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vncmd/vncmd.py) (Variational Nonlinear Chirp Mode Decomposition)       |   [[paper]](https://ieeexplore.ieee.org/document/7990179)    | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/64292-variational-nonlinear-chirp-mode-decomposition) |   ✔️   |
|       [`INCMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vncmd/incmd.py) (Iterative Nonlinear Chirp Mode Decomposition)        | [[paper]](https://www.sciencedirect.com/science/article/pii/S0022460X2030403X?via%3Dihub) |      [[code]](https://github.com/sheadan/IterativeNCMD)      |    ✔️    |
|      [`MNCMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vncmd/mncmd.py) (Multivariate Nonlinear Chirp Mode Decomposition)      | [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0165168420302103) |                          [[code]]()                          |   ✖️   |
| [`AVNCMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vncmd/avncmd.py) (Adaptive Variational Nonlinear Chirp Mode Decomposition) | [[paper]](https://ieeexplore.ieee.org/abstract/document/9746147) |         [[code]](https://github.com/HauLiang/AVNCMD)         |   ✖️   |
|               [`ACMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd/acmd.py) (Adaptive Chirp Mode Decomposition)               |                         [[paper]]()                          | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/121373-data-driven-adaptive-chirp-mode-decomposition?s_tid=srchtitle) |   ✔️   |
|    [`BA-ACMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_vmd/ba_acmd.py) (Bandwidth-aware adaptive chirp mode decomposition)    | [[paper]](https://journals.sagepub.com/doi/abs/10.1177/14759217231174699) | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/132792-bandwidth-aware-adaptive-chirp-mode-decomposition-ba-acmd?s_tid=srchtitle) |   ✖️   |
|               [`JMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_jmd/jmd.py) (Jump Plus AM-FM Mode Decomposition)                |         [[paper]](https://arxiv.org/abs/2407.07800)          | [[code]](https://www.mathworks.com/matlabcentral/fileexchange/169388-jump-plus-am-fm-mode-decomposition-jmd?s_tid=prof_contriblnk) |   ✖️   |
|        [`MJMD`](https://github.com/wwhenxuan/PySDKit/blob/main/pysdkit/_jmd/mjmd.py) (Multivariate Jump Plus AM-FM Mode Decomposition)        |         [[paper]](https://arxiv.org/abs/2407.07800)          | [[code]](https://ms-intl.mathworks.com/matlabcentral/fileexchange/169393-multivariate-jump-plus-am-fm-mode-decomposition-mjmd?s_tid=FX_rc2_behav) |   ✖️   |
|                                            [`ESMD`]() (Extreme-Point Symmetric Mode Decomposition)                                            |          [[paper]](https://arxiv.org/abs/1303.6540)          |         [[code]](https://github.com/WuShichao/esmd)          |   ✖️   |
|                                           [`STNBMD`]() (Short-Time Narrow-Band Mode Decomposition)                                            | [[paper]](https://www.sciencedirect.com/science/article/pii/S0022460X16002443?via%3Dihub) | [[code]](https://ww2.mathworks.cn/matlabcentral/fileexchange/56226-short-time-narrow-band-mode-decomposition-stnbmd-toolbox) |   ✖️   |
|                                             [`STL`]() (Seasonal-Trend decomposition using LOESS)                                              | [[paper]](https://www.nniiem.ru/file/news/2016/stl-statistical-model.pdf) | [[code]](https://www.statsmodels.org/stable/examples/notebooks/generated/stl_decomposition.html) |   ✖️   |
|                                      [`MSTL`]() (Multivariate Seasonal-Trend decomposition using LOESS)                                       | [[paper]](https://www.nniiem.ru/file/news/2016/stl-statistical-model.pdf) | [[code]](https://www.statsmodels.org/stable/examples/notebooks/generated/mstl_decomposition.html) |   ✖️   |

## Acknowledgements 🎖️ <a id="Acknowledgements"></a>

We would like to thank the researchers in signal processing for providing us with valuable algorithms and promoting the continuous progress in this field. However, since the main programming language used in `signal processing` is `Matlab`, and `Python` is the main battlefield of `machine learning` and `deep learning`, the usage of signal decomposition in machine learning and deep learning is far less extensive than `wavelet transformation`. In order to further promote the organic combination of signal decomposition and machine learning, we developed `PySDKit`. We would like to express our gratitude to [PyEMD](https://github.com/laszukdawid/PyEMD), [Sktime](https://www.sktime.net/en/latest/index.html), [Scikit-learn](https://scikit-learn.org/stable/), [Scikit-Image](https://scikit-image.org/docs/stable/), [statsmodels](https://www.statsmodels.org/stable/index.html), [vmdpy](https://github.com/vrcarva/vmdpy),  [MEMD-Python-](https://github.com/mariogrune/MEMD-Python-),  [ewtpy](https://github.com/vrcarva/ewtpy), [EWT-Python](https://github.com/bhurat/EWT-Python), [PyLMD](https://github.com/shownlin/PyLMD), [pywt](https://github.com/PyWavelets/pywt), [SP_Lib](https://github.com/hustcxl/SP_Lib)and [dsatools](https://github.com/MVRonkin/dsatools).
