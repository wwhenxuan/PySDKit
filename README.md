# PySDKit: signal decomposition in Python

[![PyPI version](https://badge.fury.io/py/PySDKit.svg)](https://pypi.org/project/PySDKit/) ![License](https://img.shields.io/github/license/wwhenxuan/PySDKit) [![codecov](https://codecov.io/gh/wwhenxuan/PySDKit/branch/main/graph/badge.svg?token=YOUR_TOKEN)](https://codecov.io/gh/wwhenxuan/PySDKit) [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-brightgreen.svg)](https://arxiv.org/abs/1234.56789) ![GitHub stars](https://img.shields.io/github/stars/wwhenxuan/PySDKit?style=social)

A Python library for signal decomposition algorithms

<center><img src=".\images\Logo_sd.png" alt="Logo_sd" style="zoom:6%;" /></center>

## Installation

You can install `PySDKit` through pip.

~~~
pip install pysdkit
~~~

We only used [`NumPy`](https://numpy.org/), [`Scipy`](https://scipy.org/) and [`matplotlib`](https://matplotlib.org/) when developing the project.

## Example script

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

![example](.\images\example.jpg)

The EMD in the above example is the most classic `empirical mode decomposition` algorithm in signal decomposition. For more complex signals, you can try other algorithms such as variational mode decomposition ([`VMD`](https://ieeexplore.ieee.org/abstract/document/6655981)).

~~~python
from pysdkit import VMD

# load new signal
signal = np.load("example.npy")

# use variational mode decomposition
vmd = VMD(alpha=500, K=3, tau=0.0, tol=1e-9)
IMFs = vmd.fit_transform(signal=signal)
print(IMFs.shape)

vmd.plot_IMFs(save_figure=True)
~~~

![vmd_example](.\images\vmd_example.jpg)

Better observe the characteristics of the decomposed intrinsic mode function in the frequency domain.

~~~python
from pysdkit.plot import plot_IMFs_amplitude_spectra

# frequency domain visualization
plot_IMFs_amplitude_spectra(IMFs, smooth="exp")   # use exp smooth
~~~

![frequency_example](.\images\frequency_example.jpg)

