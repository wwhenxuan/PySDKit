# Why develop PySDKit?<img width="15%" align="right" src="https://raw.githubusercontent.com/wwhenxuan/PySDKit/main/images/logo.png?raw=true">

## What is signal decomposition 😊

**Signal decomposition** is one of the most reliable time-frequency analysis techniques in the “post-wavelet era.” This method assumes that complex **non-stationary and nonlinear signals** in the real world are composed of multiple simple **sub-signals** (intrinsic mode functions). By analyzing the characteristics of these sub-signals, the time-frequency information of the original complex signal can be indirectly or directly revealed.

## What are the advantages of signal decomposition 🤩

Signal decomposition overcomes the limitations of Fourier transform in processing non-stationary and nonlinear signals, and the intrinsic mode functions obtained after decomposition surpass wavelet transform time-frequency representations in various aspects. Since the introduction of the **Hilbert-Huang Transform** in 1998, a series of **univariate and multivariate signal decomposition algorithms** have seen significant advancements and have been successfully applied in fields such as healthcare and industry.

## The actual use of signal decomposition in machine learning 😭

However, since MATLAB is the primary programming language used by researchers in the field of signal processing, and Python is the main platform for machine learning and deep learning, signal decomposition algorithms lack a comprehensive and integrated Python library, similar to [**PyWavelets**](https://pywavelets.readthedocs.io/en/latest/index.html) and [**PyTorch-Wavelet-Toolbox**](https://github.com/v0lta/PyTorch-Wavelet-Toolbox) for **wavelet transform**. As a result, their usage in the fields of machine learning and deep learning is far less widespread compared to wavelet transform.

## Motivation for developing the PySDKit library 😋

In order to make signal decomposition algorithms a more efficient **feature engineering** tool, and to facilitate their integration with machine learning or deep neural network models, I began developing the first comprehensive Python library for signal decomposition, PySDKit, in April 2024. This project aims to make these algorithms easier to use, reduce our research difficulty, and shorten the research cycle. The project has successfully implemented mainstream algorithms, including Empirical Mode Decomposition (EMD), Empirical Wavelet Transform (EWT), Variational Mode Decomposition (VMD), and Variational Nonlinear Chirp Mode Decomposition (VNCMD), along with a visualization platform 😍.
