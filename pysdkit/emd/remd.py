# -*- coding: utf-8 -*-
"""
Created on 2025/02/05 13:15:49
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np


class REMD(object):
    """
    Robust Empirical Mode Decomposition
    A useful adaptive signal processing tool for multi-component signal separation, non-stationary signal processing.

    The REMD is an improved empirical mode decomposition powered by soft sifting stopping criterion (SSSC).
    The SSSC is an adaptive sifting stop criterion to stop the sifting process automatically for the EMD.
    It extracts a set of mono-component signals (called intrinsic mode functions) from a temporal mixed signal.
    It can be used together with Hilbert transform (or other demodulation techniques) for advanced time-frequency analysis.

    Zhiliang Liu*, Dandan Peng, Ming J. Zuo, Jiansuo Xia, and Yong Qin.
    Improved Hilbert-Huang transform with soft sifting stopping criterion and its application to fault diagnosis of wheelset bearings.
    ISA Transactions. 125: 426-444, 2022.

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/70032-robust-empirical-mode-decomposition-remd
    """

    def __init__(self):
        pass
