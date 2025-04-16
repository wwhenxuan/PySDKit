# -*- coding: utf-8 -*-
"""
Created on 2025/04/16 22:51:55
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np


class EFD(object):
    """
    Empirical Fourier Decomposition

    The proposed EFD combines the uses of an improved Fourier spectrum segmentation technique and an ideal filter bank.
    The segmentation technique can solve the inconsistency problem by predefining the number of modes in a signal to be
    decomposed and filter functions in the ideal filter bank have no transition phases, which can solve the mode mixing problem.
    Numerical investigations are conducted to study the accuracy of the EFD. It is shown that the EFD can yield accurate
    and consistent decomposition results for signals with multiple non-stationary modes and those with closely-spaced modes,
    compared with decomposition results by the EWT, FDM, variational mode decomposition and empirical mode decomposition.

    Wei Zhou, Zhongren Feng, Y.F. Xu, Xiongjiang Wang, Hao Lv,
    Empirical Fourier decomposition: An accurate signal decomposition method for nonlinear and non-stationary time series analysis,
    Mechanical Systems and Signal Processing,
    Volume 163, 2022, 108155, ISSN 0888-3270, https://doi.org/10.1016/j.ymssp.2021.108155.
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        """allow instances to be called like functions"""
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Empirical Fourier Decomposition (EFD)"
