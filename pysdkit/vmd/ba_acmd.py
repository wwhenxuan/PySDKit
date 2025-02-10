# -*- coding: utf-8 -*-
"""
Created on 2025/02/05 13:24:43
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np


class BA_ACMD(object):
    """
    Bandwidth-Aware Adaptive Chirp Mode Decomposition

    First, the filter bank property of ACMD is thoroughly analyzed based on Monte-Carlo simulation and then a bandwidth
    expression with respect to the penalty parameter is first obtained by fitting a power law model.
    Then, a weighted spectrum trend (WST) method is proposed to partition frequency bands and then guide the parameter
    determination of ACMD through the integration of the obtained bandwidth expression.
    In addition, according to the order of magnitude of the WST in each band, the BA-ACMD adopts a recursive framework
    to extract signal modes one by one. In this way, dominating signal modes related to wheel–rail excitations can be
    extracted and then subtracted from the vibration signal in advance so that the bearing faults induced signal modes can be successfully identified.

    Chen S, Guo L, Fan J, Yi C, Wang K, Zhai W.
    Bandwidth-aware adaptive chirp mode decomposition for railway bearing fault diagnosis.
    Structural Health Monitoring. 2024;23(2):876-902. doi:10.1177/14759217231174699
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Bandwidth-Aware Adaptive Chirp Mode Decomposition (BA_ACMD)"
