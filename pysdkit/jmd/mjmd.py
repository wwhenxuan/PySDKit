# -*- coding: utf-8 -*-
"""
Created on 2025/02/05 13:32:06
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np


class MJMD(object):
    """
    Multivariate Jump Plus AM-FM Mode Decomposition

    Multivariate Jump Plus AM-FM Mode Decomposition (MJMD) is a novel method for decomposing a multivariate signal into
    amplitude- and frequency-modulated (AM-FM) oscillations and discontinuous (jump) components.
    Current multivariate signal decomposition methods are designed to either obtain constituent AM-FM oscillatory modes from the data.
    Yet, many real-world signals of interest simultaneously exhibit both behaviors i.e., jumps and oscillations.
    In MJMD method, we design and solve a variational optimization problem to accomplish this task.
    The optimization formulation includes a regularization term to minimize the bandwidth of all signal modes for effective
    oscillation modeling, and a prior for extracting the jump component. MJMD addresses the limitations of conventional
    AM-FM signal decomposition methods in extracting jumps, as well as the limitations of existing jump extraction methods in decomposing multiscale oscillations.

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/169393-multivariate-jump-plus-am-fm-mode-decomposition-mjmd?s_tid=srchtitle
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Multivariate Jump Plus AM-FM Mode Decomposition (MJMD)"
