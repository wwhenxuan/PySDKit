# -*- coding: utf-8 -*-
"""
Created on 2025/02/05 13:31:52
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np


class JMD(object):
    """
    Jump Plus AM-FM Mode Decomposition

    Jump Plus AM-FM Mode Decomposition (JMD) is a novel method for decomposing a nonstationary signal into amplitude-
    and frequency-modulated (AM-FM) oscillations and discontinuous (jump) components.
    Current nonstationary signal decomposition methods are designed to either obtain constituent AM-FM oscillatory modes
    or the discontinuous and residual components from the data, separately. Yet, many real-world signals of interest
    simultaneously exhibit both behaviors i.e., jumps and oscillations.
    In JMD method, we design and solve a variational optimization problem to accomplish this task.
    The optimization formulation includes a regularization term to minimize the bandwidth of all signal modes for
    effective oscillation modeling, and a prior for extracting the jump component. JMD addresses the limitations of
    conventional AM-FM signal decomposition methods in extracting jumps, as well as the limitations of existing jump
    extraction methods in decomposing multiscale oscillations.

    Mojtaba Nazari, Anders Rosendal Korshøj and Naveed ur Rehman,
    ''Jump Plus AM-FM Mode Decomposition,'' IEEE TSP (in press),
    available on arXiv: https://doi.org/10.48550/arXiv.2407.07800

    MATLAB code: https://www.mathworks.com/matlabcentral/fileexchange/169388-jump-plus-am-fm-mode-decomposition-jmd?s_tid=prof_contriblnk
    """

    def __init__(self):
        pass
