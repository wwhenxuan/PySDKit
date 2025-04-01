# -*- coding: utf-8 -*-
"""
Created on 2025/02/05 13:31:52
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np

from typing import Optional, Tuple


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

    def __init__(
        self,
        K: int,
        alpha: Optional[float] = 5000,
        init: Optional[str] = "zero",
        tol: Optional[float] = 1e-6,
        beta: Optional[float] = 0.03,
        b_bar: Optional[float] = 0.45,
        tau: Optional[float] = 0,
    ) -> None:
        """
        :param K: the number of modes to be recovered
        :param alpha: the balancing parameter of the mode bandwidth
        :param init: - 'zero': all omegas start at 0
                     - 'uniform': all omegas start uniformly distributed
                     - 'random': all omegas initialized randomly
        :param tol: tolerance of convergence criterion; typically around 1e-6
        :param beta: the balancing parameter of the jump constraint (1/expected number of jumps)
        :param b_bar: the balancing parameter related to the β parameter
        :param tau: the dual ascent step (set to 0 for noisy signal)
        """
        self.K = K
        self.alpha = alpha
        self.init = init
        self.tol = tol
        self.beta = beta
        self.b_bar = b_bar
        self.tau = tau

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Jump Plus AM-FM Mode Decomposition (JMD)"
