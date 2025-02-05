# -*- coding: utf-8 -*-
"""
Created on 2025/02/04 13:10:33
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
We refactored from https://github.com/mariogrune/MEMD-Python-
"""
import numpy as np


class MEMD(object):
    """
    Multivariate Empirical Mode Decomposition

    Rehman, Naveed, and Danilo P. Mandic. "Multivariate empirical mode decomposition."
    Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 466.2117 (2010): 1291-1302.

    G. Rilling, P. Flandrin and P. Goncalves, "On Empirical Mode Decomposition and its Algorithms", Proc of the IEEE-EURASIP
     Workshop on Nonlinear Signal and Image Processing, NSIP-03, Grado (I), June 2003

    N. E. Huang et al., "A confidence limit for the Empirical Mode Decomposition and Hilbert spectral analysis",
     Proceedings of the Royal Society A, Vol. 459, pp. 2317-2345, 2003

    Python code: https://github.com/mariogrune/MEMD-Python-
    MATLAB code: http://www.commsp.ee.ic.ac.uk/~mandic/research/emd.htm
    R code: https://rdrr.io/github/PierreMasselot/Library--emdr/man/memd.html
    """

    def __init__(self):
        pass

    def fit_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        Preform the Multivariate Empirical Mode Decomposition method.
        请注意该方法仅适用于输入维数大于等于3的信号（遵循最原始的MEMD的MATLAB代码）
        :param signal:
        :return:
        """