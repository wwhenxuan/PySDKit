# -*- coding: utf-8 -*-
"""
Created on 2025/02/12 11:06:23
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
import numpy as np


class CVMD2D(object):
    """
    Compact Variational Mode Decomposition for 2D Images

    Spatially Compact and Spectrally Sparse Image Decomposition and Segmentation.
    Decomposing multidimensional signals, such as images, into spatially compact,
    potentially overlapping modes of essentially wavelike nature makes these components accessible for further downstream analysis.
    This decomposition enables space-frequency analysis, demodulation, estimation of local orientation, edge and corner detection,
    texture analysis, denoising, inpainting, or curvature estimation.

    [1] D. Zosso, K. Dragomiretskiy, A.L. Bertozzi, P.S. Weiss, Two-Dimensional Compact Variational Mode Decomposition,
    Journal of Mathematical Imaging and Vision, 58(2):294–320, 2017. DOI:10.1007/s10851-017-0710-z.

    [2] K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition,
    IEEE Trans. on Signal Processing, 62(3):531-544, 2014. DOI:10.1109/TSP.2013.2288675.

    [3] K. Dragomiretskiy, D. Zosso, Two-Dimensional Variational Mode Decomposition,
    EMMCVPR 2015, Hong Kong, LNCS 8932:197-208, 2015. DOI:10.1007/978-3-319-14612-6_15.

    MATLAB code: https://ww2.mathworks.cn/matlabcentral/fileexchange/67285-two-dimensional-compact-variational-mode-decomposition-2d-tv-vmd?s_tid=FX_rc2_behav
    """

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __str__(self) -> str:
        """Get the full name and abbreviation of the algorithm"""
        return "Compact Variational Mode Decomposition for 2D Images (CVMD2D)"
