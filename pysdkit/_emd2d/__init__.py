# -*- coding: utf-8 -*-
"""
Bidimensional EMD variants.

* :class:`EMD2D` — univariate 2-D EMD of a single grayscale image ``(H, W)``.
* :class:`BMEMD` — multivariate 2-D MEMD of a channel stack ``(C, H, W)``,
  with optional multi-scale fusion.  Not a multi-channel wrapper around
  EMD2D: extrema are taken on directional projections, not per channel.
"""

from .emd2d import EMD2D

from .bmemd import BMEMD, local_var_img, fuse_images
