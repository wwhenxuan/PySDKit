# -*- coding: utf-8 -*-
"""
Created on 2025/02/03 18:36:18
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
"""
from ._splines import akima, cubic, pchip, cubic_hermite, cubic_spline_3pts

from ._find_extrema import find_extrema_parabol, find_extrema_simple

from ._prepare_points import prepare_points_parabol, prepare_points_simple

from .emd import EMD

from .eemd import EEMD

from .ceemdan import CEEMDAN

from .remd import REMD

from .memd import MEMD

from .tvf_emd import TVF_EMD

from .efd import EFD

from .hht import HHT
