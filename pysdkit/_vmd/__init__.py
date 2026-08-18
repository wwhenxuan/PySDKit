# -*- coding: utf-8 -*-
"""
Variational Mode Decomposition and related variants.
"""

from .vmd_f import vmd

from .vmd_c import VMD

from .mvmd import MVMD

from .vme import (
    VME,
    vme,
    load_vme_ecg_055m,
    generate_vme_example1,
    generate_vme_example2,
    generate_vme_example3a,
    generate_vme_example3b,
)

from .ovmd import OVMD

from .svmd import svmd, SVMD

from .stvmd import stvmd, STVMD
