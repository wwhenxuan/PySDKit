About
=====

Project
-------

PySDKit collects **signal decomposition** algorithms in one Python
package so they can be used as feature-engineering tools next to NumPy,
SciPy, and machine-learning stacks.  A non-stationary recording is
treated as a sum of simpler modes; those modes can be plotted, scored
(for example with entropy), or passed to a downstream model through the
same ``fit_transform`` style as scikit-learn.

The public API is **mode extraction**: EMD-style sifting, variational
and chirp models, multivariate / image decompositions, and related
time-frequency transforms, plus helpers in :mod:`pysdkit.data`,
:mod:`pysdkit.plot`, :mod:`pysdkit.entropy`, and :mod:`pysdkit.tsa`.

It is **not** a linear subspace-identification toolbox (state-space
``A, B, C, D`` from IO data), and it is not a replacement for
`PyWavelets <https://pywavelets.readthedocs.io/>`_.  Install and the
three-step API are in the :doc:`../user_guide/index`; the gallery is
under :doc:`../auto_examples/index`.

Citing PySDKit
--------------

Cite **two things** when a result depends on this library.

1. **The algorithm.**  Use the original paper of the method you called
   (EMD, VMD, MVMD, …).  Class and module docstrings list those
   references.

2. **The software.**  There is no journal article for the package yet.
   Name the GitHub repository and the **version you actually ran**
   (``pip show pysdkit``, or ``pysdkit.__version__``):

   .. code-block:: text

      @software{pysdkit,
        author  = {{PySDKit developers}},
        title   = {{PySDKit}: signal decomposition in {Python}},
        url     = {https://github.com/wwhenxuan/PySDKit},
        version = {X.Y.Z},
      }

   Replace ``X.Y.Z`` with the installed version.  If a software paper
   (for example JOSS) appears later, prefer that citation.

People
------

PySDKit is an open-source project started in April 2024.  Releases,
review, and the documentation are handled by the maintainers; algorithms
and fixes come from a wider set of contributors.

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Name
     - Role
     - Contact
   * - `Whenxuan Wang <https://github.com/wwhenxuan>`_
     - Maintainer
     - wwhenxuan@gmail.com
   * - `RuiZhe Wang <https://github.com/changewam>`_
     - Maintainer
     - 3133986068@qq.com

Package metadata also lists Rongkun Zhu, Kai Wu, Lei Wang, josefinez,
Deeksha Manjunath, Yuan Feng, WenTong Zhao, and JacktheFowler.  The
complete, up-to-date list is the
`GitHub contributors graph <https://github.com/wwhenxuan/PySDKit/graphs/contributors>`_.

.. image:: https://contrib.rocks/image?repo=wwhenxuan/PySDKit
   :target: https://github.com/wwhenxuan/PySDKit/graphs/contributors
   :alt: GitHub contributors to PySDKit

License and use
---------------

PySDKit is released under the
`MIT License <https://github.com/wwhenxuan/PySDKit/blob/main/LICENSE>`_.
You may use, copy, modify, and redistribute it in research, teaching,
and commercial work, provided the copyright notice and permission
notice are kept with substantial portions of the code.

The implementations follow published papers and public reference code.
Using a method here does not replace citing its authors.  The software
is provided “as is”, without warranty.

Related work
------------

We thank researchers in signal processing for the algorithms themselves,
and the Python projects that made a unified library practical:

- `PyEMD <https://github.com/laszukdawid/PyEMD>`_
- `vmdpy <https://github.com/vrcarva/vmdpy>`_ and
  `ewtpy <https://github.com/vrcarva/ewtpy>`_
- `EWT-Python <https://github.com/bhurat/EWT-Python>`_
- `MEMD-Python- <https://github.com/mariogrune/MEMD-Python->`_
- `PyLMD <https://github.com/shownlin/PyLMD>`_
- `PyWavelets <https://github.com/PyWavelets/pywt>`_
- `signal-decomposition <https://github.com/cvxgrp/signal-decomposition>`_
- `scikit-learn <https://scikit-learn.org/>`_,
  `scikit-image <https://scikit-image.org/>`_,
  `sktime <https://www.sktime.net/>`_, and
  `statsmodels <https://www.statsmodels.org/>`_
- `SP_Lib <https://github.com/hustcxl/SP_Lib>`_ and
  `dsatools <https://github.com/MVRonkin/dsatools>`_

Links
-----

- Source: https://github.com/wwhenxuan/PySDKit
- Issues: https://github.com/wwhenxuan/PySDKit/issues
- PyPI: https://pypi.org/project/PySDKit/
- Docs: https://pysdkit.readthedocs.io/
- Code of Conduct: https://github.com/wwhenxuan/PySDKit/blob/main/CODE_OF_CONDUCT.md

To add an algorithm, tests, or a gallery example, see
:doc:`../development/index`.
