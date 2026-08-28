User guide
==========

What PySDKit is
---------------

**Signal decomposition** is a post-wavelet time-frequency tool: a
complex **non-stationary, nonlinear** recording is assumed to be a sum
of simpler sub-signals (intrinsic mode functions).  Analysing those
modes recovers time-frequency structure that a single Fourier transform
cannot, and that a fixed wavelet dictionary often cannot either.

Since the Hilbert–Huang Transform (1998), a family of univariate and
multivariate decompositions has grown quickly in MATLAB.  Python is the
usual home of machine learning, yet it still lacked a library comparable
to `PyWavelets <https://pywavelets.readthedocs.io/>`_ for this class of
algorithms.

PySDKit was started in April 2024 to close that gap: one package, a
shared ``fit_transform`` API, and matching plots so decomposition can
be used as **feature engineering** next to NumPy, SciPy, and neural
networks.  Families already in the tree include EMD, EWT, VMD / OVMD,
VNCMD, ALIF, APMD, and related 2-D methods.

Please cite the **original paper** of each algorithm you use; class and
module docstrings point to those references.

Install
-------

From PyPI:

.. code-block:: bash

   pip install pysdkit

Runtime dependencies are NumPy, SciPy, and Matplotlib.

From a clone (editable, for development):

.. code-block:: bash

   git clone https://github.com/wwhenxuan/PySDKit.git
   cd PySDKit
   pip install -e .

Demo arrays ship with the package.  Prefer loaders in
:mod:`pysdkit.data` (for example ``test_vmd``) over local MATLAB files.
``.npy`` data live under ``pysdkit/data/real_world``.  The gitignored
``repo/`` tree is not installed and must not be imported at runtime.

Unified interface
-----------------

Decomposition classes share a scikit-learn-style interface:

1. Import the algorithm from :mod:`pysdkit`.
2. Create an instance (optional parameters set the rank, filter length, …).
3. Call ``fit_transform`` on a 1-D array, a channel stack, or an image.
4. Inspect the original recording and the IMFs with
   :func:`~pysdkit.plot.plot_IMFs`.

.. code-block:: python

   from pysdkit import VMD
   from pysdkit.data import test_vmd
   from pysdkit.plot import plot_IMFs

   t, signal, fs = test_vmd()
   vmd = VMD(alpha=2000, K=4, tau=0.0, tol=1e-7)
   IMFs = vmd.fit_transform(signal)
   plot_IMFs(signal, IMFs, view="2d_freq", fs=fs, freq_max=fs / 2)

Which family to pick:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Family
     - Typical input
     - Start here
   * - EMD
     - 1-D univariate / multivariate
     - :class:`~pysdkit.EMD`, :class:`~pysdkit.CEEMDAN`, :class:`~pysdkit.MEMD`
   * - VMD / chirp
     - 1-D, known or unknown mode count
     - :class:`~pysdkit.VMD`, :class:`~pysdkit.ACMD`, :class:`~pysdkit.VNCMD`
   * - Images
     - ``(H, W)`` or ``(C, H, W)``
     - :class:`~pysdkit.EMD2D`, :class:`~pysdkit.BMEMD`, :class:`~pysdkit.VMD2D`
   * - Time-iterative
     - Sequential peeling in time
     - :class:`~pysdkit.ALIF`, :class:`~pysdkit.FMD`, :class:`~pysdkit.SSA`
   * - Time-frequency
     - STFT / CWT based
     - :class:`~pysdkit.SST`, :class:`~pysdkit.SET`, :class:`~pysdkit.VTFMTD`

A full name list is on :doc:`/API/modules`.

Worked examples
---------------

The snippets below are the three README-style demos from
``examples/demo.py``.  Run that script for CEEMDAN and 2-D VMD as well.

EMD — three-tone mixture
~~~~~~~~~~~~~~~~~~~~~~~~

A sum of cosines at 5 / 25 / 80 Hz.  EMD sifts recursively by scale.

.. code-block:: python

   import numpy as np
   from pysdkit import EMD
   from pysdkit.plot import plot_IMFs

   t = np.linspace(0, 1, 1000)
   signal = (
       np.sin(2 * np.pi * 5 * t)
       + 0.7 * np.sin(2 * np.pi * 25 * t)
       + 0.45 * np.sin(2 * np.pi * 80 * t)
   )

   emd = EMD()
   IMFs = emd.fit_transform(signal, max_imfs=3)
   plot_IMFs(signal, IMFs, view="2d_freq", fs=1000, freq_max=150)

.. image:: ../auto_examples/images/sphx_glr_demo_001.png
   :width: 95%
   :alt: EMD — three-tone mixture (time and frequency)

VMD — packaged multicomponent signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Non-recursive VMD estimates several band-limited modes jointly in the
Fourier domain.  Data come from ``test_vmd()``.

.. code-block:: python

   from pysdkit import VMD
   from pysdkit.data import test_vmd
   from pysdkit.plot import plot_IMFs

   t, signal, fs = test_vmd()
   vmd = VMD(alpha=2000, K=4, tau=0.0, tol=1e-7)
   IMFs = vmd.fit_transform(signal)
   plot_IMFs(signal, IMFs, view="2d_freq", fs=fs, freq_max=fs / 2)

.. image:: ../auto_examples/images/sphx_glr_demo_002.png
   :width: 95%
   :alt: VMD — packaged multi-component example (time and frequency)

MVMD — aligned multichannel modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three channels share a 36 Hz oscillation; each also carries a distinct
tone.  MVMD keeps mode index :math:`k` aligned across channels.

.. code-block:: python

   import numpy as np
   from pysdkit import MVMD
   from pysdkit.plot import plot_IMFs

   t = np.arange(0, 1, 0.001)
   signal = np.vstack(
       [
           np.cos(2 * np.pi * 2 * t) + np.cos(2 * np.pi * 36 * t),
           np.cos(2 * np.pi * 24 * t) + np.cos(2 * np.pi * 36 * t),
           np.cos(2 * np.pi * 80 * t) + np.cos(2 * np.pi * 36 * t),
       ]
   )

   mvmd = MVMD(alpha=2000, K=4, tau=0.0, init="uniform")
   IMFs = mvmd.fit_transform(signal)  # (K, T, C)
   plot_IMFs(signal, IMFs)

.. image:: ../auto_examples/images/sphx_glr_demo_003.png
   :width: 95%
   :alt: MVMD — mode-aligned multichannel decomposition

Examples gallery
----------------

The full executed gallery (theory, figures, Python and notebook
downloads) is :doc:`/auto_examples/index`.

``examples/demo.py`` is the overview script used above.  Each algorithm
family has its own folder under ``examples/`` (for example
``examples/emd/``, ``examples/vmd/``).
