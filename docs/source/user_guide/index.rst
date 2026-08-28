User guide
==========

Three-step API
--------------

Decomposition classes share a scikit-learn-style interface:

1. Import the algorithm from :mod:`pysdkit`.
2. Create an instance (optional parameters set the rank, filter length, …).
3. Call ``fit_transform`` on a 1-D array, a channel stack, or an image.

.. code-block:: python

   from pysdkit import VMD
   from pysdkit.data import test_vmd
   from pysdkit.plot import plot_IMFs

   t, signal, fs = test_vmd()
   vmd = VMD(alpha=2000, K=4, tau=0.0, tol=1e-7)
   IMFs = vmd.fit_transform(signal)
   plot_IMFs(signal, IMFs, view="2d_freq", fs=fs, freq_max=fs / 2)

Which family to pick
--------------------

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

Packaged data
-------------

Demo arrays install with the package.  Prefer the loaders in
:mod:`pysdkit.data` over any local MATLAB dump:

.. code-block:: python

   from pysdkit.data import load_bmemd_source02, load_bss_beam, test_vmd

   images = load_bmemd_source02()["signal"]  # (2, 224, 224) in [0, 1]
   beam = load_bss_beam()["signal"]

``.npy`` files live under ``pysdkit/data/real_world``; callers still pass
a bare file name to the loader.  The gitignored ``repo/`` directory is
not part of the install and must not be used at runtime.

Next steps
----------

- Example gallery: :doc:`/auto_examples/index`
- Public classes: :doc:`/API/modules`
