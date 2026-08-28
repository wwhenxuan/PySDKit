PYSDKIT's documentation
=======================

**Version**: |version|

PySDKit is a Python library for **signal decomposition**: splitting a
non-stationary recording into simpler modes (IMFs) that can be plotted,
used as features, or fed to a downstream model.

Installation
------------

.. code-block:: bash

   pip install pysdkit

Quick start
-----------

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

The same three-step pattern (import, instantiate, ``fit_transform``)
applies to VMD, BMEMD, and the other algorithms listed in
:doc:`API/modules`.

.. toctree::
   :hidden:

   user_guide/index
   auto_examples/index
   API/modules
   release_notes/index
   development/index
   about/index
