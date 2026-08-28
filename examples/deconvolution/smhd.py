r"""
Sparse Maximum Harmonics-to-Noise-Ratio Deconvolution (SMHD)
============================================================

When the impacts are **weak**, maximising kurtosis or ICS2 can still
lock onto a louder interferer. SMHD (Miao, Zhao, Lin & Lei,
Measurement Science and Technology 2016) instead maximises the
**harmonics-to-noise ratio** of the envelope autocorrelation at lag
:math:`T`, after a Gaussian-like sparsity map has suppressed the dense
floor:

.. math::

   y\leftarrow y\cdot\bigl(1-e^{-y^2/(2\mu^2)}\bigr).

The inverse FIR is

.. math::

   f=\bigl(2A\bigr)^{-1} r_{yx}(T),

where :math:`A` is the Toeplitz autocovariance of the *raw* :math:`x`
(built once) and :math:`r_{yx}` is a lag-:math:`T` weighted cross-correlation
of the sparsified output. :math:`T` is the same envelope-ACF estimator
as IMCKD (MATLAB ```TT```), and :math:`\mu` is adapted from the change in
kurtosis:

.. math::

   \Delta_k=\frac{\mathrm{kurt}(f_{\mathrm{new}}*x)}{\mathrm{kurt}(y_{\mathrm{old}})},
   \qquad
   \mu\leftarrow\mu\cdot
   \begin{cases}
   1+0.02(\Delta_k+1)/\Delta_k & \Delta_k>1,\\
   1-0.02(\Delta_k+1)/\Delta_k & \text{otherwise.}
   \end{cases}

The returned :math:`y` is the **sparsified** trace at the iteration of
maximum HNR, matching ```smhd.m``` (not the raw FIR output).

.. math::

   \mathrm{HNR}=\frac{r_{\max}}{1-r_{\max}}

is read from the ACF peak after the first zero-crossing.

**References**

.. epigraph::

    Y. Miao, M. Zhao, J. Lin, Y. Lei, *Sparse maximum
    harmonics-to-noise-ratio deconvolution for weak fault signature
    detection in bearings*, Measurement Science and Technology 27
    (2016) 105004.

.. epigraph::

    Y. Miao, B. Zhang, J. Lin et al., *A review on the application of
    blind deconvolution in machinery fault diagnosis*, Mechanical
    Systems and Signal Processing 163 (2022) 108202.
"""

# %%
# 1. Imports
# ----------

import numpy as np
import matplotlib.pyplot as plt

from pysdkit.data import load_smhd_sig3
from pysdkit.utils import (
    annotate_harmonics,
    envelope_spectrum,
    matlab_kurtosis,
    peak_frequency,
    smhd,
)

# %%
# 2. What the code computes
# -------------------------
#
# #. If :math:`T` is omitted, estimate it from the raw envelope ACF
#    (```TT```) and round it. Initialise :math:`f` as a centred
#    differentiator :math:`[+1,-1]`.
# #. Filter :math:`y=f*x`; log kurtosis and envelope HNR *before*
#    sparsity.
# #. Apply the sparse map with the current :math:`\mu` (the MATLAB demo
#    uses :math:`1.5\,\mathrm{rms}(x)`, not the code default
#    :math:`\mathrm{mean}(x)`).
# #. Build the lag-:math:`T` weighted cross-correlation against :math:`x`;
#    :math:`f\leftarrow (2A)^{-1} r`; unit-normalise :math:`f`.
# #. Adapt :math:`\mu` from the kurtosis ratio; re-estimate :math:`T` from
#    the envelope of the **sparsified** :math:`y` (the loop does not
#    re-round :math:`T`; :math:`A` is never rebuilt).
# #. Keep the sparsified :math:`y` of maximum HNR.
#
# Like IMCKD, SMHD always runs ```term_iter``` iterations. It still
# assumes one dominant envelope period. Several incommensurate cycles
# of similar strength, or a cycle that lives in the carrier rather
# than the envelope, will mislead ```TT```.

# %%
# 3. MATLAB demo (```03 SMHD/demo.m```)
# -------------------------------------
#
# Packaged ```sig3.mat```: 20001 samples at :math:`f_s=20\,\mathrm{kHz}`.
# Call ```smhd(fs, x, 100, 30, 1.5*rms(x), [], 0)```. The demo marks
# **BPFI = 38 Hz**.

record = load_smhd_sig3()
x = record["signal"] - np.mean(record["signal"])
fs = float(record["fs"])
t = record["t"]
bpfi = float(record["bpfi"])
rms = float(np.sqrt(np.mean(x**2)))
print("N =", x.size, "fs =", fs, "BPFI =", bpfi, "rms =", rms)

freq_x, spec_x = envelope_spectrum(x, fs, scale="fs")
fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), constrained_layout=True)
axes[0].plot(t, x, color="C0", lw=0.7)
axes[0].set_ylim(-2, 2.5)
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Raw data")
axes[0].legend([f"Kurtosis={matlab_kurtosis(x):.3f}"], loc="upper right")
axes[1].plot(freq_x, spec_x, color="C0", lw=0.8)
axes[1].set_xlim(0, 200)
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel("Amplitude")
axes[1].set_title("Envelope spectrum of raw data")
plt.show()

# %%
# 4. Run SMHD
# -----------

y, fir, info = smhd(x, fs, filter_size=100, term_iter=30, mu=1.5 * rms)
print("||f|| =", np.linalg.norm(fir))
print("HNR max =", info["hnr_max"], "final T =", info["period"])
print("kurtosis(y) =", matlab_kurtosis(y))

freq_y, spec_y = envelope_spectrum(y, fs, scale="fs")
peak = peak_frequency(freq_y, spec_y, f_max=200.0)
print("envelope peak in [0, 200] Hz:", peak)

fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), constrained_layout=True)
axes[0].plot(t, y, color="C0", lw=0.7)
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Filtered signal by SMHD (sparsified y)")
axes[0].legend([f"Kurtosis={matlab_kurtosis(y):.3f}"], loc="upper right")
axes[1].plot(freq_y, spec_y, color="C0", lw=0.8)
axes[1].set_xlim(0, 200)
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel("Amplitude")
axes[1].set_title("Envelope spectrum after SMHD")
annotate_harmonics(freq_y, spec_y, bpfi, n_harmonics=6, ax=axes[1], symbol=r"f_i")
plt.show()
