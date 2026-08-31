r"""
Adaptive CYCBD (ACYCBD)
=======================

Second-order cyclostationarity is the natural language of a repeating
impact through a linear channel: the instantaneous power of :math:`y` has
Fourier coefficients at a cyclic frequency :math:`\alpha` and its
harmonics. **CYCBD** (Buzzoni, Antoni & D'Elia, JSV 2018) maximises
that ICS2 criterion by solving the generalised eigenproblem

.. math::

   XWX\,h=\kappa\,XX\,h

for the inverse FIR :math:`h`. The weight :math:`W` is a periodic projection
of :math:`|y|^p` onto the harmonics of a **known** :math:`\alpha`. As with
MCKD, a wrong prior cyclic frequency makes the filter fail.

**ACYCBD** (Zhang, Miao, Lin & Yi, MSSP 2021) estimates :math:`\alpha`
each iteration from the **envelope harmonic-product spectrum** (EHPS)
of the current output, rebuilds :math:`W`, and updates :math:`h` until the
relative change in :math:`\kappa` falls below ```param.RE``` (default
:math:`10^{-3}`) or 50 iterations elapse.

.. math::

   P(f)=\prod_{k=1}^{K}\bigl|E(kf)\bigr|,\qquad
   \hat\alpha=\arg\max_f P(f).

The estimated fundamental is then fed to CYCBD as
:math:`\alpha_k=k\hat\alpha`, :math:`k=1,\ldots,100`.

.. list-table::
   :header-rows: 1

   * -
     - **CYCBD**
     - **ACYCBD** (this notebook)
   * - Cyclic frequency
     - geometry :math:`\times` RPM
     - EHPS of current :math:`s`
   * - Linear algebra
     - :math:`XWX\,h=\kappa\,XX\,h`
     - same, with adapted :math:`W`
   * - Output
     - FIR and trimmed :math:`s`
     - FIR, :math:`s`, and :math:`\hat\alpha` versus iteration

**References**

.. epigraph::

    B. Zhang, Y. Miao, J. Lin, Y. Yi, *Adaptive maximum second-order
    cyclostationarity blind deconvolution and its application for
    locomotive bearing fault diagnosis*, Mechanical Systems and Signal
    Processing 158 (2021) 107736.

.. epigraph::

    M. Buzzoni, J. Antoni, G. D'Elia, *Blind deconvolution based on
    cyclostationarity maximization and its application to fault
    identification*, Journal of Sound and Vibration 432 (2018) 569–601.
"""

# %%
# 1. Imports
# ----------

import numpy as np
import matplotlib.pyplot as plt

from pysdkit.data import load_acycbd_sig2
from pysdkit.utils import acycbd, annotate_harmonics, envelope_spectrum, matlab_kurtosis

# %%
# 2. What the code computes
# -------------------------
#
# #. Demean :math:`x`. Initialise :math:`h` as a delayed impulse (:math:`h_2=1`).
# #. Build the unweighted correlation :math:`XX=\mathrm{CorrMatrix}(x)` once.
# #. Filter :math:`s=h*x` (causal ```lfilter```). Weights start as
#    :math:`W=|s_{N:L}|^p` with :math:`p=2`.
# #. :math:`\hat\alpha=\mathrm{EHPS}(s)` on the mean-centred Hilbert
#    envelope spectrum (DC dropped; product of :math:`K=10` harmonics;
#    search limited by MATLAB ```flim=300```).
# #. Project :math:`W` onto harmonics of :math:`\hat\alpha` (real Fourier
#    series) and zero bins below :math:`\mathrm{mean}+2\,\mathrm{std}`.
# #. Solve the largest-magnitude generalised eigenpair of
#    :math:`(XWX,\,XX)`; stop when
#    :math:`|\kappa-\kappa_{\mathrm{old}}|/|\kappa_{\mathrm{old}}|`
#    is small. The first error is :math:`\infty` because
#    :math:`\kappa_{\mathrm{old}}=0`.
# #. Return :math:`s=h*x` trimmed to the valid FIR region
#    (MATLAB ```s(N:end)```, length :math:`L-N+1`).
#
# EHPS needs a record of about one second at the given :math:`f_s` so that
# ```round(flim * fs / L)``` bins cover a few hundred hertz. It returns
# **one** fundamental; several incommensurate periods of similar
# strength will collapse to a single line.
#
# Use this when the interesting cycle lives in the **envelope** (AM of
# a resonant carrier). A pure sinusoid has a flat envelope, and EHPS
# is then unstable — SST / SET are the right tools for carrier
# instantaneous frequency.

# %%
# 3. MATLAB demo (```02 ACYCBD/demo.m```)
# ---------------------------------------
#
# Packaged ```sig2.mat```: 20000 samples, :math:`f_s=20\,\mathrm{kHz}`,
# inner-race frequency **BPFI = 47 Hz**. Call
# ```ACYCBD(x, fs, 40)```.

record = load_acycbd_sig2()
x = record["signal"] - np.mean(record["signal"])
fs = float(record["fs"])
t = record["t"]
bpfi = float(record["bpfi"])
print("N =", x.size, "fs =", fs, "BPFI =", bpfi)

freq_x, spec_x = envelope_spectrum(x, fs, scale="length")
fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), constrained_layout=True)
axes[0].plot(t, x, color="C0", lw=0.7)
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Raw data in time domain")
axes[1].plot(freq_x, spec_x, color="C0", lw=0.8)
axes[1].set_xlim(0, 200)
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel("Amplitude")
axes[1].set_title("Envelope spectrum of raw data")
plt.show()

# %%
# 4. Run ACYCBD
# -------------

fir, s, info = acycbd(x, fs, filter_size=40)
f_est = np.asarray(info["f_est"], dtype=float)
print("iterations =", info["count"], "kappa =", info["kappa"])
print("last cyclic frequency =", f_est[-1], "Hz")

t_s = np.arange(s.size) / fs
s_d = s - np.mean(s)
freq_s, spec_s = envelope_spectrum(s_d, fs, scale="length")

fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
ax.plot(np.arange(1, f_est.size + 1), f_est, "C0-o", ms=4, label="estimated")
ax.axhline(bpfi, color="C3", ls="--", marker="^", markevery=[0], label="BPFI")
ax.set_xlabel("Iteration")
ax.set_ylabel("Frequency [Hz]")
ax.set_title("Estimated cyclic frequency")
ax.legend()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), constrained_layout=True)
axes[0].plot(t_s, s, color="C0", lw=0.7)
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Deconvolved signal by ACYCBD")
axes[0].legend([f"Kurtosis={matlab_kurtosis(s):.3f}"], loc="upper right")
axes[1].plot(freq_s, spec_s, color="C0", lw=0.8)
axes[1].set_xlim(0, 200)
axes[1].set_xlabel("Frequency [Hz]")
axes[1].set_ylabel("Amplitude")
axes[1].set_title("Envelope spectrum of deconvolved signal")
annotate_harmonics(freq_s, spec_s, bpfi, n_harmonics=6, ax=axes[1], symbol=r"f_i")
plt.show()
