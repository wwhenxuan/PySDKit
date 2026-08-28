r"""
Adaptive Local Iterative Filtering (ALIF)
=========================================

This notebook introduces the main ideas behind **Adaptive Local Iterative Filtering (ALIF)** and provides executable examples with PySDKit.

**Reference**

.. epigraph::

    A. Cicone, J. Liu, H. Zhou.  
    *Adaptive Local Iterative Filtering for Signal Decomposition and Instantaneous Frequency analysis.*  
    Applied and Computational Harmonic Analysis, 41(2):384–411, 2016.  
    `arXiv:1411.6051 <https://arxiv.org/abs/1411.6051>`_

The implementation follows the authors' open MATLAB code: `Cicone/ALIF <https://github.com/Cicone/ALIF>`_.
"""

# %%
# 1. Why ALIF?
# ------------
#
# Time–frequency analysis of **nonlinear and non-stationary** signals is challenging. Classical Fourier and wavelet methods rely on predetermined bases and are not fully data-adaptive, so they often struggle to track instantaneous frequency variations.
#
# A powerful strategy is to first decompose a signal into simpler components—**Intrinsic Mode Functions (IMFs)**—and then analyze each component separately. Empirical Mode Decomposition (EMD; Huang et al., 1998) is the classical iterative approach, but it uses cubic-spline envelopes and can be sensitive to perturbations.
#
# **Iterative Filtering (IF)** keeps the same sifting framework as EMD, but replaces envelope averaging by convolution with a low-pass filter, which improves stability.  
# **ALIF** extends IF with two key ingredients:
#
# #. **Smooth compactly supported filters** (e.g., Fokker–Planck / FP filters)
# #. A **spatially adaptive mask length** that follows local oscillations in non-stationary data
#
# The resulting decomposition reads
#
# .. math::
#
#    f(x)=\sum_{j=1}^{m} I_j(x)+r(x),
#
# where :math:`I_j` are IMFs and :math:`r` is a trend (at most one extremum).

# %%
# 2. From IF to ALIF: core formulas
# ---------------------------------
#
# 2.1 Moving average and fluctuation extraction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Let :math:`L` be a moving-average operator and :math:`S` the fluctuation operator:
#
# .. math::
#
#    L(f)(x)=\int_{-l}^{l} f(x+t)\,w(t)\,dt,
#    \qquad
#    S(f)=f-L(f)=(I-L)(f).
#
# Here :math:`w` is a symmetric, compactly supported low-pass kernel and :math:`l` is the (half) mask length.
#
# The inner loop repeatedly applies :math:`S`:
#
# .. math::
#
#    f_{n+1}=S(f_n)=f_n-L(f_n),
#
# until :math:`L(f_n)` is sufficiently small. The limit is taken as one IMF.
#
# 2.2 Two nested loops
# ~~~~~~~~~~~~~~~~~~~~
#
# * **Inner loop**: with a fixed (or locally prescribed) filter length, iterate :math:`f\leftarrow f-L(f)` to extract one IMF.  
# * **Outer loop**: subtract the IMF from the residual and continue until the residual is essentially a trend.
#
# In classical IF, a common global mask-length estimate is (paper Eq. (5))
#
# .. math::
#
#    l_n := 2\left\lfloor \chi\,\frac{N}{k}\right\rfloor,
#
# where :math:`N` is the number of samples, :math:`k` is the number of extrema, and :math:`\chi\approx 1.6`.
#
# 2.3 ALIF: locally adaptive mask length
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When frequency changes over time, a single global :math:`l` is insufficient. ALIF makes the mask length a function of position, :math:`l(x)`:
#
# .. math::
#
#    L(f)(x)=\int_{-l(x)}^{l(x)} f(x+t)\,w_{l(x)}(t)\,dt.
#
# In practice one typically:
#
# #. interpolates inter-extrema spacings to obtain an instantaneous-period curve :math:`T(x)`;
# #. smooths :math:`T(x)` with IF and scales it (about :math:`2\xi` in the reference code) to get :math:`l(x)`;
# #. applies a local average at each sample with the corresponding filter length.
#
# A standard inner stopping criterion is the relative energy
#
# .. math::
#
#    \mathrm{SD}=\frac{\|L(f)\|_2^2}{\|f\|_2^2}<\delta.
#
# 2.4 Filters (FP filters)
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# The paper stresses that the kernel should be **compactly supported, smooth, and vanishing smoothly at both ends**, to avoid artificial oscillations during iteration.  
# Such filters are constructed from Fokker–Planck equations and satisfy the derived sufficient conditions for IF convergence.  
# PySDKit ships the same ``prefixed_double_filter`` used by the MATLAB reference implementation.

# %%
# 3. Algorithm sketch
# -------------------
#
# .. code-block:: text
#
#    Input:  signal f
#    Output: IMFs = {I1,...,Im, r}
#
#    while (#extrema of f > ExtPoints) and (#IMFs < NIMFs):
#        1) estimate instantaneous period T(x) from extrema; smooth it to get l(x)
#        2) h <- f
#        3) while SD > delta:
#               ave <- local average of h with mask length l(x)
#               SD  <- ||ave||^2 / ||h||^2
#               h   <- h - ave
#        4) store h as a new IMF; f <- f - h
#    append the residual f as the trend r
#
# Below we reproduce the two-chirp experiment from the paper / MATLAB ``Example.m`` using PySDKit's ``ALIF``.

import numpy as np
import matplotlib.pyplot as plt

from pysdkit import ALIF
from pysdkit.plot import plot_IMFs, plot_signal

np.set_printoptions(precision=4, suppress=True)
plt.rcParams["figure.figsize"] = (9, 3.2)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# %%
# 4. Build a demo signal (shortened MATLAB ``Example.m``)
# -------------------------------------------------------
#
# The official example superposes two quadratic-phase chirps and a DC offset:
#
# .. math::
#
#    x(t)=\cos\!\left(-\tfrac12\frac{D}{T}t^2-4t\right),
#    \quad
#    y(t)=\cos\!\left(-\tfrac12\frac{D}{T}t^2-20t\right),
#    \quad
#    f(t)=x(t)+y(t)+1.
#
# The component :math:`y` oscillates faster and is typically extracted first as IMF:math:`_1`.

def make_alif_demo_signal(n: int = 512, D: float = 16.0):
    """Two nonlinear chirps + DC"""
    T = 2.0 * np.pi
    t = np.linspace(0.0, T, n, endpoint=False)
    x = np.cos(-0.5 * D / T * t**2 - 4.0 * t)
    y = np.cos(-0.5 * D / T * t**2 - 20.0 * t)
    f = x + y + 1.0
    return t, f, x, y

t, signal, true_slow, true_fast = make_alif_demo_signal(n=512)
print(
    f"signal shape = {signal.shape}, range = [{signal.min():.3f}, {signal.max():.3f}]"
)

fig = plot_signal(t, signal)
fig.suptitle("Input: two chirps + DC", fontsize=12)
plt.tight_layout()
plt.show()

# %%
# 5. Run ALIF
# -----------
#
# Main hyperparameters:
#
# .. list-table::
#    :header-rows: 1
#
#    * - Parameter
#      - Meaning
#      - Typical value
#    * - ``max_imfs``
#      - maximum number of IMFs (excluding the trend)
#      - 1–3 (demo)
#    * - ``delta``
#      - inner relative-energy threshold :math:`\delta`
#      - :math:`10^{-4}`
#    * - ``xi``
#      - mask-length scale :math:`\xi` (MATLAB ``ALIF.xi``)
#      - :math:`\approx 1.6`
#    * - ``max_inner``
#      - maximum inner iterations
#      - tens to hundreds
#
# .. epigraph::
#
#     Note: the current implementation assumes a **periodic extension** of the signal, consistent with the reference MATLAB code.

alif = ALIF(
    max_imfs=2,
    delta=1e-4,
    xi=1.6,
    max_inner=60,
    verbose=0,
)

imfs, mask_lengths = alif.fit_transform(signal, return_masks=True)

print("IMFs shape:", imfs.shape, "  (last row = trend)")
print("mask_lengths shape:", mask_lengths.shape)

recon = imfs.sum(axis=0)
recon_err = np.linalg.norm(recon - signal) / np.linalg.norm(signal)
print(f"reconstruction relative error = {recon_err:.3e}")

# %%
# 6. Visualize the decomposition
# ------------------------------

plot_IMFs(signal, imfs)
plt.suptitle("ALIF modes (last component is the trend)", fontsize=12)
plt.show()

# %%
# Comparison with ground-truth components
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Typically the first IMF matches the faster chirp :math:`y(t)`, while the sum of the remaining components matches the slower chirp plus DC, :math:`x(t)+1`.

fig, axes = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True)

axes[0].plot(t, true_fast, "b-", lw=1.8, label="true fast chirp $y(t)$")
axes[0].plot(t, imfs[0], "r--", lw=1.5, label="ALIF IMF$_1$")
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Fast chirp vs first IMF")
axes[0].legend(loc="best", fontsize=9)

axes[1].plot(t, true_slow + 1.0, "b-", lw=1.8, label="true slow chirp + DC")
axes[1].plot(t, imfs[1:].sum(axis=0), "r--", lw=1.5, label="remaining modes + trend")
axes[1].set_xlabel("Time $t$")
axes[1].set_ylabel("Amplitude")
axes[1].set_title("Slow component vs residual sum")
axes[1].legend(loc="best", fontsize=9)

plt.tight_layout()
plt.show()

if1_err = np.linalg.norm(imfs[0] - true_fast) / np.linalg.norm(true_fast)
rest_err = np.linalg.norm(imfs[1:].sum(0) - (true_slow + 1.0)) / np.linalg.norm(
    true_slow + 1.0
)
print(f"IMF1 vs fast chirp relative error  = {if1_err:.4f}")
print(f"rest  vs slow+DC   relative error  = {rest_err:.4f}")

# %%
# 7. Adaptive mask length :math:`l(x)`
# ------------------------------------
#
# The distinctive feature of ALIF is that the filtering window varies with position. The next cell plots the mask-length function associated with the first IMF (if at least one IMF was extracted).

if mask_lengths.size > 0:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, mask_lengths[0], "b-", lw=1.8)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("mask length $l(x)$")
    ax.set_title("Adaptive mask length for IMF$_1$")
    plt.tight_layout()
    plt.show()
    print(
        "mask length stats: "
        f"min={mask_lengths[0].min():.2f}, "
        f"mean={mask_lengths[0].mean():.2f}, "
        f"max={mask_lengths[0].max():.2f}"
    )
else:
    print("No IMF mask was produced (signal may already be trend-like).")

# %%
# 8. Takeaways
# ------------
#
# * **IF** replaces EMD envelope averaging by a convolutional moving average :math:`L(f)=f*w`, and iterates :math:`f\leftarrow f-L(f)` to obtain IMFs.  
# * **ALIF** further lets the mask length :math:`l(x)` **adapt to the data**, which is better suited to signals with strongly varying instantaneous frequency.  
# * In PySDKit, use ``from pysdkit import ALIF``. The filter file ``prefixed_double_filter.npy`` is installed with the package.
#
# Further reading
# ~~~~~~~~~~~~~~~
#
# #. Cicone, Liu, Zhou, ACHA 2016 / arXiv:1411.6051  
# #. Cicone, *Nonstationary signal decomposition for dummies*, arXiv:1710.04844  
# #. Cicone & Zhou, FFT-based IF implementations, arXiv:1802.01359
