r"""
Local Mean Decomposition (LMD)
==============================

.. epigraph::

    **Classical algorithm:** Smith, J. S. (2005). *The local mean decomposition and its application to EEG perception data.* Journal of the Royal Society Interface, 2(5), 443–454.  
    https://doi.org/10.1098/rsif.2005.0058

.. epigraph::

    **Python reference (this implementation):** `PyLMD <https://github.com/shownlin/PyLMD>`_ — vectorised moving-average LMD.

.. epigraph::

    **Related (EOE-LMD):** Jia et al., *Digital Signal Processing* 87 (2019) — empirical optimal envelopes (``eoe_lmd.m``). That method replaces the Smith moving-average envelope; **PySDKit ``LMD`` follows classical Smith / PyLMD**, not EOE-LMD.

Motivation
----------

Many non-stationary records are **AM–FM** mixtures. EMD extracts IMFs by spline envelopes;
**LMD** instead builds a **local mean** and **local envelope** from successive extrema,
smooths them with a moving average, and separates each component into

.. math::

   \mathrm{PF}_i(t)=a_i(t)\,s_i(t),

where :math:`a_i(t)\ge 0` is a slowly varying amplitude and :math:`s_i(t)` is a **pure FM**
carrier with :math:`|s_i|\approx 1`. Instantaneous frequency can then be read from :math:`s_i`
(e.g. Hilbert / DQ methods).
"""

# %%
# 1. Algorithm and core formulas
# ------------------------------
#
# 1.1 Outer decomposition
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# .. math::
#
#    x(t)=\sum_{i=1}^{K}\mathrm{PF}_i(t)+u_K(t).
#
# Stop when :math:`u_K` is **monotonic**, has too few extrema, or :math:`K` PFs are reached.
#
# 1.2 Local mean and envelope (Smith)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Between successive extrema at samples :math:`n_k`, :math:`n_{k+1}`:
#
# .. math::
#
#    \bar m_k=\frac{x(n_k)+x(n_{k+1})}{2},
#    \qquad
#    \bar a_k=\frac{\bigl|x(n_k)-x(n_{k+1})\bigr|}{2}.
#
# Hold these values as a piecewise-constant (“square”) sequence, then **smooth**
# with a triangular moving average of window
#
# .. math::
#
#    W=\Bigl\lfloor \frac{\max_k(n_{k+1}-n_k)}{3}\Bigr\rfloor
#
# (forced odd, :math:`\ge 3`) until consecutive samples are no longer identical.
#
# 1.3 Inner sifting → pure FM
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Starting from :math:`h^{(0)}=x` (or the current residual):
#
# .. math::
#
#    s^{(j)}=\frac{h^{(j)}-m^{(j)}}{a^{(j)}}.
#
# Set :math:`h^{(j+1)}\leftarrow s^{(j)}` and repeat until :math:`a^{(j)}\approx 1` (pure FM) or
# the modulation converges. Accumulate envelopes
#
# .. math::
#
#    A(t)=\prod_{j}a^{(j)}(t),
#    \qquad
#    \mathrm{PF}(t)=A(t)\,s^{(\mathrm{final})}(t).
#
# 1.4 Stopping (inner)
# ~~~~~~~~~~~~~~~~~~~~
#
# .. list-table::
#    :header-rows: 1
#
#    * - Criterion
#      - Formula (PyLMD defaults)
#    * - Pure FM
#      - :math:`\frac1N\sum\|1-a(t)\|\le 0.01`
#    * - Convergence
#      - :math:`\frac1N\sum\|h-s\|\le 0.01`
#    * - Extrema
#      - :math:`\#\mathrm{extrema}\le 3`

# %%
# 2. Imports
# ----------

import numpy as np
import matplotlib.pyplot as plt

from pysdkit import LMD
from pysdkit.data import test_emd
from pysdkit.plot import plot_IMFs

plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 11
print(LMD())

# %%
# 3. Example A — PyLMD multi-tone demo
# ------------------------------------
#
# The signal used in the PyLMD package example:
#
# .. math::
#
#    y(x)=\tfrac23\sin(30x)+\tfrac23\sin(17.5x)+\tfrac45\cos(2x),
#    \quad x\in[0,100].

x = np.linspace(0.0, 100.0, 401)
y = (2 / 3) * np.sin(x * 30) + (2 / 3) * np.sin(x * 17.5) + (4 / 5) * np.cos(x * 2)

lmd = LMD(K=8)
pfs = lmd.fit_transform(y)
print("PFs + residue shape:", pfs.shape)
print("reconstruction error:", np.linalg.norm(y - pfs.sum(0)))

fig, ax = plt.subplots(figsize=(10, 2.8))
ax.plot(x, y, lw=0.9)
ax.set_title("Original multi-tone signal")
ax.set_xlabel("$x$")
fig.tight_layout()
plt.show()

_ = plot_IMFs(y, pfs)

# %%
# 4. Example B — library ``test_emd`` mixture
# -------------------------------------------
#
# A standard three-tone test used across PySDKit EMD-family notebooks.

t, signal = test_emd()
pfs2 = LMD(K=5).fit_transform(signal)

print("shape:", pfs2.shape)
print("‖x − Σ PF‖ =", np.linalg.norm(signal - pfs2.sum(0)))

fig, axes = plt.subplots(
    pfs2.shape[0] + 1, 1, figsize=(10, 1.35 * (pfs2.shape[0] + 1)), sharex=True
)
axes[0].plot(t, signal, "k", lw=0.9)
axes[0].set_ylabel("$x$")
for i in range(pfs2.shape[0] - 1):
    axes[i + 1].plot(t, pfs2[i], lw=0.9)
    axes[i + 1].set_ylabel(f"PF{i+1}")
axes[-1].plot(t, pfs2[-1], lw=0.9)
axes[-1].set_ylabel("res")
axes[-1].set_xlabel("Time")
fig.suptitle("LMD on test_emd")
fig.tight_layout()
plt.show()

# %%
# 5. One sifting step visualised
# ------------------------------
#
# For the first PF of the multi-tone signal: extrema → square mean/envelope → smoothed :math:`m(t)`, :math:`a(t)`.

lmd_vis = LMD()
extrema = lmd_vis.find_extrema(y)
m0, m, a0, a = lmd_vis.local_mean_and_envelope(y, extrema)

fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
axes[0].plot(x, y, "k", lw=0.8, label="signal")
axes[0].plot(x[extrema], y[extrema], "ro", ms=3, label="extrema")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].set_ylabel("$y$")

axes[1].plot(x, m0, lw=0.7, alpha=0.5, label="square $m$")
axes[1].plot(x, m, lw=1.0, label="smoothed $m$")
axes[1].legend(loc="upper right", fontsize=8)
axes[1].set_ylabel("local mean")

axes[2].plot(x, a0, lw=0.7, alpha=0.5, label="square $a$")
axes[2].plot(x, a, lw=1.0, label="smoothed $a$")
axes[2].legend(loc="upper right", fontsize=8)
axes[2].set_ylabel("local envelope")
axes[2].set_xlabel("$x$")
fig.suptitle("Smith LMD: extrema → square signals → moving-average smooth")
fig.tight_layout()
plt.show()

# %%
# 6. API cheat-sheet
# ------------------
#
# .. list-table::
#    :header-rows: 1
#
#    * - Parameter
#      - Default
#      - Role
#    * - ``K``
#      - 5
#      - max number of PFs (residue always appended)
#    * - ``endpoints``
#      - ``True``
#      - treat ends as pseudo-extrema
#    * - ``max_smooth_iter``
#      - 12
#      - MA iterations (PyLMD default)
#    * - ``max_envelope_iter``
#      - 200
#      - inner sifting cap
#    * - ``envelope_epsilon``
#      - 0.01
#      - pure-FM stop
#    * - ``convergence_epsilon``
#      - 0.01
#      - modulation stop
#    * - ``min_extrema``
#      - 5
#      - outer-loop extrema floor
#
# .. code-block:: python
#
#    from pysdkit import LMD
#    pfs = LMD(K=5)(signal)   # shape (n_pf+1, N); last row = residue
