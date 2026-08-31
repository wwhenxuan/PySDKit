r"""
Entropy of modes
================

After a decomposition, each IMF is a short non-stationary series. Entropy
turns that series into a **scalar complexity feature**: a pure tone is
regular (low entropy), broadband noise is irregular (high entropy). That
is why EMD / VMD papers in bearing diagnosis and EEG almost always report
sample, fuzzy or dispersion entropy of the modes.

This example first contrasts a sine with Gaussian noise (waveforms, a
numeric table, then one panel per measure), then scores the IMFs of the
packaged VMD demo.

**References**

- Pincus, *Approximate entropy as a measure of system complexity*, PNAS (1991).
- Richman & Moorman, *Physiological time-series analysis using approximate
  entropy and sample entropy*, Am. J. Physiol. (2000).
- Chen et al., *Characterization of surface EMG signal based on fuzzy
  entropy*, IEEE TNSRE (2007).
- Rostaghi & Azami, *Dispersion entropy*, IEEE SPL (2016).
- Costa, Goldberger & Peng, *Multiscale entropy analysis of complex
  physiologic time series*, Phys. Rev. Lett. (2002).
"""

# %%
# 1. Imports
# ----------

import numpy as np
import matplotlib.pyplot as plt

from pysdkit import VMD
from pysdkit.data import test_vmd
from pysdkit.plot import plot_IMFs
from pysdkit.entropy import (
    permutation_entropy,
    sample_entropy,
    approximate_entropy,
    fuzzy_entropy,
    dispersion_entropy,
    spectral_entropy,
    distribution_entropy,
    increment_entropy,
    slope_entropy,
    symbolic_dynamic_entropy,
)

# %%
# 2. Principle: sine versus noise
# -------------------------------
#
# A 5 Hz cosine is almost periodic. White noise fills the delay-embedding
# and the spectrum. Compare **within** one measure only: SpecEn is scaled
# to ``[0, 1]``, IncrEn is in nats, DistEn is a normalized histogram
# entropy — they do not share a y-axis.

n = 400
t = np.linspace(0.0, 1.0, n, endpoint=False)
sine = np.sin(2.0 * np.pi * 5.0 * t)
noise = np.random.RandomState(0).randn(n)

fig, axes = plt.subplots(2, 1, figsize=(8, 3.6), sharex=True)
axes[0].plot(t, sine, color="#4169E1", lw=0.9)
axes[0].set_ylabel("sine")
axes[1].plot(t, noise, color="#FF8C00", lw=0.6)
axes[1].set_ylabel("noise")
axes[1].set_xlabel("time [s]")
fig.tight_layout()

# %%
# Ten measures on the same two records. Each panel has its own vertical
# scale so SpecEn is not crushed by IncrEn. DistEn is the exception: a
# pure sine's embedding-distance histogram is not always simpler than
# Gaussian noise.


def _pack(x):
    return {
        "PE": permutation_entropy(x, m=3, t=1)[0],
        "SampEn": sample_entropy(x, m=2, r=0.2),
        "ApEn": approximate_entropy(x, m=2, r=0.2),
        "FuzzEn": fuzzy_entropy(x, m=2, r=0.2),
        "DispEn": dispersion_entropy(x, m=2, c=3),
        "SpecEn": spectral_entropy(x),
        "DistEn": distribution_entropy(x, m=2),
        "IncrEn": increment_entropy(x, m=2),
        "SlopEn": slope_entropy(x, m=3),
        "SyDyEn": symbolic_dynamic_entropy(x, m=2, c=4),
    }


sine_h = _pack(sine)
noise_h = _pack(noise)
names = list(sine_h.keys())

print(f"{'measure':<8} {'sine':>10} {'noise':>10}")
for name in names:
    print(f"{name:<8} {sine_h[name]:10.4f} {noise_h[name]:10.4f}")

fig, axes = plt.subplots(2, 5, figsize=(11, 4.4))
for ax, name in zip(axes.ravel(), names):
    ax.bar(
        [0, 1],
        [sine_h[name], noise_h[name]],
        color=["#4169E1", "#FF8C00"],
        width=0.65,
    )
    ax.set_title(name)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sine", "noise"])
fig.tight_layout()

# %%
# 3. Entropy of VMD modes
# -----------------------
#
# ``test_vmd`` is the packaged multi-component record. VMD with ``K=4``
# returns four modes. ``plot_IMFs`` labels them ``IMF-0`` … ``IMF-3``.
# A band-limited oscillation should have **lower spectral entropy** than
# a broadband residual.

time, signal, fs = test_vmd()
vmd = VMD(alpha=2000, K=4, tau=0.0, tol=1e-7)
IMFs = vmd.fit_transform(signal)
plot_IMFs(signal, IMFs, view="2d_freq", fs=fs, freq_max=fs / 2)

# %%
# Score the raw mixture and each mode. The table is the quantitative
# result; the bars repeat SpecEn / SampEn / FuzzEn / DispEn.

rows = ["signal"] + [f"IMF-{i}" for i in range(IMFs.shape[0])]
series = [signal] + [IMFs[i] for i in range(IMFs.shape[0])]
spec = np.array([spectral_entropy(x) for x in series])
samp = np.array([sample_entropy(x, m=2, r=0.2) for x in series])
fuzz = np.array([fuzzy_entropy(x, m=2, r=0.2) for x in series])
disp = np.array([dispersion_entropy(x, m=2, c=3) for x in series])

print(f"{'series':<8} {'SpecEn':>9} {'SampEn':>9} {'FuzzEn':>9} {'DispEn':>9}")
for name, a, b, c, d in zip(rows, spec, samp, fuzz, disp):
    print(f" {name:<7} {a:9.4f} {b:9.4f} {c:9.4f} {d:9.4f}")

values = {"SpecEn": spec, "SampEn": samp, "FuzzEn": fuzz, "DispEn": disp}
fig, axes = plt.subplots(2, 2, figsize=(8, 4.6), sharex=True)
for ax, (title, y) in zip(axes.ravel(), values.items()):
    ax.bar(np.arange(len(rows)), y, color="#4169E1", width=0.65)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(rows, rotation=30, ha="right")
fig.suptitle("Complexity of the mixture and of each VMD mode")
fig.tight_layout()
