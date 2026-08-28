# Contributing to PySDKit

Thank you for helping grow PySDKit. This page is the contributor guide for
the GitHub project; the same material lives in the documentation under
[Development](https://pysdkit.readthedocs.io/en/stable/development/index.html).

## Why PySDKit exists

**Signal decomposition** treats a non-stationary recording as a sum of simpler
modes (intrinsic mode functions). Those modes are a practical
time-frequency / feature-engineering tool: they can be plotted, used as
features, or fed to a downstream model.

Fourier analysis assumes stationarity; fixed wavelet dictionaries still
impose a predefined basis. Since the Hilbert–Huang Transform (1998), a
family of adaptive univariate and multivariate decompositions has grown
quickly in MATLAB, but Python (the usual home of machine learning) still
lacks a library comparable to [PyWavelets](https://pywavelets.readthedocs.io/)
for this class of algorithms.

PySDKit was started in April 2024 to close that gap: one package, a
shared `fit_transform` style API, and enough visualization that
decomposition can sit next to NumPy, SciPy, and neural-network stacks
without a MATLAB detour.

Please cite the **original paper** of each algorithm you use; class and
module docstrings point to those references.

## Project layout: where a new algorithm belongs

Implementation lives under `pysdkit/`. Algorithm families use a leading
underscore (`pysdkit/_emd`, `pysdkit/_vmd`, …). Shared helpers live
beside them (`pysdkit/data`, `pysdkit/plot`, `pysdkit/utils`,
`pysdkit/tsa`, `pysdkit/entropy`, `pysdkit/models`).

**Put a new method in the family it resembles.** If the principle is
close to an existing package, add a module there and export it from that
package’s `__init__.py`. If the idea is genuinely different, create a
new package (for example `pysdkit/_yourmethod/`) rather than stretching
an unrelated folder.

| Package | Typical contents | Example gallery folder |
|---|---|---|
| `pysdkit/_emd` | EMD, EEMD, CEEMDAN, REMD, SEMD, TVF-EMD, ESMD, HHT; also MEMD / APITMEMD / NSTEMD | `examples/emd`, `examples/emd_variants`, `examples/memd` |
| `pysdkit/_emd2d` | EMD2D, BMEMD | `examples/image` |
| `pysdkit/_ewt` | EWT, EWT2D, EFD | `examples/ewt` |
| `pysdkit/_faemd` | FAEMD (1-D / 2-D / 3-D) | `examples/faemd` |
| `pysdkit/_vmd` | VMD, MVMD, OVMD, STVMD, SVMD, VME | `examples/vmd` |
| `pysdkit/_vmd2d` | VMD2D, CVMD2D | `examples/image` |
| `pysdkit/_acmd` | ACMD, BA-ACMD, DD-ACMD | `examples/acmd` |
| `pysdkit/_vncmd` | VNCMD, AVNCMD, STNBMD | `examples/vncmd` |
| `pysdkit/_gdmd` | GDMD, VGNMD, IVGNMD, AGNCMD | `examples/gdmd` |
| `pysdkit/_tid` | ALIF, FMD, HVD, ITD, SSA | `examples/temp_iter`, `examples/ssa` |
| `pysdkit/_tfa` | SST, SET, VTFMTD | `examples/tfa` |
| `pysdkit/_lmd` | LMD, RLMD | `examples/lmd` |
| `pysdkit/_osd` | OSD, SWD | `examples/osd` |
| `pysdkit/_imd` | IMD, APMD | `examples/imd` |
| `pysdkit/_jmd` | JMD, SJMD | `examples/jmd` |
| `pysdkit/tsa` | STL, MSTL, moving-average decomposition | `examples/tsa` |
| `pysdkit/utils` | BSS, deconvolution, kurtogram | `examples/utils`, `examples/deconvolution` |

After the implementation:

1. Export the public class / function from the family `__init__.py`.
2. Re-export it from `pysdkit/__init__.py` and add the name to `__all__`.
3. Add a thin `autoclass` / `autofunction` line on the matching page under
   `docs/source/API/`.
4. Prefer `pysdkit.data` loaders for demo arrays; put `.npy` files in
   `pysdkit/data/real_world`.

Do not import the gitignored `repo/` tree (local MATLAB / paper sources)
from installed code or from examples.

## Tests

Please add unit tests in `pysdkit/tests/` so the public surface stays
stable. Name the file `test_<algo>.py` and follow the existing
`unittest` style (see `pysdkit/tests/test_emd.py`, `test_vmd.py`).

Aim to cover **every public class and method** you introduce, not only a
single happy-path call. A useful minimum is:

- construction / default parameters
- `fit_transform` (shape: modes × samples, length matches the input)
- `__call__` if the class is callable
- functional aliases (`emd`, `vmd`, …) when you export them
- invalid inputs (`ValueError` / `TypeError`)
- a trivial or reconstructible signal when the algorithm claims completeness

Use `pysdkit.data` generators (`test_emd`, `test_univariate_signal`, …)
instead of huge private fixtures.

```bash
python -m unittest discover -s pysdkit/tests -p "test_*.py" -v
```

## Examples

Please ship a gallery script that shows the algorithm on a small demo
(theory plus figures). Examples are **Python files**, not notebooks.

- Put the script in the matching folder under `examples/` (see the table
  above). A new family gets a new folder and a `GALLERY_HEADER.rst` with
  a title.
- Start with a **raw** title docstring (`r"""..."""`). Extra theory goes
  in `# %%` comment blocks (Sphinx-Gallery turns those into HTML text and
  into markdown cells in the generated `.ipynb` download).
- Do **not** inject `sys.path` and do **not** use `__file__`. The docs
  build already installs the package.
- Prefer `pysdkit.data` over local `.mat` files.

```python
r"""
My algorithm
============

Short theory paragraph and a paper reference.
"""

# %%
# Demo
# ----

import numpy as np
from pysdkit import EMD
from pysdkit.plot import plot_IMFs
```

Look at `examples/emd/emd.py` or `examples/vmd/` for the expected tone:
motivation, a compact statement of the method, then executable plots.

The docs gallery is generated by Sphinx-Gallery at build time. Do **not**
commit `docs/source/auto_examples/`. Figures are packed into a GitHub
Release (`gallery-cache`) so Read the Docs can skip unchanged examples;
see [`.github/docs-build.md`](.github/docs-build.md).

## Pull requests

- Keep the change focused: one algorithm (or one tightly related family)
  per PR is easier to review.
- Run the unit tests before you open the PR.
- English for public docstrings, gallery headers, and commit messages.
