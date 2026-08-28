# Building the PySDKit documentation

This note is the rebuild checklist for the Sphinx site
([pysdkit.readthedocs.io](https://pysdkit.readthedocs.io/)).
Use it when you clone the repo on another machine or host.

Docs language is **English**. Example sources are **Python files**, not notebooks.

---

## Stack

| Piece | Choice |
|---|---|
| Builder | Sphinx ≥ 8 (`docs/source/conf.py`) |
| Theme | [pydata-sphinx-theme](https://pydata-sphinx-theme.readthedocs.io/) ≥ 0.16 |
| Hosting | Read the Docs (`.readthedocs.yaml`) |
| API | `autodoc` + `autosummary` + Napoleon (NumPy/Google docstrings) |
| Examples | [Sphinx-Gallery](https://sphinx-gallery.github.io/) (scikit-image style) |
| Extra extensions | `sphinx-copybutton`, `sphinx_design`, `sphinx_gitstamp`, `sphinx-github-changelog` |

Python **3.10** is what RTD uses. Locally 3.10+ is fine. The package itself claims `>=3.8`.

---

## Install and build (local)

From the **repository root**:

```bash
python -m pip install -U pip
pip install -e .
pip install -r docs/requirements.txt

# Windows PowerShell: avoid GUI backends hanging on plt.show()
# $env:MPLBACKEND = "Agg"

sphinx-build -b html docs/source docs/build/html
```

Open `docs/build/html/index.html`.

`docs/build/` is covered by the repo `build/` gitignore. Do not commit HTML output.

`conf.py` already calls `matplotlib.use("agg")` and sets `savefig.dpi = 100`.

---

## Files you must have

### Always needed to *configure* a build

```
.readthedocs.yaml                 # RTD: OS, Python 3.10, pip install . + docs reqs
docs/requirements.txt             # Sphinx and theme extras (not runtime deps)
docs/source/conf.py               # Sphinx + gallery + theme options
docs/source/index.rst             # Hidden toctree = top navbar
pyproject.toml                    # pip install . / -e .
pysdkit/                          # autodoc imports this package
```

### Hand-written pages

```
docs/source/user_guide/index.rst
docs/source/API/modules.rst       # API toctree
docs/source/API/pysdkit*.rst      # one thin page per public family
docs/source/release_notes/index.rst  # changelog:: pulls GitHub Releases
docs/source/development/index.rst
docs/source/about/index.rst
```

Navbar order comes from the hidden toctree in `docs/source/index.rst`:

`user_guide` → `auto_examples` → `API/modules` → `release_notes` → `development` → `about`

### Theme / branding

```
docs/source/_static/logo.png              # html_logo
docs/source/_static/logo-pypi.svg         # PyPI icon in the header
docs/source/_static/custom.css            # pydata CSS variables (header blue, table zebra)
docs/source/_static/theme_overrides.css   # code-block gray, active tab, gallery cards
docs/source/_static/version_switcher.json # unused (version switcher is off)
```

`html_css_files` loads **overrides after** `custom.css`. Put high-specificity tweaks in `theme_overrides.css`.

Light-theme tokens that matter:

- Header / “on-background”: `--pst-color-on-background` (`#9cccf8`)
- Active nav item: `--pst-color-secondary` (`#8045e5`) — primary `#91a9f6` is too close to the bar
- Inline code / tables: `#f3f4f5`

API module pages hide the left sidebar (`html_sidebars` for `API/pysdkit.*`). Gallery pages keep it.

### Examples (Sphinx-Gallery)

**Input (commit these):**

```
examples/GALLERY_HEADER.rst       # landing-page title
examples/<section>/GALLERY_HEADER.rst
examples/<section>/*.py           # one script per card; start with a title docstring
docs/ipynb_to_gallery.py          # one-shot: old .ipynb → gallery .py
```

**Output (gitignored — not on the main git tree):**

```
docs/source/auto_examples/        # rst, png, md5, downloadable .py/.ipynb/.zip
docs/source/sg_execution_times.rst
```

Those files are packed as `auto_examples.tar.gz` and stored on a rolling
**GitHub Release** tagged `gallery-cache` (a prerelease, not a software
version). Read the Docs downloads the tarball in `pre_build`, Sphinx-Gallery
skips examples whose `.md5` still matches, and the built HTML (including
PNGs) is what RTD actually serves.

A generic cloud drive does **not** help by itself: Sphinx needs the PNG +
`.md5` files on the **build machine**, not a public image URL in the
browser.

Gallery config (in `conf.py`):

```python
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",   # relative to docs/source/
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py$",        # execute all .py, not only plot_*
    "nested_sections": True,             # each subfolder = sidebar chapter
    "download_all_examples": True,
    "reset_modules": ("matplotlib",),
    "abort_on_example_error": False,
}
```

Sphinx-Gallery only supports **one** level of subfolders under `examples/`.

---

## Read the Docs

`.readthedocs.yaml` does:

1. Ubuntu 24.04, Python 3.10
2. `pre_build`: download `auto_examples.tar.gz` from the `gallery-cache` GitHub Release (if it exists)
3. `pip install .` (the checkout, **not** PyPI `pysdkit`)
4. `pip install -r docs/requirements.txt`
5. HTML only (`formats: []` — skip PDF/ePub)
6. Sphinx with `docs/source/conf.py` (only examples whose source `.md5` changed are re-run)

Community RTD is **15 minutes** per build. A cold gallery run of all 73
examples can exceed that. The GitHub Release cache is what keeps ordinary
docs builds inside the limit (same idea as committing `auto_examples/`,
without putting 90 MB of PNGs on the default git clone).

Refresh the cache after example (or plot-changing) edits:

```bash
# local, if docs/source/auto_examples/ already exists
tar -czf auto_examples.tar.gz docs/source/auto_examples docs/source/sg_execution_times.rst
gh release create gallery-cache auto_examples.tar.gz --prerelease \
  --title "Sphinx-Gallery cache" \
  --notes "Generated example figures for Read the Docs. Not a software release."
# later refreshes:
gh release upload gallery-cache auto_examples.tar.gz --clobber
```

Or on GitHub: **Actions → Gallery cache → Run workflow**
(`.github/workflows/gallery-cache.yml` also runs on pushes that touch
`examples/`). That workflow can take ~20+ minutes; RTD does not wait for it.
If you change one example and push before the cache job finishes, RTD still
downloads the *previous* tarball and re-executes only the changed script.

If the `gallery-cache` release is missing, RTD falls back to a full gallery
run (and may time out). Email support@readthedocs.org only if you need a
longer timeout for that cold path.

`emd/emd_forecasting.py` needs **scikit-learn** (listed in `docs/requirements.txt`). Runtime package deps (numpy, scipy, matplotlib, …) come from `pip install .`.

---

## Incremental vs full gallery rebuild

| Situation | What happens |
|---|---|
| Local rebuild, unchanged `examples/**/*.py`, existing `auto_examples/*.md5` | Skip execution; reuse figures |
| RTD with a current `gallery-cache` tarball, unchanged examples | Skip execution; reuse figures |
| Edited one example `.py` | Re-run that file only (md5 mismatch) |
| Fresh clone with no local `auto_examples/` and no Release | Full gallery run |
| Force one local file | Delete its `.md5` under `auto_examples/`, or change a newline in the `.py` |

Skip execution entirely (layout only, no figures):

```bash
sphinx-build -D sphinx_gallery_conf.plot_gallery=False -b html docs/source docs/build/html
```

---

## Adding or changing content

### New algorithm → API page

1. Implement and export from the subpackage `__init__.py`.
2. Re-export from `pysdkit` and add to `__all__`.
3. Add a thin `autoclass` / `autofunction` line on the matching `docs/source/API/pysdkit._*.rst` (or a new family page + entry in `API/modules.rst`).
4. Demo data: `pysdkit.data` loaders; `.npy` files live in `pysdkit/data/real_world`.

Do **not** import `repo/` (gitignored MATLAB/paper tree) from docs or examples.

### New gallery example

Write `examples/<section>/my_algo.py` (not a notebook):

```python
r"""
My algorithm
============

Theory paragraph. Use RST (``:math:`x` ``, ``.. math::``).
"""

# %%
# More explanation
# ----------------

import numpy as np
from pysdkit import EMD
```

- First block **must** be a title docstring with an underline (`====`).
- Use a **raw** docstring (`r"""`) so LaTeX backslashes are valid Python.
- Do **not** inject `sys.path` and do **not** use `__file__` or a `ROOT = Path.cwd()` hack. Sphinx-Gallery `exec`s the script; `__file__` is undefined.
- Prefer `pysdkit.data` over local `.mat` files.
- New folder: add `examples/<section>/GALLERY_HEADER.rst` with a title.

Legacy notebooks:

```bash
python docs/ipynb_to_gallery.py           # writes .py and deletes .ipynb
python docs/ipynb_to_gallery.py --keep-ipynb
```

### Release notes

Do not hand-edit `docs/source/release_notes/index.rst` for each version.
Write the notes in the **GitHub Release** body (Markdown) for that tag.
Sphinx pulls them at build time via `.. changelog::`.

On Read the Docs: **Admin → Environment variables** →
`SPHINX_GITHUB_CHANGELOG_TOKEN`.

### Theme tweaks

| Want to change | File |
|---|---|
| Header / table / token colors | `docs/source/_static/custom.css` |
| Code-block background, active tab, gallery cards | `docs/source/_static/theme_overrides.css` |
| Logo | `docs/source/_static/logo.png` + `html_logo` in `conf.py` |
| Navbar labels / order | titles of the rst files in `index.rst` toctree |

---

## Pitfalls

1. **Installing PyPI `pysdkit` instead of the checkout** — autodoc and examples will see an old package. Always `pip install -e .` (local) or `pip install .` (RTD).
2. **`plt.show()` without Agg** — hangs a local smoke-test of a gallery script. Docs builds set Agg in `conf.py`.
3. **Huge figures** — new runs use `savefig.dpi = 100`. If a card image is enormous, shrink the `figsize` in that script. Do not commit `auto_examples/` to GitHub.
4. **Gallery failure still fails the Sphinx build** — `abort_on_example_error` is `False` so later examples still run, but Sphinx-Gallery raises `ExtensionError` at the end if any script failed unexpectedly. Fix the traceback (see `docs/build/html` log or the Sphinx temp log).
5. **RST warnings** from markdown→RST conversion (`unindent`, unmatched `` ` ``) do not fail the build unless you pass `-W`.
6. **`autosectionlabel`** uses `autosectionlabel_prefix_document = True` because many examples share headings like “1. Imports”.
7. **Release notes need GitHub** — `sphinx-github-changelog` calls the Releases API at build time. Public anonymous access works until you hit the rate limit (60/hour). On Read the Docs, set environment variable `SPHINX_GITHUB_CHANGELOG_TOKEN` (or `GITHUB_TOKEN`) to a PAT. Do not commit the token.

---

## `docs/requirements.txt`

```
sphinx>=8.0
sphinx-copybutton
sphinx_design>=0.5
pydata-sphinx-theme>=0.16
sphinx_gitstamp
sphinx-gallery>=0.18
ipython
nbformat
pillow
scikit-learn
sphinx-github-changelog>=2.3
```

`ipython` + `nbformat` are for generated notebook downloads. `pillow` is for gallery image handling. `scikit-learn` is only for the EMD forecasting example. `sphinx-github-changelog` fills Release notes from GitHub. Pre-releases (including the `gallery-cache` asset) are omitted (`sphinx_github_changelog_include_prereleases = False`).

---

## Quick command card

```bash
# editable install + docs deps
pip install -e . -r docs/requirements.txt

# normal HTML
sphinx-build -b html docs/source docs/build/html

# treat warnings as errors (strict; gallery RST noise may fail)
sphinx-build -W -b html docs/source docs/build/html

# convert leftover notebooks (if any)
python docs/ipynb_to_gallery.py

# pack and upload the RTD gallery cache (after a local sphinx-build)
tar -czf auto_examples.tar.gz docs/source/auto_examples docs/source/sg_execution_times.rst
gh release upload gallery-cache auto_examples.tar.gz --clobber
```

Preview: `docs/build/html/index.html`  
Gallery: `docs/build/html/auto_examples/index.html`
