# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date
import sys
from pathlib import Path

import matplotlib

matplotlib.use("agg")
matplotlib.rcParams["savefig.dpi"] = 100
matplotlib.rcParams["figure.max_open_warning"] = 40

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PYSDKIT"

copyright = f"2025-{date.today().year}, the pysdkit team"
author = "the pysdkit team"

sys.path.insert(0, str(Path("..", "..").resolve()))

import pysdkit

release = pysdkit.__version__
version = pysdkit.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx.ext.napoleon",
    "sphinx_gitstamp",
    "sphinx_gallery.gen_gallery",
    "sphinx_github_changelog",
]

autosectionlabel_prefix_document = True

# The gallery-cache GitHub Release is a prerelease; keep it off this page.
sphinx_github_changelog_include_prereleases = False

# Subsection paths are relative to this file (docs/source/conf.py).
# First eight families are the ones we want at the top of Examples;
# the rest stay A–Z by folder name.
_GALLERY_SECTION_ORDER = [
    "../../examples/emd",
    "../../examples/emd_variants",
    "../../examples/memd",
    "../../examples/ewt",
    "../../examples/faemd",
    "../../examples/vmd",
    "../../examples/vncmd",
    "../../examples/gdmd",
    "../../examples/acmd",
    "../../examples/deconvolution",
    "../../examples/image",
    "../../examples/imd",
    "../../examples/jmd",
    "../../examples/lmd",
    "../../examples/osd",
    "../../examples/ssa",
    "../../examples/temp_iter",
    "../../examples/tfa",
    "../../examples/tsa",
    "../../examples/utils",
]

# Gallery HTML/PNG is generated at build time (gitignored under
# docs/source/auto_examples/). Read the Docs downloads a tarball from
# the rolling GitHub Release tag gallery-cache, then hosts the figures;
# do not commit that folder to GitHub.
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py$",
    "ignore_pattern": r"__init__\.py",
    "subsection_order": _GALLERY_SECTION_ORDER,
    "nested_sections": True,
    "download_all_examples": True,
    "reset_modules": ("matplotlib",),
    "abort_on_example_error": False,
    "min_reported_time": 1,
    "matplotlib_animations": False,
}

autodoc_default_options = {
    "members": True,
    "imported-members": True,
    "show-inheritance": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
}

templates_path = []

source_suffix = {".rst": "restructuredtext"}

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"

language = "en"

show_warning_types = True
suppress_warnings = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_css_files = ["theme_overrides.css", "custom.css"]

html_theme_options = {
    "announcement": "",
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
        "text": "PYSDKIT",
        "alt_text": "PySDKit",
        "link": "https://pysdkit.readthedocs.io/",
    },
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wwhenxuan/PySDKit",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/PySDKit/",
            "icon": "https://raw.githubusercontent.com/changewam/PySDKit/refs/heads/main/docs/source/_static/logo-pypi.svg",
            "type": "url",
        },
    ],
    "navbar_align": "content",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "show_version_warning_banner": False,
    "secondary_sidebar_items": {
        "**": ["page-toc", "sourcelink"],
    },
    "show_toc_level": 4,
    "collapse_navigation": True,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "pygments_light_style": "xcode",
    "pygments_dark_style": "monokai",
    "show_prev_next": False,
    "show_nav_level": 1,
    "back_to_top_button": True,
}

remove_from_toctrees = []

html_sidebars = {
    "index": [],
    "API/pysdkit.*": [],
    # Gallery indexes: left "Section Navigation" duplicates the right TOC.
    "auto_examples/index": [],
    "auto_examples/*/index": [],
    "release_notes/index": [],
}
html_show_sourcelink = False

htmlhelp_basename = "pysdkitdoc"
