# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date
import os
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PYSDKIT"

copyright = f"2025-{date.today().year}, the pysdkit team"
author = "the pysdkit team"

release = "0.4.15"
version = "0.4.15"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#sys.path.append(str(Path('exts').resolve()))
sys.path.insert(0, str(Path('..', '..').resolve()))

import pysdkit

extensions = [
    "sphinx_copybutton",
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    'sphinx.ext.napoleon',

    "sphinx_gitstamp",
]

apidoc_modules = [
    {
        'path': '../../pysdkit',
        'destination': 'source/API',
        'exclude_patterns': ['**/test*'],
        'max_depth': 4,
        'follow_links': False,
        'separate_modules': False,
        'include_private': True,
        'no_headings': False,
        'module_first': False,
        'implicit_namespaces': True,
        'automodule_options': {
            'members', 'show-inheritance', 'undoc-members'
        },
    },
]
templates_path = ["_templates"]

source_suffix = {".rst": "restructuredtext"}
#__________________________________________________________________________
html_logo = "_static/logo.png"

language = "en"

show_warning_types = True
suppress_warnings = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_css_files = ["theme_overrides.css","custom.css"]

html_theme_options = {
    "announcement": "",#You can specify an arbitrary URL that will be used as the HTML source for your announcement. 
    # Navigation bar
    "logo": {
        "text": "pysdkit",
        "link": "https://pysdkit.readthedocs.io/",
    },
    "header_links_before_dropdown":6 ,
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
    "navbar_start": ["navbar-logo","version-switcher"],
    "navbar_center": ["navbar-nav"],
    #"navbar_end": ["navbar-icon-links"], 
    #"navbar_persistent": ["search-button"],
    #______________________________________________________________________________________
    "switcher": {
        "json_url": ("https://raw.githubusercontent.com/changewam/PySDKit/refs/heads/main/docs/source/_static/version_switcher.json"),#the persistent location of the JSON file
        "version_match": "dev" if "dev" in version else version,
    },
    "show_version_warning_banner": True,
    # Secondary_sidebar_items
    "secondary_sidebar_items": {
    "**": ["page-toc", "sourcelink"],
    "index": [],
    },
    # Footer
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    # Color
   "pygments_light_style": "xcode",
   "pygments_dark_style": "monokai",
    # Other
    "show_prev_next": False,
    "show_nav_level": 1,
    "back_to_top_button": True,
    
    #"use_edit_page_button": True,


}

remove_from_toctrees = []

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "index": [] , # Hide sidebar in home page
    "API/pysdkit.*": []
}
html_show_sourcelink = False

# Output file base name for HTML help builder.
htmlhelp_basename = "scikitimagedoc"