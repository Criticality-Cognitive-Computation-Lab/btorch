"""Sphinx configuration for Btorch documentation."""

import os
import sys


# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "Btorch"
copyright = "2026, Btorch contributors"
author = "Btorch contributors"
version = "0.1.0"
release = "0.1.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False

# Napoleon settings (Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# HTML output
html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {
    "description": "Brain-inspired Torch library for neuromorphic research",
    "github_user": "Criticality-Cognitive-Computation-Lab",
    "github_repo": "btorch",
}

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Master document
master_doc = "index"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
