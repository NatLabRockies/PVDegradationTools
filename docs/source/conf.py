# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pvdeg"
copyright = "2026, NLR"
author = "Alliance for Energy Innovation LLC"

import pvdeg

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

########################################################################################
### INSTALL pydoc with conda NOT PIP and run in same conda environment when building ###
########################################################################################

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    # 'sphinx_gallery.gen_gallery',
    "sphinx_gallery.load_style",  # thumbnail gallery for .ipynb
    "nbsphinx",  # convert .ipynb to html, install pandoc using CONDA not pip
    "sphinx_toggleprompt",
]

autodoc_default_options = {
    "member-order": "bysource",
    "show-inheritance": True,
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

napoleon_use_rtype = False  # group rtype on same line together with return

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The short X.Y version.
version = "%s" % (pvdeg.__version__)
# The full version, including alpha/beta/rc tags.
release = version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# whatsnew/releases/*.rst files are only meant to be pulled in via `.. include::`
# in whatsnew/index.rst, not built as standalone documents; excluding them here
# avoids duplicate label registration and stray "not in any toctree" warnings.
exclude_patterns = [
    "**.ipynb_checkpoints",
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "whatsnew/releases/*.rst",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# List of external link aliases.  Allows use of :pull:`123` to autolink that PR
extlinks = {
    "issue": ("https://github.com/NREL/PVDegradationTools/issues/%s", "issue %s"),
    "pull": ("https://github.com/NREL/PVDegradationTools/pull/%s", "pull %s"),
    "ghuser": ("https://github.com/%s", "ghuser %s"),
}

## Generate autodoc stubs with summaries from code
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# https://pydata-sphinx-theme.rtfd.io/en/latest/user_guide/configuring.html

html_theme_options = {
    "navigation_depth": 4,
    "github_url": "https://github.com/NREL/PVDegradationTools",
    "show_toc_level": 1,
}

# Image (filename) to place at the top of the sidebar.
html_logo = (
    "./_static/logo-vectors/PVdeg-Logo-Horiz-Color.svg"
)

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "./_static/pvdeg.ico"

html_static_path = ["_static"]

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# Output file base name for HTML help builder.
htmlhelp_basename = "pvdeg_pythondoc"

ipython_warning_is_error = False

# suppress "WARNING: Footnote [1] is not referenced." messages
# https://github.com/pvlib/pvlib-python/issues/837
suppress_warnings = ["ref.footnote"]


# supress warnings in gallery output
# https://sphinx-gallery.github.io/stable/configuration.html
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure."
    )
