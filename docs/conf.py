#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("exoplanet").version
except DistributionNotFound:
    __version__ = "unknown version"


# General stuff
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_nb",
]

autodoc_mock_imports = [
    "numpy",
    "scipy",
    "astropy",
    "pymc3",
    "theano",
    "aesara_theano_fallback",
    "tqdm",
]

# RTDs-action
if "GITHUB_TOKEN" in os.environ:
    extensions.append("rtds_action")

    rtds_action_github_repo = "exoplanet-dev/exoplanet"
    rtds_action_path = "tutorials"
    rtds_action_artifact_prefix = "notebooks-for-"
    rtds_action_github_token = os.environ["GITHUB_TOKEN"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "astropy": ("http://docs.astropy.org/en/stable/", None),
}

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project.
project = "exoplanet"
author = "Dan Foreman-Mackey"
copyright = "2018, 2019, 2020, 2021, " + author

version = __version__
release = __version__

exclude_patterns = ["_build"]

# HTML theme
html_theme = "sphinx_book_theme"
html_title = "exoplanet"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/exoplanet-dev/exoplanet",
    "repository_branch": "main",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}
# jupyter_execute_notebooks = "off"
execution_timeout = -1
