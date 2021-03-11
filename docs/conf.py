#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

import nbsphinx
import sphinx_typlog_theme
from nbsphinx import markdown2rst as original_markdown2rst
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("exoplanet").version
except DistributionNotFound:
    __version__ = "unknown version"


def setup(app):
    app.add_css_file("css/exoplanet.css?v=2020-01-15")


# nbsphinx hacks
nbsphinx.RST_TEMPLATE = nbsphinx.RST_TEMPLATE.replace(
    "{%- if width %}", "{%- if 0 %}"
).replace("{%- if height %}", "{%- if 0 %}")


def subber(m):
    return m.group(0).replace("``", "`")


prog = re.compile(r":(.+):``(.+)``")


def markdown2rst(text):
    return prog.sub(subber, original_markdown2rst(text))


nbsphinx.markdown2rst = markdown2rst


# General stuff
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "nbsphinx",
]

autodoc_mock_imports = [
    "numpy",
    "scipy",
    "astropy",
    "pymc3",
    "theano",
    "aesara_theano_fallback",
    "tqdm",
    "rebound_pymc3",
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
pygments_style = "sphinx"

# HTML theme
html_favicon = "_static/logo.png"
html_theme = "exoplanet"
html_theme_path = ["_themes", sphinx_typlog_theme.get_path()]
html_theme_options = {"logo": "logo.png", "color": "#F55826"}
html_sidebars = {
    "**": ["logo.html", "globaltoc.html", "relations.html", "searchbox.html"]
}
html_static_path = ["_static"]

# Get the git branch name
branch_name = os.environ.get("SOURCE_BRANCH_NAME", "master")
html_context = dict(
    this_branch=branch_name,
    this_version="latest" if branch_name == "master" else branch_name,
)
