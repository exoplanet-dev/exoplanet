#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import subprocess
from itertools import chain

import sphinx_nameko_theme

sys.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup(app):
    app.add_stylesheet("css/exoplanet.css")


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

autodoc_mock_imports = [
    "numpy",
    "scipy",
    "astropy",
    "pymc3",
    "theano",
]

# Convert the tutorials
for fn in chain(glob.glob("_static/notebooks/*.ipynb"),
                glob.glob("_static/notebooks/gallery/*.ipynb")):
    name = os.path.splitext(os.path.split(fn)[1])[0]
    outfn = os.path.join("tutorials", name + ".rst")
    print("Building {0}...".format(name))
    subprocess.check_call(
        "jupyter nbconvert --template tutorials/tutorial_rst --to rst "
        + fn + " --output-dir tutorials", shell=True)
    subprocess.check_call(
        "python fix_internal_links.py " + outfn, shell=True)

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'astropy': ('http://docs.astropy.org/en/stable/', None),
}

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project.
project = "exoplanet"
author = "Dan Foreman-Mackey"
copyright = "2018, 2019, " + author

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "exoplanet"))
from exoplanet_version import __version__  # NOQA
version = __version__
release = __version__

exclude_patterns = ["_build"]
pygments_style = "sphinx"

# Readthedocs.
# on_rtd = os.environ.get("READTHEDOCS", None) == "True"
html_theme_path = [sphinx_nameko_theme.get_html_theme_path()]
html_theme = "nameko"

html_context = dict(
    display_github=True,
    github_user="dfm",
    github_repo="exoplanet",
    github_version="master",
    conf_py_path="/docs/",
)
html_static_path = ["_static"]
html_show_sourcelink = False
