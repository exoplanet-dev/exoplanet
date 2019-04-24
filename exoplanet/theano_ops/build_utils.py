# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["get_compile_args", "get_cache_version"]

import sys
from ..exoplanet_version import __version__


def get_compile_args(compiler):
    opts = ["-std=c++11", "-O2", "-DNDEBUG"]
    if sys.platform == "darwin":
        opts += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
    return opts


def get_cache_version():
    if "dev" in __version__:
        return ()
    return tuple(map(int, __version__.split(".")))
