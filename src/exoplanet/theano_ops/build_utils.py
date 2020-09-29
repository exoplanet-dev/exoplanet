# -*- coding: utf-8 -*-

__all__ = ["get_compile_args", "get_cache_version", "get_header_dirs"]

import sys

import pkg_resources

from ..exoplanet_version import __version__


def get_compile_args(compiler):
    opts = ["-std=c++11", "-DNDEBUG"]
    if sys.platform == "darwin":
        opts += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]
    if sys.platform.startswith("win"):
        opts += ["-D_USE_MATH_DEFINES", "-D_hypot=hypot"]
    else:
        opts += ["-O2"]
    return opts


def get_cache_version():
    try:
        return tuple(map(int, __version__.split(".")))
    except ValueError:
        return ()


def get_header_dirs(eigen=True):
    dirs = [
        pkg_resources.resource_filename(__name__, "lib/include"),
        pkg_resources.resource_filename(__name__, "lib/vendor/eigen-3.3.7"),
    ]
    return dirs
