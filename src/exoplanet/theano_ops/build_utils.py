# -*- coding: utf-8 -*-

__all__ = ["get_compile_args", "get_cache_version", "get_header_dirs"]

import sys

import pkg_resources

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


def get_header_dirs(eigen=True):
    dirs = [pkg_resources.resource_filename(__name__, "lib/include")]
    if eigen:
        dirs += [
            pkg_resources.resource_filename(__name__, "lib/vendor/eigen_3.3.3")
        ]
    return dirs
