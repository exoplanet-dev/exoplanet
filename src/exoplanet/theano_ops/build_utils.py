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
        opts += [
            "-D_USE_MATH_DEFINES",
            "-D_hypot=hypot",
            "-ffixed-xmm16",
            "-ffixed-xmm17",
            "-ffixed-xmm18",
            "-ffixed-xmm19",
            "-ffixed-xmm20",
            "-ffixed-xmm21",
            "-ffixed-xmm22",
            "-ffixed-xmm23",
            "-ffixed-xmm24",
            "-ffixed-xmm25",
            "-ffixed-xmm26",
            "-ffixed-xmm27",
            "-ffixed-xmm28",
            "-ffixed-xmm29",
            "-ffixed-xmm30",
            "-ffixed-xmm31",
        ]
    else:
        opts += ["-O2"]
    return opts


def get_cache_version():
    if "dev" in __version__:
        return ()
    return tuple(map(int, __version__.split(".")))


def get_header_dirs(eigen=True):
    dirs = [pkg_resources.resource_filename(__name__, "lib/include")]
    if eigen:
        dirs += [
            pkg_resources.resource_filename(__name__, "lib/vendor/eigen-3.3.7")
        ]
    return dirs
