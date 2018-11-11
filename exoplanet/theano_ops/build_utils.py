# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["get_compile_args"]

import sys


def get_compile_args(compiler):
    opts = ["-std=c++11", "-O2", "-DNDEBUG"]
    if sys.platform == "darwin":
        opts += ["-stdlib=libc++", "-mmacosx-version-min=10.7"]

    return opts
