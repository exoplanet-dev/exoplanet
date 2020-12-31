# -*- coding: utf-8 -*-

__all__ = [
    "abs_",
    "and_",
    "as_tensor",
    "eq",
    "ifelse",
    "isscalar",
    "numpy",
    "ops",
    "searchsorted",
    "set_subtensor",
    "switch",
]

import numpy
from exoplanet_core.numpy import ops

isscalar = numpy.isscalar
abs_ = numpy.abs
searchsorted = numpy.searchsorted


def and_(m1, m2):
    return m1 & m2


def as_tensor(x, dtype="float64", **kwargs):
    return numpy.ascontiguousarray(x, dtype=dtype)


def eq(a, b):
    return a == b


def switch(m, a, b):
    return a * m + b * (~m)


def ifelse(flag, a, b):
    if flag:
        return a
    return b


def set_subtensor(inds, a, b):
    a[inds] = b
    return a
