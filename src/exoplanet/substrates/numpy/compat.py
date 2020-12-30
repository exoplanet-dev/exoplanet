# -*- coding: utf-8 -*-


__all__ = [
    "and_",
    "as_tensor",
    "ifelse",
    "numpy",
    "ops",
    "searchsorted",
    "set_subtensor",
    "switch",
]

import numpy
from exoplanet_core.numpy import ops

isscalar = numpy.isscalar


def as_tensor(x, dtype="float64", **kwargs):
    return numpy.ascontiguousarray(x, dtype=dtype)


def and_(m1, m2):
    return m1 & m2


def switch(m, a, b):
    return a * m + b * (~m)


def ifelse(flag, a, b):
    if flag:
        return a
    return b


def set_subtensor(inds, a, b):
    a[inds] = b
    return a


def searchsorted(a, v, **kwargs):
    return numpy.searchsorted(a, v, **kwargs)
