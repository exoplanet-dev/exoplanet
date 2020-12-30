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

import theano.tensor as tt
from exoplanet_core.theano import ops
from theano.ifelse import ifelse

numpy = tt
eq = tt.eq
abs_ = tt.abs_
searchsorted = tt.extra_ops.searchsorted


def as_tensor(x, dtype="float64", **kwargs):
    t = tt.as_tensor_variable(x, **kwargs)
    if dtype is None:
        return t
    return t.astype(dtype)


def and_(m1, m2):
    return tt.and_(m1, m2)


def isscalar(x):
    return as_tensor(x, dtype=None).ndim == 0


def switch(m, a, b):
    return tt.switch(m, a, b)


def set_subtensor(inds, a, b):
    return tt.set_subtensor(a[inds], b)
