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

import theano.tensor as tt
from exoplanet_core.theano import ops
from theano.ifelse import ifelse

numpy = tt


def as_tensor(x, dtype="float64", **kwargs):
    t = tt.as_tensor_variable(x, **kwargs)
    if dtype is None:
        return t
    return t.astype(dtype)


def and_(m1, m2):
    return tt.and_(m1, m2)


def switch(m, a, b):
    return tt.switch(m, a, b)


def set_subtensor(inds, a, b):
    tt.set_subtensor(a[inds], b)


def searchsorted(a, v, **kwargs):
    return tt.extra_ops.searchsorted(a, v, **kwargs)
