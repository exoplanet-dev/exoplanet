# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["UnitVectorTransform", "unit_vector_transform", "UnitVector"]

import numpy as np

import pymc3 as pm
import pymc3.distributions.transforms as tr

import theano.tensor as tt


class UnitVectorTransform(tr.Transform):
    """A unit vector transformation for PyMC3

    The variable is normalized so that the sum of squares over the last axis
    is unity.

    """
    name = "unitvector"

    def backward(self, y):
        norm = tt.sqrt(tt.sum(tt.square(y), axis=-1, keepdims=True))
        return y / norm

    def forward(self, x):
        return tt.as_tensor_variable(x)

    def forward_val(self, x, point=None):
        return np.copy(x)

    def jacobian_det(self, y):
        return -0.5*tt.sum(tt.square(y), axis=-1)


unit_vector_transform = UnitVectorTransform()


class UnitVector(pm.Flat):

    def __init__(self, *args, **kwargs):
        kwargs["transform"] = unit_vector_transform
        super(UnitVector, self).__init__(*args, **kwargs)
