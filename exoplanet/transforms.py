# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "UnitVectorTransform", "unit_vector",
    "AngleTransform", "angle",
]

import numpy as np

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


unit_vector = UnitVectorTransform()


def _expand_first_dim(x):
    return tt.reshape(x, tt.concatenate([[1], x.shape]), ndim=x.ndim+1)


class AngleTransform(tr.Transform):
    """An angle transformation for PyMC3

    The variable is augmented to sample an isotropic 2D normal and the angle
    is given by the arctan of the ratio of the two coordinates. This will have
    a uniform distribution between -pi and pi.

    """

    name = "angle"

    def backward(self, y):
        yp = tt.swapaxes(y, 0, -1)
        return tt.arctan2(yp[1], yp[0])

    def forward(self, x):
        return tt.concatenate((
            tt.shape_padright(tt.sin(x)),
            tt.shape_padright(tt.cos(x))
        ), axis=-1)

    def forward_val(self, x, point=None):
        return np.swapaxes([np.sin(x), np.cos(x)], 0, -1)

    def jacobian_det(self, y):
        return -0.5*tt.sum(tt.square(y), axis=-1)


angle = AngleTransform()
