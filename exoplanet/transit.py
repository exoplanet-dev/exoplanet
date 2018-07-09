# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = [
    "QuadraticLimbDarkening", "NonLinearLimbDarkening",
    "transit_depth"
]

import numpy as np
import tensorflow as tf

from .tf_utils import load_op_library


ops = load_op_library("transit_op")


class ConstantLimbDarkening(object):

    def __init__(self):
        self.inv_I0 = 1.0 / np.pi
        self.params = []

    def evaluate(self, radius):
        return self.inv_I0 * tf.ones_like(radius)


class QuadraticLimbDarkening(object):

    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2
        self.I0 = np.pi * (1.0 - c1 / 3.0 - c2 / 6.0)
        self.params = [self.c1, self.c2]

    def evaluate(self, radius):
        mu = 1.0 - tf.sqrt(1.0 - radius**2)
        return (1.0 - self.c1 * mu - self.c2 * mu**2) / self.I0


class NonLinearLimbDarkening(object):

    def __init__(self, c1, c2, c3, c4):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.I0 = np.pi * (1.0 - c1/5.0 - c2/3.0 - 3.0*c3/7.0 - c4/2.0)
        self.params = [self.c1, self.c2, self.c3, self.c4]

    def evaluate(self, radius):
        mu2 = 1.0 - radius**2
        mu = tf.sqrt(mu2)
        sqrtmu = tf.sqrt(mu)
        return (1.0
                - self.c1 * (1.0 - sqrtmu)
                - self.c2 * (1.0 - mu)
                - self.c3 * (1.0 - mu*sqrtmu)
                - self.c4 * (1.0 - mu2)) / self.I0


def radius_to_index(N_grid, radius):
    return tf.cast(N_grid - 1, radius.dtype) \
        * (1.0 - tf.sqrt(1.0 - tf.clip_by_value(radius, 0.0, 1.0)))


def transit_depth(limb_darkening, z, r, direction=None, n_integrate=1000):
    """Compute the depth of a set of transit configurations

    Args:
        limb_darkening: A limb darkening model with an ``evaluate`` method.
        z: The projected sky distance between the star and the transiting body
        r: The radius of the transiting body in stellar units (must have the
            same shape as z)
        direction (Optional): If ``direction > 0`` this is a transit, but if
            ``direction <= 0`` this will always return zero.
        n_integrate (Optional): The number of annuli to use to numerically
            integrate the limb darkening profile

    Return:
        delta: The relative transit depth evaluated at the coordinates given
            by ``z``

    """
    if direction is None:
        direction = tf.ones_like(z)
    radius = 1.0 - tf.square(1.0 - tf.cast(tf.linspace(0.0, 1.0, n_integrate),
                                           z.dtype))
    n_min = tf.cast(tf.floor(radius_to_index(n_integrate, z - r)), tf.int32)
    n_max = tf.cast(tf.ceil(radius_to_index(n_integrate, z + r)), tf.int32)
    I = limb_darkening.evaluate(radius)
    return ops.transit_depth(radius, I, n_min, n_max, z, r, direction)


@tf.RegisterGradient("TransitDepth")
def _transit_depth_grad(op, *grads):
    results = ops.transit_depth_rev(*(list(op.inputs) + [grads[0]]))
    return (None, results[0], None, None, results[1], results[2], None)


@tf.RegisterGradient("OccultedArea")
def _occulted_area_grad(op, *grads):
    results = ops.occulted_area_rev(*(list(op.inputs) + [grads[0]]))
    return (None, results[0], results[1])
