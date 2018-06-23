# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["transit_depth"]

import os
import sysconfig
import numpy as np
import tensorflow as tf


# Load the ops library
suffix = sysconfig.get_config_var("EXT_SUFFIX")
dirname = os.path.dirname(os.path.abspath(__file__))
libfile = os.path.join(dirname, "transit_op")
if suffix is not None:
    libfile += suffix
else:
    libfile += ".so"
ops = tf.load_op_library(libfile)


class QuadraticLimbDarkening(object):

    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2
        self.I0 = np.pi * (1.0 - c1 / 3.0 - c2 / 6.0)

    def evaluate(self, radius):
        mu = 1.0 - tf.sqrt(1.0 - radius**2)
        return (1.0 - self.c1 * mu - self.c2 * mu**2) / self.I0


def radius_to_index(N_grid, radius):
    return tf.cast(N_grid - 1, radius.dtype) \
        * (1.0 - tf.sqrt(1.0 - tf.clip_by_value(radius, 0.0, 1.0)))


def transit_depth(limb_darkening, z, r, n_integrate=1000):
    radius = 1.0 - tf.square(1.0 - tf.cast(tf.linspace(0.0, 1.0, n_integrate),
                                           z.dtype))
    n_min = tf.cast(tf.floor(radius_to_index(n_integrate, z - r)), tf.int32)
    n_max = tf.cast(tf.ceil(radius_to_index(n_integrate, z + r)), tf.int32)

    return ops.transit_depth(radius, limb_darkening.evaluate(radius),
                             n_min, n_max, z, r)


@tf.RegisterGradient("TransitDepth")
def _transit_depth_grad(op, *grads):
    results = ops.transit_depth_rev(*(list(op.inputs) + [grads[0]]))
    return (None, results[0], None, None, results[1], results[2])
