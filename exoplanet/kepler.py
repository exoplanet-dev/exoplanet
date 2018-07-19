# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["kepler"]

import numpy as np
import tensorflow as tf

from .tf_utils import load_op_library


ops = load_op_library("kepler_op")


def kepler(M, e, **kwargs):
    """Solve Kepler's equation

    Args:
        M: A Tensor of mean anomaly values.
        e: The eccentricity.
        maxiter (Optional): The maximum number of iterations to run.
        tol (Optional): The convergence tolerance.

    Returns:
        A Tensor with the eccentric anomaly evaluated for each entry in ``M``.

    """
    return ops.kepler(M, e, **kwargs)


@tf.RegisterGradient("Kepler")
def _kepler_grad(op, *grads):
    M, e = op.inputs
    E = op.outputs[0]
    bE = grads[0]
    bM = bE / (1.0 - e * tf.cos(E))
    be = tf.sin(E) * bM
    return [bM, be]


def sky_position(period, t0, e, omega, incl, t, **kwargs):
    # Shorthands
    n = 2.0 * np.pi / period
    cos_omega = tf.cos(omega)
    sin_omega = tf.sin(omega)

    # Find the reference time that puts the center of transit at t0
    E0 = 2.0 * tf.atan2(tf.sqrt(1.0 - e) * cos_omega,
                        tf.sqrt(1.0 + e) * (1.0 + sin_omega))
    tref = t0 - (E0 - e * tf.sin(E0)) / n

    # Solve Kepler's equation for the true anomaly
    M = (t - tref) * n
    E = kepler(M, e + tf.zeros_like(M), **kwargs)
    f = 2.0 * tf.atan2(tf.sqrt(1.0 + e) * tf.tan(0.5*E),
                       tf.sqrt(1.0 - e) + tf.zeros_like(E))

    # Rotate into sky coordinates
    cf = tf.cos(f)
    sf = tf.sin(f)
    swpf = sin_omega * cf + cos_omega * sf
    cwpf = cos_omega * cf - sin_omega * sf

    # Star / planet distance
    r = (1.0 - tf.square(e)) / (1 + e * cf)

    return tf.stack([
        -r * cwpf,
        -r * swpf * tf.cos(incl),
        r * swpf * tf.sin(incl)
    ])
