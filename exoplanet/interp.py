# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["interp1d"]

import tensorflow as tf

from .tf_utils import load_op_library


# ops = load_op_library("search_sorted_op")
ops = load_op_library("interp_op")


def interp1d(t, period, x, y):
    return ops.interp(t, period, x, y)[0]


@tf.RegisterGradient("Interp")
def _interp_grad(op, *grads):
    t, period, x, y = op.inputs
    v, a, inds = op.outputs
    bv = grads[0]
    bt, dp, by = ops.interp_rev(t, period, x, y, a, inds, bv)
    return [bt, dp, None, by]
