# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["tri_diag_solve"]

import tensorflow as tf

from .tf_utils import load_op_library


ops = load_op_library("cubic_op")


def tri_diag_solve(diag, upper, lower, y, **kwargs):
    return ops.tri_diag_solve(diag, upper, lower, y, **kwargs)


@tf.RegisterGradient("TriDiagSolve")
def _tri_diag_solve_grad(op, *grads):
    diag, upper, lower, y = op.inputs
    x = op.outputs[0]
    bx = grads[0]
    by = ops.tri_diag_solve(diag, lower, upper, bx)
    axes = tf.range(tf.rank(diag), tf.rank(y))
    bdiag = -tf.reduce_sum(x * by, axis=axes)

    n_inner = tf.shape(diag)[-1]
    axis = tf.rank(diag) - 1
    bupper = -tf.reduce_sum(
        tf.gather(x, tf.range(1, n_inner), axis=axis) *
        tf.gather(by, tf.range(n_inner-1), axis=axis),
        axis=axes)
    blower = -tf.reduce_sum(
        tf.gather(x, tf.range(n_inner-1), axis=axis) *
        tf.gather(by, tf.range(1, n_inner), axis=axis),
        axis=axes)

    return [bdiag, bupper, blower, by]
