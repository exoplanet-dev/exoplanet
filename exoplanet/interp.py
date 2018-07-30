# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["interp1d", "CubicInterpolator"]

import tensorflow as tf

from .tf_utils import load_op_library
from .tri_diag_solve import tri_diag_solve


cubic_op = load_op_library("cubic_op")
interp_op = load_op_library("interp_op")


def interp1d(t, x, y):
    return interp_op.interp(t, x, y)[0]


@tf.RegisterGradient("Interp")
def _interp_grad(op, *grads):
    t, x, y = op.inputs
    v, inds = op.outputs
    bv = grads[0]
    bt, by = interp_op.interp_rev(t, x, y, inds, bv)
    return [bt, None, by]


@tf.RegisterGradient("CubicGather")
def _cubic_gather_rev(op, *grads):
    x = op.inputs[1]
    inds = op.outputs[-1]
    args = [x, inds] + list(grads)
    results = cubic_op.cubic_gather_rev(*args)
    return [tf.zeros_like(op.inputs[0])] + list(results)


class CubicInterpolator(object):

    def __init__(self, x, y, fpa=None, fpb=None,
                 dtype=tf.float32, name=None):
        with tf.name_scope(name, "CubicInterpolator"):
            x = tf.cast(x, dtype)
            y = tf.cast(y, dtype)

            # Compute the deltas
            size = tf.shape(x)[-1]
            axis = tf.rank(x) - 1
            dx = tf.gather(x, tf.range(1, size), axis=axis) \
                - tf.gather(x, tf.range(size-1), axis=axis)
            dy = tf.gather(y, tf.range(1, size), axis=axis) \
                - tf.gather(y, tf.range(size-1), axis=axis)

            # Compute the slices
            upper_inds = tf.range(1, size-1)
            lower_inds = tf.range(size-2)
            s_up = lambda a: tf.gather(a, upper_inds, axis=axis)  # NOQA
            s_lo = lambda a: tf.gather(a, lower_inds, axis=axis)  # NOQA
            dx_up = s_up(dx)
            dx_lo = s_lo(dx)
            dy_up = s_up(dy)
            dy_lo = s_lo(dy)

            first = lambda a: tf.gather(a, tf.zeros(1, dtype=tf.int64),  # NOQA
                                        axis=axis)
            last = lambda a: tf.gather(a, [size-2], axis=axis)  # NOQA

            fpa_ = fpa if fpa is not None else tf.constant(0, dtype)
            fpb_ = fpb if fpb is not None else tf.constant(0, dtype)

            diag = 2*tf.concat((first(dx), dx_up+dx_lo, last(dx)), axis)
            upper = dx
            lower = dx
            Y = 3*tf.concat((first(dy)/first(dx) - fpa_,
                             dy_up/dx_up - dy_lo/dx_lo,
                             fpb_ - last(dy)/last(dx)), axis)

            # Solve the tri-diagonal system
            c = tri_diag_solve(diag, upper, lower, Y)
            c_up = tf.gather(c, tf.range(1, size), axis=axis)
            c_lo = tf.gather(c, tf.range(size-1), axis=axis)
            b = dy / dx - dx * (c_up + 2*c_lo) / 3
            d = (c_up - c_lo) / (3*dx)

            self.x = x
            self.y = y
            self.b = b
            self.c = c_lo
            self.d = d

    def evaluate(self, t, name=None):
        with tf.name_scope(name, "evaluate"):
            res = cubic_op.cubic_gather(t, self.x, self.y, self.b, self.c,
                                        self.d)
            tau = t - res.xk
            return res.ak + res.bk * tau + res.ck * tau**2 + res.dk * tau**3

            # inds = cubic_op.search_sorted(self.x, t)

            # if self._endpoints == "natural":
            #     inds = tf.clip_by_value(
            #         inds-1,
            #         tf.constant(0, dtype=tf.int64),
            #         tf.cast(tf.shape(self.x)[-1], tf.int64) - 2)

            # inds = tf.stack(tf.meshgrid(
            #     *[tf.range(s, dtype=tf.int64) for s in t.shape],
            #     indexing="ij")[:-1] + [inds], axis=-1)

            # tau = t - tf.gather_nd(self.x_ext, inds)
            # mod = tf.gather_nd(self.y_ext, inds)
            # mod += tau * tf.gather_nd(self.b, inds)
            # mod += tau**2 * tf.gather_nd(self.c, inds)
            # mod += tau**3 * tf.gather_nd(self.d, inds)

            # return mod
