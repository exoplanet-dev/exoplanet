# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["interp1d"]

import tensorflow as tf

from .tf_utils import load_op_library


ops = load_op_library("search_sorted_op")


def interp1d(t, x, y):
    inds = ops.search_sorted(x, t)
    x_ext = tf.concat((x[:1], x, x[-1:]), axis=0)
    y_ext = tf.concat((y[:1], y, y[-1:]), axis=0)
    dx = x_ext[1:] - x_ext[:-1]
    dy = y_ext[1:] - y_ext[:-1]
    dx = tf.where(tf.greater(tf.abs(dx), tf.zeros_like(dx)),
                  dx, tf.ones_like(dx))

    x0 = tf.gather(x_ext, inds)
    y0 = tf.gather(y_ext, inds)
    slope = tf.gather(dy / dx, inds)

    return slope * (t - x0) + y0
