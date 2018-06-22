#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import tensorflow as tf
session = tf.InteractiveSession()

from transit import custom_ops

T = tf.float64

c1 = tf.constant(0.5, dtype=T)
c2 = tf.constant(0.1, dtype=T)
I0 = np.pi * (1.0 - (2.0 * c1 + c2) / 6.0)

x = np.linspace(0.0, 1, 1400)
mu = 1.0 - np.sqrt(1.0 - x**2)
I = (1.0 - c1 * mu - c2 * mu**2) / I0

N = 10*100000
# N = 100
z = tf.constant(np.linspace(0, 1.1, N), dtype=T)
r = 0.01 + tf.zeros(N, dtype=T)

delta = custom_ops.transit(I, z, r)

session.run(delta)
get_ipython().magic('timeit session.run(delta)')
