#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import tensorflow as tf
session = tf.InteractiveSession()

np.random.seed(42)

T = tf.float64

N = 56773
K = 2
p = tf.constant(np.random.uniform(5, 10, K), dtype=T)
x = tf.constant(np.random.uniform(0, 5000, N*K).reshape((K, N)), dtype=T)
phase = tf.mod(x, p[:, None])
grad = tf.gradients(phase, [x, p])

session.run(phase)
session.run(grad)
get_ipython().magic('timeit session.run(phase)')
get_ipython().magic('timeit session.run(grad)')
