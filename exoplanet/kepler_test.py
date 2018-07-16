# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

from . import kepler


class KeplerTest(tf.test.TestCase):

    dtypes = [tf.float64]

    def test_edge(self):
        E = np.array([0.0, 2*np.pi, -226.2])
        e = np.ones_like(E)
        M = E - e * np.sin(E)
        with self.test_session() as sess:
            for T in self.dtypes:
                E0 = kepler.kepler(tf.constant(M, dtype=T),
                                   tf.constant(e, dtype=T))
                assert np.all(np.isfinite(sess.run(E0)))

    def test_solver(self):
        e = np.linspace(0, 1, 500)
        E = np.linspace(-300, 300, 1001)
        e = e[None, :] + np.zeros((len(E), len(e)))
        E = E[:, None] + np.zeros_like(e)
        M = E - e * np.sin(E)
        with self.test_session() as sess:
            for T in self.dtypes:
                E0 = kepler.kepler(tf.constant(M, dtype=T),
                                   tf.constant(e, dtype=T), maxiter=50)
                assert np.allclose(E, sess.run(E0))
