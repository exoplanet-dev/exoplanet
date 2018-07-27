# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

from batman import _rsky

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

    def test_sky_coords(self):
        t = np.linspace(-100, 100, 5000)

        t0, period, e, omega, incl = (x.flatten() for x in np.meshgrid(
            np.linspace(-5.0, 5.0, 4),
            np.exp(np.linspace(np.log(5.0), np.log(50.0), 3)),
            np.linspace(0.0, 0.9, 5),
            np.linspace(-np.pi, np.pi, 3),
            np.arccos(np.linspace(0, 1, 5)[:-1]),
        ))
        r_batman = np.empty((len(t0), len(t)))

        for i in range(len(t0)):
            r_batman[i] = _rsky._rsky(t, t0[i], period[i], 1.0, incl[i], e[i],
                                      omega[i], 1, 1)
        m = r_batman < 100.0

        with self.test_session() as sess:
            for T in self.dtypes:
                coords = kepler.sky_position(
                    tf.constant(period, dtype=T)[:, None],
                    tf.constant(t0, dtype=T)[:, None],
                    tf.constant(e, dtype=T)[:, None],
                    tf.constant(omega, dtype=T)[:, None],
                    tf.constant(incl, dtype=T)[:, None],
                    tf.constant(t, dtype=T)[None, :], tol=1e-7)
                r = tf.sqrt(tf.reduce_sum(tf.square(coords[:2]), axis=0))
                assert np.allclose(r_batman[m], sess.run(r)[m], atol=1e-6)
