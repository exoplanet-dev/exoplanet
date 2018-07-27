# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

from .tri_diag_solve import tri_diag_solve


class TriDiagSolveTest(tf.test.TestCase):

    dtypes = [tf.float32, tf.float64]

    def test_solve(self):
        np.random.seed(42)
        N_in = 5
        N_out = 2
        N = 100
        diag = np.random.uniform(5, 10, (N_in, N))
        upper = np.random.uniform(1, 2.5, (N_in, N-1))
        lower = np.random.uniform(1, 2.5, (N_in, N-1))
        y = np.random.randn(N_in, N, N_out)

        x_np = np.empty_like(y)
        for n in range(N_in):
            A = np.diag(diag[n])
            A += np.diag(upper[n], k=1)
            A += np.diag(lower[n], k=-1)
            x_np[n] = np.linalg.solve(A, y[n])

        with self.test_session() as sess:
            for T in self.dtypes:
                diag_t = tf.constant(diag, dtype=T)
                upper_t = tf.constant(upper, dtype=T)
                lower_t = tf.constant(lower, dtype=T)
                y_t = tf.constant(y, dtype=T)

                x = tri_diag_solve(diag_t, upper_t, lower_t, y_t)

                assert np.allclose(sess.run(x), x_np)

    def test_gradient(self):
        np.random.seed(42)
        N_in = 5
        N_out = 2
        N = 100
        diag = np.random.uniform(5, 10, (N_in, N))
        upper = np.random.uniform(1, 2.5, (N_in, N-1))
        lower = np.random.uniform(1, 2.5, (N_in, N-1))
        y = np.random.randn(N_in, N, N_out)

        with self.test_session() as sess:
            for T, eps in zip(self.dtypes, [1e-3, 1e-5]):
                diag_t = tf.constant(diag, dtype=T)
                upper_t = tf.constant(upper, dtype=T)
                lower_t = tf.constant(lower, dtype=T)
                y_t = tf.constant(y, dtype=T)

                x = tri_diag_solve(diag_t, upper_t, lower_t, y_t)

                params = [diag_t, upper_t, lower_t, y_t]
                vals = sess.run(params)
                shapes = [np.shape(v) for v in vals]
                err = tf.test.compute_gradient_error(
                    params, shapes,
                    x, sess.run(x).shape,
                    x_init_value=vals, delta=eps
                )
                assert np.allclose(err, 0.0, atol=eps)
