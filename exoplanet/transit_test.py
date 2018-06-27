# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

from . import transit

from batman import _quadratic_ld, _nonlinear_ld

quadratic_ld = [
    lambda T: transit.QuadraticLimbDarkening(tf.constant(0.8, dtype=T),
                                             tf.constant(0.0, dtype=T)),
    lambda T: transit.QuadraticLimbDarkening(tf.constant(0.0, dtype=T),
                                             tf.constant(0.8, dtype=T)),
    lambda T: transit.QuadraticLimbDarkening(tf.constant(0.1, dtype=T),
                                             tf.constant(0.9, dtype=T)),
    lambda T: transit.QuadraticLimbDarkening(tf.constant(0.5, dtype=T),
                                             tf.constant(0.2, dtype=T)),
    lambda T: transit.QuadraticLimbDarkening(tf.constant(0.0, dtype=T),
                                             tf.constant(0.0, dtype=T)),
]
nonlinear_ld = [
    lambda T: transit.NonLinearLimbDarkening(tf.constant(0.5, dtype=T),
                                             tf.constant(0.4, dtype=T),
                                             tf.constant(0.3, dtype=T),
                                             tf.constant(0.2, dtype=T)),
]


class LimbDarkeningTest(tf.test.TestCase):

    dtypes = [tf.float32, tf.float64]
    limb_darkening_profiles = quadratic_ld + nonlinear_ld

    def test_integral(self):
        with self.test_session() as sess:
            for T in self.dtypes:
                x = tf.cast(tf.linspace(0.0, 1.0, 100000), T)
                for ld_factory in self.limb_darkening_profiles:
                    ld = ld_factory(T)
                    y = x * ld.evaluate(x)
                    a, b = sess.run([x, y])
                    I = 2.0*np.pi*np.sum(0.5*(b[1:] + b[:-1])*(a[1:]-a[:-1]))
                    assert np.allclose(I, 1.0)


class BatmanTest(tf.test.TestCase):

    rors = [0.01, 0.1, 0.5]

    def test_batman_quadratic(self):
        T = tf.float64
        with self.test_session() as sess:
            for ld_factory in quadratic_ld:
                ld = ld_factory(T)
                for ror in self.rors:
                    N = 1000
                    z = tf.constant(np.linspace(0, 1+2*ror, N), dtype=T)
                    r = tf.constant(ror, dtype=T)
                    delta = transit.transit_depth(ld, z, r, n_integrate=1000)
                    args = [z.eval(), ror, ld.c1.eval(), ld.c2.eval(), 1]
                    lc0 = _quadratic_ld._quadratic_ld(*args)
                    assert np.allclose(1.0 - sess.run(delta), lc0)

    def test_batman_nonlinear(self):
        T = tf.float64
        with self.test_session() as sess:
            for ld_factory in nonlinear_ld:
                ld = ld_factory(T)
                for ror in self.rors:
                    N = 1000
                    z = tf.constant(np.linspace(0, 1+2*ror, N), dtype=T)
                    r = tf.constant(ror, dtype=T)
                    delta = transit.transit_depth(ld, z, r, n_integrate=1000)
                    args = [z.eval(), ror]
                    args += list(sess.run([ld.c1, ld.c2, ld.c3, ld.c4]))
                    args += [1e-3, 1]
                    lc0 = _nonlinear_ld._nonlinear_ld(*args)
                    assert np.allclose(1.0 - sess.run(delta), lc0)


class ConvergenceTest(tf.test.TestCase):

    dtypes = [tf.float32, tf.float64]
    limb_darkening_profiles = quadratic_ld + nonlinear_ld
    rors = [0.01, 0.1, 0.5]

    def test_convergence(self):
        with self.test_session() as sess:
            for T in self.dtypes:
                for ld_factory in self.limb_darkening_profiles:
                    ld = ld_factory(T)
                    for ror in self.rors:
                        N = 100
                        z = tf.constant(np.linspace(0, 1+2*ror, N), dtype=T)
                        r = tf.constant(ror, dtype=T)
                        delta = transit.transit_depth(ld, z, r,
                                                      n_integrate=1000)
                        delta_exact = transit.transit_depth(
                            ld, z, r, n_integrate=10000)
                        assert not np.any(np.isnan(delta_exact.eval()))
                        assert np.allclose(*sess.run([delta_exact, delta]),
                                           rtol=1.0, atol=1e-6)
