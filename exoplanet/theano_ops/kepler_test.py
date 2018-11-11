# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

from batman import _rsky

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from .kepler import KeplerOp, get_sky_coords


class TestKepler(utt.InferShapeTester):

    def setUp(self):
        super(TestKepler, self).setUp()
        self.op_class = KeplerOp
        self.op = KeplerOp()

    def test_edge(self):
        E = np.array([0.0, 2*np.pi, -226.2])
        e = np.ones_like(E)
        M = E - e * np.sin(E)

        M_t = tt.vector()
        e_t = tt.vector()
        func = theano.function([M_t, e_t], self.op(M_t, e_t))
        E0 = func(M, e)

        assert np.all(np.isfinite(E0))
        utt.assert_allclose(E, E0)

    def test_solver(self):
        e = np.linspace(0, 1, 500)
        E = np.linspace(-300, 300, 1001)
        e = e[None, :] + np.zeros((len(E), len(e)))
        E = E[:, None] + np.zeros_like(e)
        M = E - e * np.sin(E)

        M_t = tt.matrix()
        e_t = tt.matrix()
        func = theano.function([M_t, e_t], self.op(M_t, e_t))
        E0 = func(M, e)
        utt.assert_allclose(E, E0)

    def test_infer_shape(self):
        np.random.seed(42)
        M = tt.vector()
        e = tt.vector()
        M_val = np.linspace(-10, 10, 50)
        e_val = np.random.uniform(0, 0.9, len(M_val))
        self._compile_and_check([M, e],
                                [self.op(M, e)],
                                [M_val, e_val],
                                self.op_class)

    def test_grad(self):
        np.random.seed(42)
        M_val = np.linspace(-10, 10, 50)
        e_val = np.random.uniform(0, 0.9, len(M_val))
        utt.verify_grad(self.op, [M_val, e_val])


def test_sky_coords():
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

    func = theano.function([], get_sky_coords(
        period[:, None],
        t0[:, None],
        e[:, None],
        omega[:, None],
        incl[:, None],
        t[None, :], tol=1e-7)
    )
    r = np.sqrt(np.sum(func()[:2]**2, axis=0))
    utt.assert_allclose(r_batman[m], r[m], atol=1e-6)
