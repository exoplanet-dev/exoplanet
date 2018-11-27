# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from .solver import KeplerOp


class TestKeplerSolver(utt.InferShapeTester):

    def setUp(self):
        super(TestKeplerSolver, self).setUp()
        self.op_class = KeplerOp
        self.op = KeplerOp()

    def _get_M_and_f(self, e, E):
        M = E - e * np.sin(E)
        f = 2 * np.arctan(np.sqrt((1+e)/(1-e)) * np.tan(0.5*E))
        return M, f

    def test_edge(self):
        E = np.array([0.0, 2*np.pi, -226.2])
        e = (1 - 1e-6) * np.ones_like(E)
        M, f = self._get_M_and_f(e, E)

        M_t = tt.vector()
        e_t = tt.vector()
        func = theano.function([M_t, e_t], self.op(M_t, e_t))
        E0, f0 = func(M, e)

        assert np.all(np.isfinite(E0))
        utt.assert_allclose(E, E0)
        assert np.all(np.isfinite(f0))
        utt.assert_allclose(f, f0)

    def test_solver(self):
        e = np.linspace(0, 1, 500)[:-1]
        E = np.linspace(-300, 300, 1001)
        e = e[None, :] + np.zeros((len(E), len(e)))
        E = E[:, None] + np.zeros_like(e)
        M, f = self._get_M_and_f(e, E)

        M_t = tt.matrix()
        e_t = tt.matrix()
        func = theano.function([M_t, e_t], self.op(M_t, e_t))
        E0, f0 = func(M, e)

        utt.assert_allclose(E, E0)
        utt.assert_allclose(f, f0)

    def test_infer_shape(self):
        np.random.seed(42)
        M = tt.vector()
        e = tt.vector()
        M_val = np.linspace(-10, 10, 50)
        e_val = np.random.uniform(0, 0.9, len(M_val))
        self._compile_and_check([M, e],
                                self.op(M, e),
                                [M_val, e_val],
                                self.op_class)

    def test_grad(self):
        np.random.seed(42)
        M_val = np.linspace(-10, 10, 50)
        e_val = np.random.uniform(0, 0.8, len(M_val))

        a = lambda *args: self.op(*args)[0]  # NOQA
        utt.verify_grad(a, [M_val, e_val], eps=1e-8)

        a = lambda *args: self.op(*args)[1]  # NOQA
        utt.verify_grad(a, [M_val, e_val], eps=1e-8)
