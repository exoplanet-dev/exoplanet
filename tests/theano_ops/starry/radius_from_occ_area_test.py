# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from exoplanet.theano_ops.starry.limbdark import LimbDarkOp
from exoplanet.theano_ops.starry.radius_from_occ_area import (
    RadiusFromOccAreaOp,
)


class TestRadiusFromOccArea(utt.InferShapeTester):
    def setUp(self):
        super(TestRadiusFromOccArea, self).setUp()
        self.op_class = RadiusFromOccAreaOp
        self.op = RadiusFromOccAreaOp()

    def get_args(self):
        delta = tt.vector()
        b = tt.vector()
        f = theano.function([delta, b], self.op(delta, b))

        delta_val = np.linspace(0.01, 0.99, 100)
        b_val = np.linspace(0.01, 1.5, len(delta_val))

        return f, [delta, b], [delta_val, b_val]

    def test_basic(self):
        f, t_args, v_args = self.get_args()
        r = f(*v_args)
        expect = -LimbDarkOp()(
            np.array([1.0 / np.pi, 0.0]), v_args[1], r, np.ones_like(r)
        )[0].eval()
        utt.assert_allclose(expect, v_args[0])

    def test_infer_shape(self):
        f, args, arg_vals = self.get_args()
        self._compile_and_check(
            args, [self.op(*args)], arg_vals, self.op_class
        )

    def test_grad(self):
        func, t_args, v_args = self.get_args()
        g = theano.function(
            t_args, theano.grad(tt.sum(self.op(*t_args)), t_args)
        )(*v_args)
        eps = 1.234e-6
        for n in range(len(t_args)):
            v_args[n] += eps
            plus = func(*v_args)
            v_args[n] -= 2 * eps
            minus = func(*v_args)
            v_args[n] += eps

            est = 0.5 * (plus - minus) / eps
            utt.assert_allclose(est, g[n], atol=2 * eps)
