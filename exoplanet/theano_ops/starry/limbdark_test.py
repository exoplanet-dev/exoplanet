# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from .limbdark import LimbDarkOp


class TestLimbDark(utt.InferShapeTester):

    def setUp(self):
        super(TestLimbDark, self).setUp()
        self.op_class = LimbDarkOp
        self.op = LimbDarkOp()

    def get_args(self):
        c = tt.vector()
        b = tt.vector()
        r = tt.vector()
        los = tt.vector()
        f = theano.function([c, b, r, los], self.op(c, b, r, los)[0])

        c_val = np.array([-0.85, 2.5, -0.425, 0.1])
        b_val = np.linspace(-1.5, 1.5, 100)
        r_val = 0.1 + np.zeros_like(b_val)
        los_val = -np.ones_like(b_val)

        return f, [c, b, r, los], [c_val, b_val, r_val, los_val]

    def test_basic(self):
        f, _, in_args = self.get_args()
        out = f(*in_args)
        utt.assert_allclose(0.0, out[0])
        utt.assert_allclose(0.0, out[-1])

    def test_los(self):
        f, _, in_args = self.get_args()
        in_args[-1] = np.ones_like(in_args[-1])
        out = f(*in_args)
        utt.assert_allclose(0.0, out)

    def test_infer_shape(self):
        f, args, arg_vals = self.get_args()
        self._compile_and_check(args,
                                self.op(*args),
                                arg_vals,
                                self.op_class)

    def test_grad(self):
        _, _, in_args = self.get_args()
        func = lambda *args: self.op(*args)[0]  # NOQA
        utt.verify_grad(func, in_args)
