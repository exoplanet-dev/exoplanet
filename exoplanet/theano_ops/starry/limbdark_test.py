# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from .limbdark import LimbDarkOp
from .limbdark_rev import LimbDarkRevOp


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
        f = theano.function([c, b, r, los], self.op(c, b, r, los))

        c_val = np.array([-0.85, 2.5, -0.425, 0.1])
        b_val = np.linspace(-1.5, 1.5, 100)
        r_val = 0.1 + np.zeros_like(b_val)
        los_val = -np.ones_like(b_val)

        return f, [c, b, r, los], [c_val, b_val, r_val, los_val]

    def test_basic(self):
        f, _, in_args = self.get_args()
        out = f(*in_args)
        utt.assert_allclose(1.0, out[0])
        utt.assert_allclose(1.0, out[-1])

    def test_infer_shape(self):
        f, args, arg_vals = self.get_args()
        self._compile_and_check(args,
                                [self.op(*args)],
                                arg_vals,
                                self.op_class)

    def test_grad(self):
        _, _, in_args = self.get_args()
        utt.verify_grad(self.op, in_args)


class TestLimbDarkRev(utt.InferShapeTester):

    def setUp(self):
        super(TestLimbDarkRev, self).setUp()
        self.op_class = LimbDarkRevOp
        self.op = LimbDarkRevOp()

    def get_args(self):
        c = tt.vector()
        b = tt.vector()
        r = tt.vector()
        los = tt.vector()
        bf = tt.vector()
        f = theano.function([c, b, r, los, bf], self.op(c, b, r, los, bf))

        c_val = np.array([-0.85, 2.5, -0.425, 0.1])
        b_val = np.linspace(-1.5, 1.5, 100)
        r_val = 0.1 + np.zeros_like(b_val)
        los_val = -np.ones_like(b_val)
        bf_val = np.ones_like(b_val)

        return f, [c, b, r, los, bf], [c_val, b_val, r_val, los_val, bf_val]

    def test_basic(self):
        f, _, in_args = self.get_args()
        f(*in_args)

    def test_infer_shape(self):
        f, args, arg_vals = self.get_args()
        self._compile_and_check(args,
                                self.op(*args),
                                arg_vals,
                                self.op_class)
