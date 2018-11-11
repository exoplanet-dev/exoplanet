# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from .get_cl import GetClOp
from .get_cl_rev import GetClRevOp


class TestGetCl(utt.InferShapeTester):

    def setUp(self):
        super(TestGetCl, self).setUp()
        self.op_class = GetClOp
        self.op = GetClOp()

    def test_basic(self):
        x = tt.vector()
        f = theano.function([x], self.op(x))

        inp = np.array([-1, 0.3, 0.2, 0.5])
        out = f(inp)

        utt.assert_allclose(np.array([-0.85, 2.5, -0.425, 0.1]), out)

    def test_infer_shape(self):
        x = tt.vector()
        self._compile_and_check([x],
                                [self.op(x)],
                                [np.asarray(np.random.rand(5))],
                                self.op_class)

    def test_grad(self):
        utt.verify_grad(self.op, [np.array([-1, 0.3, 0.2, 0.5])])


class TestGetClRev(utt.InferShapeTester):

    def setUp(self):
        super(TestGetClRev, self).setUp()
        self.op_class = GetClRevOp
        self.op = GetClRevOp()

    def test_basic(self):
        x = tt.vector()
        f = theano.function([x], self.op(x))

        inp = np.array([-1, 0.3, 0.2, 0.5])
        out = f(inp)

        utt.assert_allclose(np.array([0, 1.3, 2.05, 3.53]), out)

    def test_infer_shape(self):
        x = tt.vector()
        self._compile_and_check([x],
                                [self.op(x)],
                                [np.asarray(np.random.rand(5))],
                                self.op_class)
