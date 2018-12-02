# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from .contact_points import ContactPointsOp


class TestContactPoints(utt.InferShapeTester):

    def setUp(self):
        super(TestContactPoints, self).setUp()
        self.op_class = ContactPointsOp
        self.op = ContactPointsOp()

    def test_basic(self):
        a = 100.0
        e = 0.3
        w = 0.1
        i = 0.5*np.pi
        r = 0.1
        R = 1.1

        f_expect = [
            -0.012742089585723981,
            -0.010625255673145872,
            0.010695194623367321,
            0.012842804902413185
        ]

        f_calc = self.op(a, e, w, i, r, R).eval()

        utt.assert_allclose(f_expect, f_calc)

    # def test_infer_shape(self):
    #     np.random.seed(42)
    #     args = [tt.vector() for i in range(6)]
    #     vals = [np.random.rand(50) for i in range(6)]
    #     self._compile_and_check(args,
    #                             self.op(*args),
    #                             vals,
    #                             self.op_class)
