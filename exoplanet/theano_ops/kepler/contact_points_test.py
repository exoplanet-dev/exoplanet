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

        f_expect = np.array([
            4.599646890798966,
            4.601763724711544,
            4.623084175008057,
            4.625231785287103,
        ])
        E_expect = 2 * np.arctan(np.sqrt((1-e)/(1+e))*np.tan(0.5*f_expect))
        M_expect = E_expect - e * np.sin(E_expect)

        M_calc = theano.function([], self.op(a, e, w, i, r, R))()

        utt.assert_allclose(M_expect, M_calc)

    def test_infer_shape(self):
        np.random.seed(42)
        args = [tt.vector() for i in range(6)]
        vals = [np.random.rand(50) for i in range(6)]
        self._compile_and_check(args,
                                self.op(*args),
                                vals,
                                self.op_class)
