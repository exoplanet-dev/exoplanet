# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from .contact_points import ContactPointsOp, CircularContactPointsOp


class TestContactPoints(utt.InferShapeTester):

    def setUp(self):
        super(TestContactPoints, self).setUp()
        self.op_class = ContactPointsOp
        self.op = ContactPointsOp()

    def test_infer_shape(self):
        np.random.seed(42)
        args = [tt.dvector() for i in range(6)]
        vals = [np.random.rand(50) for i in range(6)]
        self._compile_and_check(args,
                                self.op(*args),
                                vals,
                                self.op_class)

    def test_basic(self):
        a = np.float64(100.0)
        e = np.float64(0.3)
        w = np.float64(0.1)
        i = np.float64(0.5*np.pi)
        r = np.float64(0.1)
        R = np.float64(1.1)

        M_expect = np.array([0.88452506, 0.8863776, 0.90490204, 0.90675455])
        M_calc = theano.function([], self.op(a, e, w, i, r, R))()

        utt.assert_allclose(M_calc, M_expect)


class TestCircularContactPoints(utt.InferShapeTester):

    def setUp(self):
        super(TestCircularContactPoints, self).setUp()
        self.op_class = CircularContactPointsOp
        self.op = CircularContactPointsOp()

    def test_infer_shape(self):
        np.random.seed(42)
        args = [tt.dvector() for i in range(4)]
        vals = [np.random.rand(50) for i in range(4)]
        self._compile_and_check(args,
                                self.op(*args),
                                vals,
                                self.op_class)

    def test_basic(self):
        a = np.float64(100.0)
        e = np.float64(0.0)
        w = np.float64(0.0)
        i = np.float64(0.5*np.pi)
        r = np.float64(0.1)
        R = np.float64(1.1)

        M_circ = theano.function([], self.op(a, i, r, R))()
        M_gen = theano.function([], ContactPointsOp()(a, e, w, i, r, R))()

        utt.assert_allclose(M_circ, M_gen)
