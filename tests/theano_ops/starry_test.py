# -*- coding: utf-8 -*-

import pickle

import numpy as np
import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from exoplanet.theano_ops.driver import SimpleLimbDark
from exoplanet.theano_ops.starry import (
    GetCl,
    GetClRev,
    LimbDark,
    RadiusFromOccArea,
)


class TestGetCl(utt.InferShapeTester):
    def setUp(self):
        super(TestGetCl, self).setUp()
        self.op_class = GetCl
        self.op = GetCl()

    def test_basic(self):
        x = tt.dvector()
        f = theano.function([x], self.op(x))

        inp = np.array([-1, 0.3, 0.2, 0.5])
        out = f(inp)

        utt.assert_allclose(np.array([-0.85, 2.5, -0.425, 0.1]), out)

    def test_infer_shape(self):
        x = tt.dvector()
        self._compile_and_check(
            [x], [self.op(x)], [np.asarray(np.random.rand(5))], self.op_class
        )

    def test_grad(self):
        utt.verify_grad(self.op, [np.array([-1, 0.3, 0.2, 0.5])])


class TestGetClRev(utt.InferShapeTester):
    def setUp(self):
        super(TestGetClRev, self).setUp()
        self.op_class = GetClRev
        self.op = GetClRev()

    def test_basic(self):
        x = tt.dvector()
        f = theano.function([x], self.op(x))

        inp = np.array([-1, 0.3, 0.2, 0.5])
        out = f(inp)

        utt.assert_allclose(np.array([0, 1.3, 2.05, 3.53]), out)

    def test_infer_shape(self):
        x = tt.dvector()
        self._compile_and_check(
            [x], [self.op(x)], [np.asarray(np.random.rand(5))], self.op_class
        )


class TestLimbDark(utt.InferShapeTester):
    def setUp(self):
        super(TestLimbDark, self).setUp()
        self.op_class = LimbDark
        self.op = LimbDark()

    def get_args(self):
        c = tt.vector()
        b = tt.vector()
        r = tt.vector()
        los = tt.vector()
        f = theano.function([c, b, r, los], self.op(c, b, r, los)[0])

        c_val = np.array([-0.85, 2.5, -0.425, 0.1])
        b_val = np.linspace(-1.5, 1.5, 100)
        r_val = 0.1 + np.zeros_like(b_val)
        los_val = np.ones_like(b_val)

        return f, [c, b, r, los], [c_val, b_val, r_val, los_val]

    def test_basic(self):
        f, _, in_args = self.get_args()
        out = f(*in_args)
        utt.assert_allclose(0.0, out[0])
        utt.assert_allclose(0.0, out[-1])

    def test_los(self):
        f, _, in_args = self.get_args()
        in_args[-1] = -np.ones_like(in_args[-1])
        out = f(*in_args)
        utt.assert_allclose(0.0, out)

    def test_infer_shape(self):
        f, args, arg_vals = self.get_args()
        self._compile_and_check(args, self.op(*args), arg_vals, self.op_class)

    def test_grad(self):
        _, _, in_args = self.get_args()
        func = lambda *args: self.op(*args)[0]  # NOQA
        utt.verify_grad(func, in_args)

    def test_pickle(self):
        f, _, in_args = self.get_args()
        data = pickle.dumps(self.op, -1)
        new_op = pickle.loads(data)
        utt.assert_allclose(f(*in_args), new_op(*in_args)[0].eval())


class TestRadiusFromOccArea(utt.InferShapeTester):
    def setUp(self):
        super(TestRadiusFromOccArea, self).setUp()
        self.op_class = RadiusFromOccArea
        self.op = RadiusFromOccArea()

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
        expect = -LimbDark()(
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


def test_simple():
    np.random.seed(4502934)
    r = np.random.uniform(0, 1.0, 50)
    b = np.random.uniform(-1.2, 1.2, len(r))

    u = np.array([0.5, 0.3])
    cl = GetCl()(tt.as_tensor_variable(u)).eval()
    cl /= np.pi * (cl[0] + 2 * cl[1] / 3)
    f0 = LimbDark()(cl, b, r, np.ones_like(b))[0].eval()

    ld = SimpleLimbDark()
    assert np.allclose(ld.apply(b, r), 0.0)

    ld.set_u(u)
    assert np.allclose(ld.apply(b, r), f0)


def test_simple_pickle():
    np.random.seed(4502934)
    r = np.random.uniform(0, 1.0, 50)
    b = np.random.uniform(-1.2, 1.2, len(r))
    u = np.array([0.5, 0.3])

    ld = SimpleLimbDark()
    ld.set_u(u)
    f0 = ld.apply(b, r)

    data = pickle.dumps(ld, -1)
    ld2 = pickle.loads(data)
    assert np.allclose(ld2.apply(b, r), f0)
