# -*- coding: utf-8 -*-

import numpy as np
import pytest
import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from .integrated_limbdark import IntegratedLimbDarkOp


class TestIntegratedLimbDark(utt.InferShapeTester):
    def setUp(self):
        super(TestIntegratedLimbDark, self).setUp()
        self.op_class = IntegratedLimbDarkOp
        self.op = IntegratedLimbDarkOp(Nc=3)

    def get_args(self):
        np.random.seed(1234)

        c = tt.dvector()
        r = tt.dvector()
        t = tt.dvector()
        n = tt.dvector()
        aome2 = tt.dvector()
        e = tt.dvector()
        sinw = tt.dvector()
        cosw = tt.dvector()
        sini = tt.dvector()
        cosi = tt.dvector()
        dt = tt.dvector()
        f = theano.function(
            [c, r, t, dt, n, aome2, sini, cosi, e, sinw, cosw],
            self.op(c, r, t, dt, n, aome2, sini, cosi, e, sinw, cosw)[0],
        )

        c_val = np.array([0.14691226, 0.25709645, -0.01836403])
        r_val = np.random.uniform(0.01, 0.2, 1000)
        P_val = 35.0
        t_val = np.linspace(-0.5 * P_val, 0.5 * P_val, len(r_val))
        n_val = 2 * np.pi / P_val + np.zeros_like(r_val)
        e_val = np.linspace(0, 1.0, len(r_val) + 1)[:-1]
        aome2_val = 10 * (1 - e_val ** 2)
        w_val = np.linspace(-np.pi, np.pi, len(r_val))
        sinw_val = np.sin(w_val)
        cosw_val = np.cos(w_val)
        i_val = 0.5 * np.pi + np.linspace(-0.01, 0.01, len(r_val))
        sini_val = np.sin(i_val)
        cosi_val = np.cos(i_val)

        dt_val = np.random.uniform(0.01, 0.1, len(r_val))

        return (
            f,
            [c, r, t, dt, n, aome2, sini, cosi, e, sinw, cosw],
            [
                c_val,
                r_val,
                t_val,
                dt_val,
                n_val,
                aome2_val,
                sini_val,
                cosi_val,
                e_val,
                sinw_val,
                cosw_val,
            ],
        )

    @pytest.mark.skip(reason="Too slow")
    def test_basic(self):
        f, _, in_args = self.get_args()
        out = f(*in_args)
        utt.assert_allclose(0.0, out[0])
        utt.assert_allclose(0.0, out[-1])

    @pytest.mark.skip(reason="Too slow")
    def test_infer_shape(self):
        f, args, arg_vals = self.get_args()
        self._compile_and_check(args, self.op(*args), arg_vals, self.op_class)

    @pytest.mark.skip(reason="Too slow")
    def test_grad(self):
        _, args, in_args = self.get_args()

        # This is a hack because we don't compute gradients wrt texp
        func = lambda *args: self.op(  # NOQA
            *([args[0]] + [in_args[1]] + list(args[1:]))
        )[0]

        utt.verify_grad(func, [in_args[0]] + list(in_args[2:]), eps=1e-8)


class TestCircIntegratedLimbDark(utt.InferShapeTester):
    def setUp(self):
        super(TestCircIntegratedLimbDark, self).setUp()

        class _DummyClass(IntegratedLimbDarkOp):
            def __init__(self, *args, **kwargs):
                kwargs["circular"] = True
                super(_DummyClass, self).__init__(*args, **kwargs)

        self.op_class = _DummyClass
        self.op = _DummyClass(Nc=3)

    def get_args(self):
        np.random.seed(1234)

        c = tt.dvector()
        r = tt.dvector()
        t = tt.dvector()
        n = tt.dvector()
        aome2 = tt.dvector()
        sini = tt.dvector()
        cosi = tt.dvector()
        dt = tt.dvector()
        nothing1 = tt.dscalar()
        nothing2 = tt.dscalar()
        nothing3 = tt.dscalar()
        f = theano.function(
            [c, dt, t, r, n, aome2, sini, cosi, nothing1, nothing2, nothing3],
            self.op(
                c, dt, t, r, n, aome2, sini, cosi, nothing1, nothing2, nothing3
            )[0],
        )

        c_val = np.array([0.14691226, 0.25709645, -0.01836403])
        r_val = np.random.uniform(0.01, 0.2, 1000)
        P_val = 35.0
        t_val = np.linspace(-0.5 * P_val, 0.5 * P_val, len(r_val))
        n_val = 2 * np.pi / P_val + np.zeros_like(r_val)
        aome2_val = 10 + np.zeros_like(r_val)
        i_val = 0.5 * np.pi + np.linspace(-0.01, 0.01, len(r_val))
        sini_val = np.sin(i_val)
        cosi_val = np.cos(i_val)

        dt_val = np.random.uniform(0.01, 0.1, len(r_val))

        return (
            f,
            [c, dt, t, r, n, aome2, sini, cosi, nothing1, nothing2, nothing3],
            [
                c_val,
                dt_val,
                t_val,
                r_val,
                n_val,
                aome2_val,
                sini_val,
                cosi_val,
                0.0,
                0.0,
                0.0,
            ],
        )

    @pytest.mark.skip(reason="Too slow")
    def test_basic(self):
        f, _, in_args = self.get_args()
        out = f(*in_args)
        utt.assert_allclose(0.0, out[0])
        utt.assert_allclose(0.0, out[-1])

    @pytest.mark.skip(reason="Too slow")
    def test_infer_shape(self):
        f, args, arg_vals = self.get_args()
        self._compile_and_check(args, self.op(*args), arg_vals, self.op_class)

    @pytest.mark.skip(reason="Too slow")
    def test_grad(self):
        _, args, in_args = self.get_args()
        func = lambda *args: self.op(  # NOQA
            *([args[0]] + [in_args[1]] + list(args[1:]) + [0.0, 0.0, 0.0])
        )[0]
        utt.verify_grad(func, [in_args[0]] + list(in_args[2:-3]), eps=1e-8)
