# -*- coding: utf-8 -*-

from __future__ import division, print_function

import pytest
import numpy as np

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
            [c, r, t, n, aome2, e, sinw, cosw, sini, cosi, dt],
            self.op(c, r, t, n, aome2, e, sinw, cosw, sini, cosi, dt)[0],
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
            [c, r, t, n, aome2, e, sinw, cosw, sini, cosi, dt],
            [
                c_val,
                r_val,
                t_val,
                n_val,
                aome2_val,
                e_val,
                sinw_val,
                cosw_val,
                sini_val,
                cosi_val,
                dt_val,
            ],
        )

    def test_basic(self):
        f, _, in_args = self.get_args()
        out = f(*in_args)
        utt.assert_allclose(0.0, out[0])
        utt.assert_allclose(0.0, out[-1])

    def test_infer_shape(self):
        f, args, arg_vals = self.get_args()
        self._compile_and_check(args, self.op(*args), arg_vals, self.op_class)

    def test_grad(self):
        _, args, in_args = self.get_args()
        func = lambda *args: self.op(*(list(args) + in_args[2:]))[0]  # NOQA
        utt.verify_grad(func, in_args[:2])
