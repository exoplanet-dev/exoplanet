# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from ...gp import terms
from .factor import FactorOp


class TestFactor(utt.InferShapeTester):

    def setUp(self):
        super(TestFactor, self).setUp()
        self.op_class = FactorOp
        self.op = FactorOp()

    def get_celerite_matrices(self):
        ar = tt.vector()
        cr = tt.vector()
        kernel = terms.RealTerm(a=ar, c=cr)

        ac = tt.vector()
        bc = tt.vector()
        cc = tt.vector()
        dc = tt.vector()
        kernel += terms.ComplexTerm(a=ac, b=bc, c=cc, d=dc)

        x = tt.vector()
        diag = tt.vector()
        matrices = kernel.get_celerite_matrices(x, diag)
        args = [ar, cr, ac, bc, cc, dc, x, diag]
        M = theano.function(args, matrices)
        K = theano.function(args, kernel.to_dense(x, diag))

        np.random.seed(42)
        N = 15
        vals = [
            np.array([1.5, 0.1, 0.6, 0.3, 0.8, 0.7]),
            np.array([1.0, 0.3, 0.05, 0.01, 0.1, 0.2]),
            np.array([1.0, 2.0]),
            np.array([0.1, 0.5]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.sort(np.random.rand(N)),
            np.random.uniform(0.1, 0.5, N),
        ]

        return M(*vals), K(*vals)

    def get_args(self):
        args = [
            tt.vector(),
            tt.matrix(),
            tt.matrix(),
            tt.matrix(),
        ]
        f = theano.function(args, self.op(*args))
        vals, K = self.get_celerite_matrices()
        return f, args, vals, K

    def test_grad(self):
        eps = 1e-5
        f, args, vals, _ = self.get_args()
        output0 = f(*vals)

        # Go through and backpropagate all of the gradients from the outputs
        grad0 = []
        for i in range(len(output0) - 2):
            grad0.append([])
            for j in range(output0[i].size):
                ind = np.unravel_index(j, output0[i].shape)

                g = theano.function(
                    args, theano.grad(self.op(*args)[i][ind], args))
                grad0[-1].append(g(*vals))

        # Loop over each input and numerically compute the gradient
        for k in range(len(vals)):
            for l in range(vals[k].size):
                inner = np.unravel_index(l, vals[k].shape)
                vals[k][inner] += eps
                plus = f(*vals)
                vals[k][inner] -= 2*eps
                minus = f(*vals)
                vals[k][inner] += eps

                # Compare to the backpropagated gradients
                for i in range(len(output0) - 2):
                    for j in range(output0[i].size):
                        ind = np.unravel_index(j, output0[i].shape)
                        delta = 0.5 * (plus[i][ind] - minus[i][ind]) / eps
                        ref = grad0[i][j][k][inner]
                        assert np.abs(delta - ref) < 2*eps
