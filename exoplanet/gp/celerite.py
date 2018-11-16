# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GP"]

import numpy as np
import theano.tensor as tt

from ..theano_ops.celerite.solve import SolveOp
from ..theano_ops.celerite.factor import FactorOp


class GP(object):

    def __init__(self, kernel, x, diag, J=-1):
        self.kernel = kernel
        self.J = J
        self.x = x
        self.diag = diag
        self.a, self.U, self.V, self.P = self.kernel.get_celerite_matrices(
            self.x, self.diag)
        self.factor_op = FactorOp(J=self.J)
        self.d, self.W, _ = self.factor_op(self.a, self.U, self.V, self.P)

        self.vector_solve_op = SolveOp(J=self.J, n_rhs=1)
        self.general_solve_op = SolveOp(J=self.J)

    def log_likelihood(self, y):
        self.y = y
        self.z, _, _ = self.vector_solve_op(self.U, self.P, self.d, self.W,
                                            tt.reshape(self.y,
                                                       (self.y.size, 1)))
        loglike = -0.5 * tt.sum(self.y * self.z[:, 0] + tt.log(self.d))
        loglike -= 0.5 * self.y.size * np.log(2*np.pi)
        return loglike

    def apply_inverse(self, rhs):
        return self.general_solve_op(self.U, self.P, self.d, self.W, rhs)[0]

    def predict(self, t=None):
        if t is None:
            return self.y - self.diag * self.z[:, 0]
        Ks = self.kernel.value(t[:, None] - self.x[None, :])
        return tt.dot(Ks, self.z)
