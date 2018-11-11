# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["log_likelihood"]

import numpy as np
import theano.tensor as tt

from .solve import SolveOp
from .factor import FactorOp


def log_likelihood(kernel, diag, x, y, J=-1):
    a, U, V, P = kernel.get_celerite_matrices(x, diag)
    factor_op = FactorOp(J=J)
    d, W, _ = factor_op(a, U, V, P)
    solve_op = SolveOp(J=J, n_rhs=1)
    z, _, _ = solve_op(U, P, d, W, tt.reshape(y, (y.size, 1)))
    loglike = -0.5 * tt.sum(y * z[:, 0] + tt.log(d))
    loglike -= 0.5 * y.size * np.log(2*np.pi)
    return loglike
