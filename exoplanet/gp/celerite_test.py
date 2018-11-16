# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np

import theano
import theano.tensor as tt

from . import terms
from .celerite import GP


def test_broadcast_dim():
    logS0 = tt.scalar()
    logw0 = tt.scalar()
    logQ = tt.scalar()
    kernel = terms.SHOTerm(S0=tt.exp(logS0), w0=tt.exp(logw0), Q=tt.exp(logQ))

    x = tt.vector()
    y = tt.vector()
    diag = tt.vector()
    gp = GP(kernel, x, diag, J=2)
    loglike = gp.log_likelihood(y)

    args = [logS0, logw0, logQ, x, y, diag]
    grad = theano.function(args, theano.grad(loglike, args))

    np.random.seed(42)
    N = 50
    x = np.sort(10 * np.random.rand(N))
    y = np.sin(x)
    diag = np.random.rand(N)
    grad(-5.0, -2.0, 1.0, x, y, diag)
