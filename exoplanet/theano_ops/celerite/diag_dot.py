# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["DiagDotOp"]

import theano.tensor as tt

from .base_op import CeleriteBaseOp


class DiagDotOp(CeleriteBaseOp):

    func_file = "./diag_dot.cc"
    func_name = "APPLY_SPECIFIC(diag_dot)"
    num_input = 2
    output_ndim = (1, )

    def grad(self, inputs, gradients):
        A, B = inputs
        return (
            gradients[0][:, None] * tt.transpose(B),
            gradients[1][None, :] * tt.transpose(A)
        )
