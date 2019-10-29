# -*- coding: utf-8 -*-

__all__ = ["DiagDotOp"]

import theano
import theano.tensor as tt
from theano import gof

from .base_op import CeleriteBaseOp


class DiagDotOp(CeleriteBaseOp):

    func_file = "./diag_dot.cc"
    func_name = "APPLY_SPECIFIC(diag_dot)"

    def make_node(self, *args):
        in_args = [tt.as_tensor_variable(a) for a in args]
        out_args = [tt.vector(dtype=theano.config.floatX).type()]
        return gof.Apply(self, in_args, out_args)

    def grad(self, inputs, gradients):
        A, B = inputs
        return (
            gradients[0][:, None] * tt.transpose(B),
            gradients[1][None, :] * tt.transpose(A),
        )
