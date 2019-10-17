# -*- coding: utf-8 -*-

__all__ = ["SolveOp"]

import theano
import theano.tensor as tt
from theano import gof

from .base_op import CeleriteBaseOp
from .solve_rev import SolveRevOp


class SolveOp(CeleriteBaseOp):

    func_file = "./solve.cc"
    func_name = "APPLY_SPECIFIC(solve)"
    num_input = 5
    output_ndim = (2, 2, 2)

    def __init__(self, J=-1, n_rhs=-1):
        self.grad_op = SolveRevOp(J=J, n_rhs=n_rhs)
        super(SolveOp, self).__init__(J=J, n_rhs=n_rhs)

    def make_node(self, *args):
        in_args = [tt.as_tensor_variable(a) for a in args]
        out_args = [
            in_args[-1].type(),
            tt.matrix(dtype=theano.config.floatX).type(),
            tt.matrix(dtype=theano.config.floatX).type(),
        ]
        return gof.Apply(self, in_args, out_args)

    def grad(self, inputs, gradients):
        U, P, d, W, Y = inputs
        Z, F, G = self(*inputs)
        bZ, bF, bG = gradients
        if isinstance(bZ.type, theano.gradient.DisconnectedType):
            return [None] * 5
        if not isinstance(bF.type, theano.gradient.DisconnectedType):
            raise ValueError("can't propagate gradients wrt F")
        if not isinstance(bG.type, theano.gradient.DisconnectedType):
            raise ValueError("can't propagate gradients wrt G")
        return self.grad_op(U, P, d, W, Z, F, G, bZ)
