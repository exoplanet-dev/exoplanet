# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["FactorOp"]

import theano
from theano import gof
import theano.tensor as tt

from .base_op import CeleriteBaseOp
from .factor_rev import FactorRevOp


class FactorOp(CeleriteBaseOp):

    func_file = "./factor.cc"
    func_name = "APPLY_SPECIFIC(factor)"
    num_input = 4
    output_ndim = (1, 2, 2)

    def __init__(self, J=-1):
        self.grad_op = FactorRevOp(J=J)
        super(FactorOp, self).__init__(J=J)

    def make_node(self, *args):
        in_args, out_args = self._get_node_args(*args)
        out_args.append(tt.iscalar().type())
        return gof.Apply(self, in_args, out_args)

    def grad(self, inputs, gradients):
        a, U, V, P = inputs
        d, W, S, flag = self(*inputs)
        bd, bW, bS, bflag = gradients
        if isinstance(bd.type, theano.gradient.DisconnectedType):
            bd = tt.zeros_like(d)
        if isinstance(bW.type, theano.gradient.DisconnectedType):
            bW = tt.zeros_like(W)
        if not isinstance(bS.type, theano.gradient.DisconnectedType):
            raise ValueError("can't propagate gradients wrt S")
        return self.grad_op(U, P, d, W, S, bd, bW)
