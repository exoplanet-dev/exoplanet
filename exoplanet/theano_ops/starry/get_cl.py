# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GetClOp"]

import theano
from theano import gof
import theano.tensor as tt

from .base_op import StarryBaseOp
from .get_cl_rev import GetClRevOp


class GetClOp(StarryBaseOp):

    __props__ = ()
    func_file = "./get_cl.cc"
    func_name = "APPLY_SPECIFIC(get_cl)"
    num_input = 1

    def __init__(self):
        self.grad_op = GetClRevOp()
        super(GetClOp, self).__init__()

    def make_node(self, arg):
        return gof.Apply(self, [tt.as_tensor_variable(arg)], [arg.type()])

    def grad(self, inputs, gradients):
        bc, = gradients
        if isinstance(bc.type, theano.gradient.DisconnectedType):
            return tt.zeros_like(inputs[0]),
        return self.grad_op(bc),
