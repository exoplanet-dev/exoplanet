# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GetClOp"]

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

    def infer_shape(self, node, shapes):
        return shapes[0],

    def grad(self, inputs, gradients):
        return self.grad_op(gradients[0]),

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
