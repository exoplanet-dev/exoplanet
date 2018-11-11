# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["LimbDarkOp"]

from theano import gof
import theano.tensor as tt

from .base_op import StarryBaseOp
from .limbdark_rev import LimbDarkRevOp


class LimbDarkOp(StarryBaseOp):

    __props__ = ()
    func_file = "./limbdark.cc"
    func_name = "APPLY_SPECIFIC(limbdark)"

    def __init__(self):
        self.grad_op = LimbDarkRevOp()
        super(LimbDarkOp, self).__init__()

    def make_node(self, c, b, r):
        in_args = [
            tt.as_tensor_variable(c),
            tt.as_tensor_variable(b),
            tt.as_tensor_variable(r),
        ]
        out_args = [in_args[1].type()]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        return shapes[1],

    def grad(self, inputs, gradients):
        c, b, r = inputs
        bf, = gradients
        return self.grad_op(c, b, r, bf)

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
