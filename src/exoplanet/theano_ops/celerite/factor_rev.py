# -*- coding: utf-8 -*-

__all__ = ["FactorRevOp"]

import theano.tensor as tt
from theano import gof

from .base_op import CeleriteBaseOp


class FactorRevOp(CeleriteBaseOp):

    func_file = "./factor_rev.cc"
    func_name = "APPLY_SPECIFIC(factor_rev)"

    def __init__(self, J=-1):
        super(FactorRevOp, self).__init__(J=J)

    def make_node(self, *args):
        in_args = [tt.as_tensor_variable(a) for a in args]
        out_args = [
            in_args[2].type(),
            in_args[0].type(),
            in_args[3].type(),
            in_args[1].type(),
        ]
        return gof.Apply(self, in_args, out_args)
