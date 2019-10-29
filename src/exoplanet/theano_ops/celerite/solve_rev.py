# -*- coding: utf-8 -*-

__all__ = ["SolveRevOp"]

import theano.tensor as tt
from theano import gof

from .base_op import CeleriteBaseOp


class SolveRevOp(CeleriteBaseOp):

    func_file = "./solve_rev.cc"
    func_name = "APPLY_SPECIFIC(solve_rev)"

    def make_node(self, *args):
        in_args = [tt.as_tensor_variable(a) for a in args]
        out_args = [a.type() for a in args[:5]]
        return gof.Apply(self, in_args, out_args)
