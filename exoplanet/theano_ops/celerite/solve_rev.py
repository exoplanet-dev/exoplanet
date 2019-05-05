# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["SolveRevOp"]

from theano import gof
import theano.tensor as tt
from .base_op import CeleriteBaseOp


class SolveRevOp(CeleriteBaseOp):

    func_file = "./solve_rev.cc"
    func_name = "APPLY_SPECIFIC(solve_rev)"

    def make_node(self, *args):
        in_args = [tt.as_tensor_variable(a) for a in args]
        out_args = [a.type() for a in args[:5]]
        return gof.Apply(self, in_args, out_args)
