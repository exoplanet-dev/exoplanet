# -*- coding: utf-8 -*-

__all__ = ["ConditionalMeanOp"]

import theano.tensor as tt
from theano import gof

from .base_op import CeleriteBaseOp


class ConditionalMeanOp(CeleriteBaseOp):

    func_file = "./conditional_mean.cc"
    func_name = "APPLY_SPECIFIC(conditional_mean)"
    num_input = 7
    output_ndim = (1,)

    def make_node(self, *args):
        in_args = [tt.as_tensor_variable(a) for a in args]
        out_args = [in_args[3].type()]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        return (shapes[-1],)
