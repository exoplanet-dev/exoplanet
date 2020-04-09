# -*- coding: utf-8 -*-

__all__ = ["DotLOp"]

import theano.tensor as tt
from theano import gof

from .base_op import CeleriteBaseOp


class DotLOp(CeleriteBaseOp):

    func_file = "./dot_l.cc"
    func_name = "APPLY_SPECIFIC(dot_l)"
    num_input = 5
    output_ndim = (2,)

    def make_node(self, *args):
        in_args = [tt.as_tensor_variable(a) for a in args]
        out_args = [in_args[-1].type()]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        return (shapes[-1],)
