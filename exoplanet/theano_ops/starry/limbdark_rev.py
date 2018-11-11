# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["LimbDarkRevOp"]

from theano import gof
import theano.tensor as tt

from .base_op import StarryBaseOp


class LimbDarkRevOp(StarryBaseOp):

    __props__ = ()
    func_file = "./limbdark_rev.cc"
    func_name = "APPLY_SPECIFIC(limbdark_rev)"

    def make_node(self, c, b, r, bf):
        in_args = [
            tt.as_tensor_variable(c),
            tt.as_tensor_variable(b),
            tt.as_tensor_variable(r),
            tt.as_tensor_variable(bf),
        ]
        out_args = [in_args[0].type(), in_args[1].type(), in_args[2].type()]
        return gof.Apply(self, in_args, out_args)

    def infer_shape(self, node, shapes):
        return shapes[:3]
