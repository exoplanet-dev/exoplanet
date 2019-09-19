# -*- coding: utf-8 -*-

__all__ = ["GetClRevOp"]

import theano.tensor as tt
from theano import gof

from .base_op import StarryBaseOp


class GetClRevOp(StarryBaseOp):

    __props__ = ()
    func_file = "./get_cl_rev.cc"
    func_name = "APPLY_SPECIFIC(get_cl_rev)"

    def make_node(self, bc):
        return gof.Apply(self, [tt.as_tensor_variable(bc)], [bc.type()])

    def infer_shape(self, node, shapes):
        return (shapes[0],)
